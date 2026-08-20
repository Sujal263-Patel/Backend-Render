"""
The harness is the structured orchestration layer around the raw
model calls: it defines typed request/response contracts, retries
transient failures, catches and classifies errors per-stage, runs
guardrails at the right points, and always returns a structured
PipelineResult (never a bare string) with per-stage timing attached.

Pipeline stages:
  1. transcribe (STT)                     -> retried on transient network errors
  2. input guardrail                      -> hard stop, no retry
  3. retrieve (vector DB search)          -> no retry needed (deterministic, in-memory)
  4. retrieval guardrail                  -> hard stop if nothing relevant
  5. generate (LLM or extractive)         -> retried once on transient failure
  6. output guardrail (grounding check)   -> downgrades to refusal if ungrounded
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import stt
import generation
import guardrails as gr
from vectorstore import HybridRetriever, RetrievedChunk


class PipelineStatus(str, Enum):
    OK = "ok"
    REFUSED_UNSAFE_INPUT = "refused_unsafe_input"
    REFUSED_OFF_TOPIC = "refused_off_topic"
    REFUSED_UNGROUNDED = "refused_ungrounded"
    ERROR = "error"


@dataclass
class StageTiming:
    stage: str
    ms: float


class StageException(Exception):
    def __init__(self, stage: str, ms: float, orig_err: Exception):
        super().__init__(str(orig_err))
        self.stage = stage
        self.ms = ms
        self.orig_err = orig_err


@dataclass
class PipelineResult:
    status: PipelineStatus
    query_text: str = ""
    answer: str = ""
    retrieved: list[dict] = field(default_factory=list)
    guard_events: list[dict] = field(default_factory=list)
    timings_ms: list[StageTiming] = field(default_factory=list)
    total_ms: float = 0.0
    error: str | None = None
    error_stage: str | None = None
    error_message: str | None = None
    generation_method: str | None = None

    def to_dict(self):
        d = asdict(self)
        d["status"] = self.status.value
        return d


class TransientError(Exception):
    """Raised for retryable failures (network hiccups etc.)."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.2, min=0.2, max=2),
    retry=retry_if_exception_type(TransientError),
    reraise=True,
)
def _transcribe_with_retry(audio_bytes: bytes, filename: str) -> dict:
    try:
        return stt.transcribe(audio_bytes, filename)
    except stt.STTError as e:
        msg = str(e).lower()
        if "request failed" in msg or "timeout" in msg:
            raise TransientError(str(e)) from e
        raise


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.3, min=0.3, max=2),
    retry=retry_if_exception_type(TransientError),
    reraise=True,
)
def _generate_with_retry(query: str, context_chunks: list[str]) -> dict:
    try:
        return generation.generate_answer(query, context_chunks)
    except generation.GenerationError as e:
        msg = str(e).lower()
        if "timeout" in msg or "429" in msg or "503" in msg:
            raise TransientError(str(e)) from e
        raise


def _timed(label: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        return result, StageTiming(stage=label, ms=ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        raise StageException(label, ms, e) from e


class RAGHarness:
    def __init__(self, retriever: HybridRetriever, top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k

    def run_from_text(self, query_text: str) -> PipelineResult:
        """Skip STT — used for text queries and latency benchmarking."""
        return self._run(query_text=query_text, timings=[])

    def run_from_audio(self, audio_bytes: bytes, filename: str = "audio.wav") -> PipelineResult:
        timings: list[StageTiming] = []
        try:
            stt_result, t = _timed("stt", _transcribe_with_retry, audio_bytes, filename)
            timings.append(t)
        except StageException as se:
            timings.append(StageTiming(stage=se.stage, ms=se.ms))
            return PipelineResult(
                status=PipelineStatus.ERROR,
                error=f"STT failed: {se.orig_err}",
                error_stage="stt",
                error_message=str(se.orig_err),
                timings_ms=timings,
                total_ms=sum(t.ms for t in timings),
            )
        except Exception as e:
            return PipelineResult(
                status=PipelineStatus.ERROR,
                error=f"STT failed: {e}",
                error_stage="stt",
                error_message=str(e),
                timings_ms=timings,
                total_ms=sum(t.ms for t in timings),
            )
        return self._run(query_text=stt_result["text"], timings=timings, stt_provider=stt_result.get("provider"))

    def _run(self, query_text: str, timings: list[StageTiming], stt_provider: str | None = None) -> PipelineResult:
        guard_events: list[dict] = []
        t_start = time.perf_counter()

        # --- Stage: input guardrail ---
        g_in, t = _timed("guard_input", gr.check_input_safety, query_text)
        timings.append(t)
        guard_events.append(vars(g_in))
        if not g_in.passed:
            return self._finish(
                PipelineStatus.REFUSED_UNSAFE_INPUT, query_text, gr.UNSAFE_MESSAGE,
                [], guard_events, timings, t_start,
            )

        # --- Stage: retrieval ---
        retrieved, t = _timed("retrieve", self.retriever.search, query_text, self.top_k)
        timings.append(t)
        top_score = retrieved[0].score if retrieved else 0.0

        # --- Stage: retrieval relevance guardrail ---
        retrieved_text = " ".join(r.chunk.text for r in retrieved)
        g_rel, t = _timed("guard_retrieval", gr.check_retrieval_relevance, top_score, query_text, retrieved_text)
        timings.append(t)
        guard_events.append(vars(g_rel))
        if not g_rel.passed:
            refusal_msg = g_rel.reason if g_rel.reason else gr.REFUSAL_MESSAGE
            return self._finish(
                PipelineStatus.REFUSED_OFF_TOPIC, query_text, refusal_msg,
                retrieved, guard_events, timings, t_start,
            )

        context_texts = [r.chunk.text for r in retrieved]

        # --- Stage: generation ---
        try:
            gen_result, t = _timed("generate", _generate_with_retry, query_text, context_texts)
            timings.append(t)
        except StageException as se:
            timings.append(StageTiming(stage=se.stage, ms=se.ms))
            return self._finish(
                PipelineStatus.ERROR, query_text, "", retrieved, guard_events, timings, t_start,
                error=f"Generation failed: {se.orig_err}", error_stage="generation", error_message=str(se.orig_err),
            )
        except Exception as e:
            return self._finish(
                PipelineStatus.ERROR, query_text, "", retrieved, guard_events, timings, t_start,
                error=f"Generation failed: {e}", error_stage="generation", error_message=str(e),
            )

        answer = gen_result["text"]

        # --- Stage: output grounding guardrail ---
        g_ground, t = _timed("guard_output", gr.check_answer_grounded, answer, context_texts)
        timings.append(t)
        guard_events.append(vars(g_ground))
        if not g_ground.passed:
            return self._finish(
                PipelineStatus.REFUSED_UNGROUNDED, query_text, gr.REFUSAL_MESSAGE,
                retrieved, guard_events, timings, t_start,
                generation_method=gen_result.get("method"),
            )

        return self._finish(
            PipelineStatus.OK, query_text, answer, retrieved, guard_events, timings, t_start,
            generation_method=gen_result.get("method"),
        )

    def _rc_dict(self, r: RetrievedChunk) -> dict:
        return {
            "chunk_id": r.chunk.chunk_id,
            "doc_id": r.chunk.doc_id,
            "strategy": r.chunk.strategy,
            "score": r.score,
            "text": r.chunk.text,
        }

    def _finish(self, status, query_text, answer, retrieved, guard_events, timings, t_start, generation_method=None, error=None, error_stage=None, error_message=None):
        total_ms = sum(t.ms for t in timings)
        return PipelineResult(
            status=status,
            query_text=query_text,
            answer=answer,
            retrieved=[self._rc_dict(r) for r in retrieved],
            guard_events=guard_events,
            timings_ms=timings,
            total_ms=total_ms,
            generation_method=generation_method,
            error=error,
            error_stage=error_stage,
            error_message=error_message,
        )
