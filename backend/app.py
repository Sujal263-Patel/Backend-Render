"""
FastAPI backend for the voice-enabled RAG system.

Endpoints:
  GET  /health                 -> liveness check
  POST /api/query/text         -> text-only query (skips STT), useful for
                                   testing and for browsers without mic access
  POST /api/query/audio        -> full pipeline: audio -> STT -> RAG
  GET  /api/corpus/stats       -> chunking/index stats, for the UI's debug panel
  POST /api/benchmark/run      -> runs the latency benchmark on demand,
                                   returns P50/P70/P100

Run with:  uvicorn app:app --reload --port 8000
"""
from __future__ import annotations

import logging
import math
from collections import deque
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from chunking import chunk_corpus
from vectorstore import build_index
from harness import RAGHarness
from data.sample_corpus import DatasetLoadError, load_index_corpus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-rag")

app = FastAPI(title="Voice-Enabled RAG (MSMARCO-XI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only -- restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_corpus() -> tuple[list[dict], str]:
    """Load ai4bharat/MSMARCO-XI from Hugging Face (competition default)."""
    return load_index_corpus(
        data_source=settings.DATA_SOURCE,
        config=settings.MSMARCO_CONFIG,
        split=settings.MSMARCO_SPLIT,
        limit=settings.MSMARCO_LIMIT,
        strict_dataset_only=settings.STRICT_DATASET_ONLY,
    )


# ---- Build the index once at startup ----
# Fails fast and loudly if the real dataset can't be reached -- see
# DatasetLoadError / STRICT_DATASET_ONLY above. No silent substitution.
_docs, _data_source_used = _load_corpus()
logger.info(f"Loaded {len(_docs)} docs via: {_data_source_used}")
_chunks = chunk_corpus(_docs, strategy="hybrid")
_retriever = build_index(_chunks)
_harness = RAGHarness(_retriever, top_k=settings.TOP_K)
_stt_latencies_ms: deque[float] = deque(maxlen=500)
_retrieval_latencies_ms: deque[float] = deque(maxlen=500)
_generation_latencies_ms: deque[float] = deque(maxlen=500)
_voice_latencies_ms: deque[float] = deque(maxlen=500)
_text_latencies_ms: deque[float] = deque(maxlen=500)
logger.info(f"Active index: {len(_chunks)} chunks built successfully into FAISS Vector DB.")


class TextQuery(BaseModel):
    query: str
    gen_mode: str | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_db": "FAISS (Dense Semantic IndexFlatIP + BM25 Lexical + TF-IDF Hybrid)",
        "stt_provider": settings.STT_PROVIDER,
        "generation_mode": settings.GENERATION_MODE,
        "data_source": _data_source_used,
        "num_docs": len(_docs),
        "num_chunks": len(_chunks),
    }


@app.get("/api/vectordb/info")
def vectordb_info():
    """Detailed Vector Database metadata and indexing status."""
    stats = _retriever.get_stats() if hasattr(_retriever, "get_stats") else {}
    return {
        "vector_db": "FAISS",
        "index_type": settings.FAISS_INDEX_TYPE,
        "embedding_dim": settings.EMBEDDING_DIM,
        "retriever_type": settings.RETRIEVER_TYPE,
        "num_chunks": len(_chunks),
        "details": stats,
    }


@app.get("/api/corpus/stats")
def corpus_stats():
    by_strategy: dict[str, int] = {}
    for c in _chunks:
        by_strategy[c.strategy] = by_strategy.get(c.strategy, 0) + 1
    return {
        "data_source": _data_source_used,
        "vector_db": "FAISS (IndexFlatIP)",
        "num_docs": len(_docs),
        "num_chunks": len(_chunks),
        "chunks_by_strategy": by_strategy,
        "top_k": settings.TOP_K,
        "min_retrieval_score": settings.MIN_RETRIEVAL_SCORE,
    }


@app.post("/api/query/text")
def query_text(body: TextQuery):
    if not body.query or not body.query.strip():
        raise HTTPException(400, "query must not be empty")
    if body.gen_mode:
        settings.GENERATION_MODE = body.gen_mode
    result = _harness.run_from_text(body.query)
    _text_latencies_ms.append(result.total_ms)
    if "retrieve" in result.timings_ms:
        _retrieval_latencies_ms.append(result.timings_ms["retrieve"])
    if "generate" in result.timings_ms:
        _generation_latencies_ms.append(result.timings_ms["generate"])
    return result.to_dict()


@app.post("/api/query/audio")
async def query_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "uploaded audio file is empty")
    result = _harness.run_from_audio(audio_bytes, filename=file.filename or "audio.wav")
    _voice_latencies_ms.append(result.total_ms)
    if "stt" in result.timings_ms:
        _stt_latencies_ms.append(result.timings_ms["stt"])
    if "retrieve" in result.timings_ms:
        _retrieval_latencies_ms.append(result.timings_ms["retrieve"])
    if "generate" in result.timings_ms:
        _generation_latencies_ms.append(result.timings_ms["generate"])
    return result.to_dict()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower, upper = math.floor(position), math.ceil(position)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 2)


@app.get("/api/latency/voice")
def voice_latency_stats():
    """Observed end-to-end Sarvam + FAISS RAG timings with stage breakdown."""
    def p_block(vals: deque[float]):
        lst = list(vals)
        return {
            "p50_ms": _percentile(lst, 50),
            "p70_ms": _percentile(lst, 70),
            "p100_ms": round(max(lst), 2) if lst else None,
            "samples": len(lst),
        }

    return {
        "stt": p_block(_stt_latencies_ms),
        "faiss_retrieval": p_block(_retrieval_latencies_ms),
        "generation": p_block(_generation_latencies_ms),
        "total_voice_e2e": p_block(_voice_latencies_ms),
        "local_pipeline_text_in": p_block(_text_latencies_ms),
    }


@app.get("/api/dataset/verify")
def verify_dataset():
    """Live Verification & Provenance Endpoint for ai4bharat/MSMARCO-XI Dataset."""
    sample_docs = []
    for d in _docs[:5]:
        sample_docs.append({
            "doc_id": d.get("doc_id"),
            "dataset_origin": "ai4bharat/MSMARCO-XI",
            "metadata": d.get("metadata", {}),
            "passage_snippet": d.get("text", "")[:120] + "...",
        })
    return {
        "verified": _data_source_used.startswith("ai4bharat/MSMARCO-XI/"),
        "dataset_name": "ai4bharat/MSMARCO-XI",
        "huggingface_url": "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI",
        "huggingface_config": settings.MSMARCO_CONFIG,
        "loaded_via": _data_source_used,
        "total_documents_indexed": len(_docs),
        "total_chunks_indexed": len(_chunks),
        "chunking_strategies": ["fixed_size", "sentence_window", "passage", "hybrid"],
        "sample_indexed_records": sample_docs,
        "strict_dataset_only": settings.STRICT_DATASET_ONLY,
    }


@app.post("/api/benchmark/run")
def run_benchmark():
    from benchmarks.latency_bench import run_benchmark as _bench
    from data.sample_corpus import benchmark_queries_from_docs

    test_queries = benchmark_queries_from_docs(_docs, limit=100, randomize=True)
    return _bench(_harness, n_repeats=1, test_queries=test_queries)


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
