"""
Chunking strategies for the MSMARCO-XI corpus.

We deliberately implement more than one strategy because different
query types benefit from different chunk shapes:

  1. FixedSizeChunker      - character-window with overlap. Cheap,
                              predictable, good recall floor.
  2. SentenceWindowChunker - groups N sentences with a sliding stride,
                              a lightweight stand-in for "semantic"
                              chunking that respects sentence boundaries
                              instead of cutting mid-word.
  3. PassageChunker        - MSMARCO-XI is already organized as
                              (query, passage) pairs; this strategy
                              treats each source passage as one
                              metadata-aware chunk (no splitting), which
                              is best when passages are already short.
  4. HybridChunker         - runs fixed-size + sentence-window and
                              deduplicates, giving retrieval two
                              differently-shaped views of the same text.

Every chunk carries metadata (doc_id, strategy, char span, source
passage id) so retrieval results are traceable back to the source and
so we can filter/boost by strategy at query time.
"""
from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from typing import Iterable


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")  # handles Hindi '।' too


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents or [text]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    strategy: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "strategy": self.strategy,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }


def _mk_id(doc_id: str, strategy: str, start: int) -> str:
    raw = f"{doc_id}:{strategy}:{start}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


class FixedSizeChunker:
    name = "fixed_size"

    def __init__(self, size: int = 220, overlap: int = 40):
        self.size = size
        self.overlap = overlap

    def chunk(self, doc_id: str, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        chunks = []
        if not text:
            return chunks
        step = max(1, self.size - self.overlap)
        for start in range(0, len(text), step):
            end = min(start + self.size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    Chunk(
                        chunk_id=_mk_id(doc_id, self.name, start),
                        doc_id=doc_id,
                        text=piece,
                        strategy=self.name,
                        start_char=start,
                        end_char=end,
                        metadata=metadata,
                    )
                )
            if end == len(text):
                break
        return chunks


class SentenceWindowChunker:
    name = "sentence_window"

    def __init__(self, window: int = 3, stride: int = 1):
        self.window = window
        self.stride = stride

    def chunk(self, doc_id: str, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        sents = split_sentences(text)
        chunks = []
        if not sents:
            return chunks
        i = 0
        cursor = 0  # approximate char offset for traceability
        while i < len(sents):
            window_sents = sents[i : i + self.window]
            piece = " ".join(window_sents).strip()
            start = text.find(window_sents[0], cursor) if window_sents else cursor
            start = max(start, 0)
            end = start + len(piece)
            if piece:
                chunks.append(
                    Chunk(
                        chunk_id=_mk_id(doc_id, self.name, start),
                        doc_id=doc_id,
                        text=piece,
                        strategy=self.name,
                        start_char=start,
                        end_char=end,
                        metadata={**metadata, "sentence_span": [i, i + len(window_sents)]},
                    )
                )
            cursor = start
            if i + self.window >= len(sents):
                break
            i += self.stride
        return chunks


class PassageChunker:
    """Treat each already-short MSMARCO passage as a single metadata-rich
    chunk. Best when passages are already retrieval-sized (~1-3 sentences)."""

    name = "passage"

    def __init__(self, max_chars: int = 600):
        self.max_chars = max_chars

    def chunk(self, doc_id: str, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.max_chars:
            return [
                Chunk(
                    chunk_id=_mk_id(doc_id, self.name, 0),
                    doc_id=doc_id,
                    text=text,
                    strategy=self.name,
                    start_char=0,
                    end_char=len(text),
                    metadata=metadata,
                )
            ]
        # Passage too long to stay atomic -> fall back to fixed-size split
        # for this doc only, still tagged under the passage strategy.
        fsc = FixedSizeChunker(size=self.max_chars, overlap=60)
        return [
            Chunk(c.chunk_id, c.doc_id, c.text, self.name, c.start_char, c.end_char, metadata)
            for c in fsc.chunk(doc_id, text, metadata)
        ]


class HybridChunker:
    """Runs multiple strategies and merges, giving the index more than one
    'shape' of each document to match against differently-phrased queries."""

    name = "hybrid"

    def __init__(self, strategies: Iterable | None = None):
        self.strategies = list(strategies) if strategies else [
            # Preserve MSMARCO's supplied passage boundary when it is already
            # retrieval-sized, then add two complementary views for recall.
            PassageChunker(),
            FixedSizeChunker(),
            SentenceWindowChunker(),
        ]

    def chunk(self, doc_id: str, text: str, metadata: dict | None = None) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        seen_text = set()
        for strat in self.strategies:
            for c in strat.chunk(doc_id, text, metadata):
                # Keep equivalent text from different strategies: the strategy
                # itself is a retrieval view and must remain traceable. Only
                # remove accidental duplicates produced by the same strategy.
                key = (c.strategy, c.text.strip().lower())
                if key in seen_text:
                    continue
                seen_text.add(key)
                all_chunks.append(c)
        return all_chunks


STRATEGY_REGISTRY = {
    "fixed_size": FixedSizeChunker(),
    "sentence_window": SentenceWindowChunker(),
    "passage": PassageChunker(),
    "hybrid": HybridChunker(),
}


def chunk_corpus(docs: list[dict], strategy: str = "hybrid") -> list[Chunk]:
    """docs: list of {"doc_id": str, "text": str, "metadata": dict}"""
    chunker = STRATEGY_REGISTRY[strategy]
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunker.chunk(d["doc_id"], d["text"], d.get("metadata")))
    return out
