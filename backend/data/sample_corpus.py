"""
Dataset ingestion for ai4bharat/MSMARCO-XI.

100% of data is streamed live from the official Hugging Face dataset repository.
Zero offline sample datasets or mock files.
"""
from __future__ import annotations

import logging
import os
import requests

logger = logging.getLogger(__name__)

DATASETS_SERVER_BASE = "https://datasets-server.huggingface.co"
HF_DATASET_ID = "ai4bharat/MSMARCO-XI"


class DatasetLoadError(RuntimeError):
    """Raised when the official dataset cannot be loaded."""


def load_index_corpus(
    *,
    data_source: str = "hf_lib",
    config: str = "hi",
    split: str = "train",
    limit: int | None = 100,
    strict_dataset_only: bool = True,
) -> tuple[list[dict], str]:
    """Load the official MSMARCO-XI index corpus directly from Hugging Face.

    Returns (docs, provenance_label).
    """
    errors: list[str] = []
    for streaming in (True, False):
        try:
            docs = load_real_dataset(
                config=config,
                split=split,
                limit=limit,
                streaming=streaming,
            )
            if docs:
                mode = "streamed" if streaming else "downloaded"
                return docs, f"ai4bharat/MSMARCO-XI/{config} {mode} ({len(docs)} passages)"
        except Exception as exc:
            errors.append(f"streaming={streaming}: {exc}")
            logger.warning("MSMARCO-XI load attempt failed (%s)", errors[-1])

    detail = "; ".join(errors) if errors else "no passages returned"
    raise DatasetLoadError(
        f"Failed to stream official ai4bharat/MSMARCO-XI dataset ({detail}). "
        "Check internet connection to Hugging Face."
    )


def benchmark_queries_from_docs(docs: list[dict], limit: int = 10, randomize: bool = True) -> list[str]:
    """Build benchmark queries from indexed MSMARCO-XI query metadata."""
    import random
    seen: set[str] = set()
    all_queries: list[str] = []
    for doc in docs:
        query = str((doc.get("metadata") or {}).get("query", "")).strip()
        if query and query not in seen:
            seen.add(query)
            all_queries.append(query)
    
    if not all_queries:
        return []
    
    if randomize:
        random.shuffle(all_queries)
        return all_queries[:limit]
    return all_queries[:limit]


LANG_MAP = {
    "hi": "hin",
    "hindi": "hin",
    "hin": "hin",
    "gu": "guj",
    "guj": "guj",
    "bn": "beng",
    "beng": "beng",
    "mr": "mar",
    "mar": "mar",
    "ta": "tam",
    "tam": "tam",
    "te": "tel",
    "tel": "tel",
    "kn": "knn",
    "knn": "knn",
    "ml": "mlm",
    "mlm": "mlm",
    "pa": "pan",
    "pan": "pan",
    "or": "ors",
    "ors": "ors",
    "as": "ass",
    "ass": "ass",
    "ur": "urd",
    "urd": "urd",
}


def load_real_dataset(config: str = "hi", split: str = "train", limit: int | None = 100, streaming: bool = True) -> list[dict]:
    """Stream official MSMARCO-XI records and expand every translated passage.

    Optimized for low-memory environments (Render/Heroku/Docker <512MB RAM).
    """
    import gc
    lang_code = LANG_MAP.get(config.lower(), config.lower())
    parquet_filename = f"{lang_code}{split}.parquet"
    target_limit = min(limit or 100, 100)
    logger.info(f"Streaming {target_limit} MSMARCO-XI records live from Hugging Face ({parquet_filename})...")
    rows = []

    # Strategy 1: Ultra-lightweight Hugging Face Datasets API (0 RAM overhead)
    try:
        hf_api_url = f"{DATASETS_SERVER_BASE}/rows"
        resp = requests.get(
            hf_api_url,
            params={
                "dataset": HF_DATASET_ID,
                "config": config,
                "split": split,
                "offset": 0,
                "length": target_limit,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            rows = [item["row"] for item in data.get("rows", [])]
            if rows:
                logger.info(f"Successfully streamed {len(rows)} records via low-memory Hugging Face API.")
    except Exception as exc_hf:
        logger.warning(f"HF API notice: {exc_hf}")

    # Strategy 2: Fast fsspec protocol-aware reader reading only a 100-record batch
    if not rows:
        try:
            import fsspec
            import pyarrow.parquet as pq
            with fsspec.open(parquet_url, "rb", block_size=512 * 1024) as f:
                parquet_file = pq.ParquetFile(f)
                available_cols = set(parquet_file.schema.names)
                desired_cols = {
                    "passages", "Translated_passages", "query", "Query", "Eng_Query",
                    "Answer", "Eng_Answer", "query_id", "id", "passage", "text", "Hindi_Passage", "Marathi_Passage"
                }
                read_cols = [c for c in available_cols if c in desired_cols] or None
                for batch in parquet_file.iter_batches(batch_size=target_limit, columns=read_cols):
                    rows = batch.to_pylist()[:target_limit]
                    break
            gc.collect()
            logger.info(f"Successfully streamed {len(rows)} query records directly into Vector DB pipeline.")
        except Exception as exc1:
            logger.warning(f"Strategy 2 (fsspec) notice: {exc1}; using fast in-memory HTTP stream...")
            try:
                import io
                import pyarrow.parquet as pq
                resp = requests.get(parquet_url, stream=True, timeout=30)
                resp.raise_for_status()
                buf = io.BytesIO()
                for chunk in resp.iter_content(chunk_size=256 * 1024):
                    buf.write(chunk)
                    if buf.tell() >= 2 * 1024 * 1024:
                        break
                buf.seek(0)
                try:
                    parquet_file = pq.ParquetFile(buf)
                    for batch in parquet_file.iter_batches(batch_size=target_limit):
                        rows = batch.to_pylist()[:target_limit]
                        break
                except Exception:
                    pass
                del buf
                gc.collect()
                if rows:
                    logger.info(f"Successfully streamed {len(rows)} query records via stream batch.")
            except Exception as exc2:
                logger.warning(f"Strategy 3 notice: {exc2}")
                raise RuntimeError(f"Could not stream MSMARCO-XI partition {parquet_filename}: {exc1} | {exc2}")

    docs = expand_msmarco_rows(rows, config=config, split=split)
    if not docs:
        raise ValueError("MSMARCO-XI returned no passages; check dataset schema.")
    del rows
    gc.collect()
    logger.info(f"Expanded into {len(docs)} authentic {config.upper()} + English index passages.")
    return docs


def expand_msmarco_rows(rows: list[dict], config: str = "hi", split: str = "train") -> list[dict]:
    """Convert official query records to traceable Hindi and English index documents."""
    docs = []
    for row_number, row in enumerate(rows):
        query_id = str(row.get("query_id", row.get("id", row_number)))
        hi_query = str(row.get("query") or row.get("Query") or "").strip()
        hi_answer = str(row.get("Answer") or row.get("answer") or "").strip()
        eng_query = str(row.get("Eng_Query") or "").strip()
        eng_answer = str(row.get("Eng_Answer") or "").strip()

        # 1. Hindi passage extraction
        passages = row.get("passages") or {}
        hi_passages = []
        if isinstance(passages, dict):
            hi_passages = passages.get("Translated_passages") or passages.get("passage_text") or []
        elif isinstance(passages, list):
            hi_passages = passages
        
        if not hi_passages:
            t = row.get("Translated_passages") or row.get("passage") or row.get("text")
            if t:
                hi_passages = [t] if isinstance(t, str) else t

        if not hi_passages and hi_answer and hi_answer not in ("कोई उत्तर नहीं मिला।", "No Answer Present.", "No Answer Present"):
            hi_passages = [hi_answer]

        for p_idx, p_text in enumerate(hi_passages):
            if isinstance(p_text, dict):
                p_text = p_text.get("text") or p_text.get("passage_text") or str(p_text)
            if isinstance(p_text, str) and p_text.strip():
                docs.append({
                    "doc_id": f"{query_id}:{config}:{p_idx}",
                    "text": p_text.strip(),
                    "metadata": {
                        "dataset": HF_DATASET_ID,
                        "language": config,
                        "split": split,
                        "query_id": query_id,
                        "query": hi_query,
                        "answer": hi_answer,
                        "passage_index": p_idx,
                    },
                })

        # 2. English passage extraction for bilingual retrieval
        if eng_answer and eng_answer not in ("No Answer Present.", "No Answer Present", "कोई उत्तर नहीं मिला।"):
            docs.append({
                "doc_id": f"{query_id}:en:0",
                "text": eng_answer.strip(),
                "metadata": {
                    "dataset": HF_DATASET_ID,
                    "language": "en",
                    "split": split,
                    "query_id": query_id,
                    "query": eng_query,
                    "answer": eng_answer,
                    "passage_index": 0,
                },
            })
    return docs


def list_available_splits(dataset: str = HF_DATASET_ID) -> list[dict]:
    """GET /splits from Hugging Face."""
    resp = requests.get(
        f"{DATASETS_SERVER_BASE}/splits",
        params={"dataset": dataset},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("splits", [])


def load_rows_paged(
    dataset: str = HF_DATASET_ID,
    config: str = "default",
    split: str = "train",
    limit: int | None = 2000,
    page_size: int = 100,
) -> list[dict]:
    """GET /rows, paged live from Hugging Face."""
    all_rows = []
    offset = 0
    while limit is None or offset < limit:
        length = page_size if limit is None else min(page_size, limit - offset)
        resp = requests.get(
            f"{DATASETS_SERVER_BASE}/rows",
            params={
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            },
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("rows", [])
        if not rows:
            break
        all_rows.extend(r["row"] for r in rows)
        offset += len(rows)
        if len(rows) < length:
            break

    return _rows_to_docs(all_rows)


def _rows_to_docs(rows) -> list[dict]:
    docs = []
    seen = set()
    for i, row in enumerate(rows):
        passage = row.get("passage") or row.get("text") or ""
        doc_id = str(row.get("passage_id", row.get("query_id", i)))
        if not passage or doc_id in seen:
            continue
        seen.add(doc_id)
        docs.append({"doc_id": doc_id, "text": passage, "metadata": {"query": row.get("query", "")}})
    return docs
