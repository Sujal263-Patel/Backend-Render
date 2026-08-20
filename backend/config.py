"""
Central configuration. All secrets come from environment variables —
never hardcode API keys.
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)


class Settings:
    # ---- Speech-to-Text provider ----
    # This submission deliberately uses one provider: Sarvam Saarika v2.
    STT_PROVIDER: str = "sarvam"

    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
    SARVAM_STT_URL: str = "https://api.sarvam.ai/speech-to-text"
    SARVAM_LANGUAGE_CODE: str = os.getenv("SARVAM_LANGUAGE_CODE", "hi-IN")

    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    ELEVENLABS_STT_URL: str = "https://api.elevenlabs.io/v1/speech-to-text"
    ELEVENLABS_MODEL_ID: str = os.getenv("ELEVENLABS_MODEL_ID", "scribe_v1")

    # ---- Answer generation ----
    # Local grounded extractive synthesis on the query path.
    GENERATION_MODE: str = os.getenv("GENERATION_MODE", "fast")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # ---- Retrieval / Vector DB / chunking ----
    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "FAISS")  # "FAISS" | "InMemory"
    FAISS_INDEX_TYPE: str = os.getenv("FAISS_INDEX_TYPE", "IndexFlatIP")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", 768))
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "fast_local")  # "fast_local" (<1ms) | "gemini"
    RETRIEVER_TYPE: str = os.getenv("RETRIEVER_TYPE", "hybrid_faiss")  # "hybrid_faiss" | "dense_faiss" | "sparse_bm25"
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
    DATASET_NAME: str = "ai4bharat/MSMARCO-XI"
    DATA_PATH: str = os.getenv("DATA_PATH", "")
    TOP_K: int = int(os.getenv("TOP_K", 5))

    # ---- Data source ----
    # "hf_lib" streams the official dataset; only the bounded index slice is
    # retained in memory. "sample" is explicitly development-only.
    DATA_SOURCE: str = os.getenv("DATA_SOURCE", "hf_lib")
    MSMARCO_LIMIT: int = int(os.getenv("MSMARCO_LIMIT", 100))  # query records
    MSMARCO_SPLIT: str = os.getenv("MSMARCO_SPLIT", "train")
    MSMARCO_CONFIG: str = os.getenv("MSMARCO_CONFIG", "hi")
    # If true (default), the app refuses to start rather than silently
    # substitute the offline sample corpus when ai4bharat/MSMARCO-XI
    # can't be reached. Only set to "false" for local dev when you
    # explicitly want DATA_SOURCE=sample.
    STRICT_DATASET_ONLY: bool = os.getenv("STRICT_DATASET_ONLY", "true").lower() == "true"

    FIXED_CHUNK_SIZE: int = 220        # chars
    FIXED_CHUNK_OVERLAP: int = 40      # chars
    SENTENCE_WINDOW: int = 3           # sentences per semantic chunk
    SENTENCE_STRIDE: int = 1           # sliding stride in sentences

    # ---- Guardrails ----
    MIN_RETRIEVAL_SCORE: float = float(os.getenv("MIN_RETRIEVAL_SCORE", 0.08))
    GROUNDING_OVERLAP_THRESHOLD: float = float(os.getenv("GROUNDING_OVERLAP_THRESHOLD", 0.12))

    # ---- Latency ----
    # NOTE: 10ms is only realistic for the in-memory retrieval stage.
    # Network STT calls and LLM generation calls run in the hundreds-of-ms
    # to seconds range regardless of implementation quality -- see README.
    RETRIEVAL_LATENCY_TARGET_MS: float = 10.0


settings = Settings()
