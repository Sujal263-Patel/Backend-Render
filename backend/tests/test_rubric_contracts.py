"""Offline contract checks for the competition-critical behaviour."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from data.sample_corpus import DatasetLoadError, expand_msmarco_rows, load_index_corpus
from guardrails import check_answer_grounded
from chunking import chunk_corpus


def test_official_msmarco_schema_expands_translated_passages():
    rows = [{
        "query_id": 7,
        "query": "ताजमहल कहाँ है?",
        "Answer": "आगरा में",
        "passages": {"Translated_passages": ["ताजमहल आगरा में है।", "दूसरा संदर्भ।"], "is_selected": [1, 0]},
    }]
    docs = expand_msmarco_rows(rows, config="hi")
    assert [doc["doc_id"] for doc in docs] == ["7:0", "7:1"]
    assert docs[0]["metadata"]["is_selected"] is True
    assert docs[0]["metadata"]["language"] == "hi"


def test_hybrid_chunking_preserves_passage_and_multiple_windows():
    chunks = chunk_corpus([{"doc_id": "x", "text": "पहला वाक्य। दूसरा वाक्य। तीसरा वाक्य।", "metadata": {}}])
    assert {chunk.strategy for chunk in chunks} >= {"passage", "fixed_size", "sentence_window"}


def test_grounding_rejects_unsupported_answer():
    verdict = check_answer_grounded("The moon is made of cheese.", ["The Taj Mahal is in Agra."])
    assert verdict.passed is False


def test_strict_mode_blocks_offline_local_source():
    try:
        load_index_corpus(
            data_source="local",
            strict_dataset_only=True,
        )
        assert False, "expected DatasetLoadError"
    except DatasetLoadError as exc:
        assert "development-only" in str(exc)


def test_faiss_vector_database_hybrid_retrieval():
    from vectorstore import build_index
    test_chunks = chunk_corpus([
        {"doc_id": "doc1", "text": "ताजमहल भारत के आगरा शहर में स्थित एक विश्व प्रसिद्ध स्मारक है।", "metadata": {}},
        {"doc_id": "doc2", "text": "प्रकाश संश्लेषण पौधों में भोजन बनाने की एक जैविक प्रक्रिया है।", "metadata": {}},
    ])
    retriever = build_index(test_chunks)
    assert retriever.faiss_index.ntotal == len(test_chunks)
    results = retriever.search("ताजमहल कहाँ है?", top_k=2)
    assert len(results) > 0
    assert "ताजमहल" in results[0].chunk.text
    stats = retriever.get_stats()
    assert stats["vector_db"] == "FAISS (IndexFlatIP)"
    assert stats["embedding_dimension"] == 768
