"""
Unit tests for Standalone BM25 Retriever (Person 2 Workstream).
Tests:
1. 60-word chunk size with 15-word overlap
2. Raw document indexing
3. Direct chunk indexing
4. Retrieval and top-k filtering
5. Raw score preservation (unnormalized)
6. Deterministic tie-breaking
7. Empty index handling
8. Empty / whitespace query handling
9. Metadata preservation
10. Backend identification
"""

import os
import sys
import pytest

# Ensure person2_hybrid_retrieval/src is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from hybrid_retrieval.bm25_retriever import BM25Retriever, SimpleBM25Fallback


def test_bm25_backend_identification():
    retriever = BM25Retriever()
    backend = retriever.backend_name
    assert backend in ["rank_bm25", "fallback"]
    stats = retriever.stats()
    assert stats["backend"] == backend
    assert stats["chunk_size_words"] == 60
    assert stats["chunk_overlap_words"] == 15


def test_bm25_chunking_60_15():
    retriever = BM25Retriever(chunk_size_words=60, chunk_overlap_words=15)
    # Generate 120 words
    words = [f"word{i}" for i in range(120)]
    text = " ".join(words)
    chunks = retriever.add_document("doc_test.txt", text)

    # Step is 60 - 15 = 45 words.
    # Chunk 1: words 0..59 (60 words)
    # Chunk 2: words 45..104 (60 words)
    # Chunk 3: words 90..119 (30 words)
    assert len(chunks) == 3
    assert chunks[0]["chunk_id"] == "chunk_1"
    assert chunks[1]["chunk_id"] == "chunk_2"
    assert chunks[2]["chunk_id"] == "chunk_3"

    # Verify word count of chunk 1
    assert len(chunks[0]["text"].split()) == 60
    # Verify word count of chunk 3
    assert len(chunks[2]["text"].split()) == 30


def test_bm25_direct_chunk_indexing():
    retriever = BM25Retriever()
    direct_chunks = [
        {"chunk_id": "c_1", "doc_name": "manual.pdf", "text": "Warranty lasts 24 months from purchase.", "metadata": {"page": 1}},
        {"chunk_id": "c_2", "doc_name": "manual.pdf", "text": "Returns accepted within thirty calendar days.", "metadata": {"page": 2}},
    ]
    retriever.add_chunks_directly(direct_chunks)
    assert len(retriever.chunks) == 2

    res = retriever.retrieve("warranty 24 months", top_k=1)
    assert len(res) == 1
    assert res[0]["chunk_id"] == "c_1"
    assert res[0]["metadata"]["page"] == 1


def test_bm25_raw_scores_preserved():
    retriever = BM25Retriever()
    retriever.add_document(
        "policy.txt",
        "The customer shall remit the payment within thirty days of invoice."
    )
    res = retriever.retrieve("customer remit payment", top_k=1)
    assert len(res) == 1
    # Raw BM25 scores can be any positive floating point, e.g. > 0.0
    assert isinstance(res[0]["score"], float)
    assert res[0]["score"] > 0.0


def test_bm25_top_k():
    retriever = BM25Retriever()
    chunks = [
        {"chunk_id": f"chunk_{i}", "doc_name": "bulk.txt", "text": f"Common term frequency test item {i} with audit metadata."}
        for i in range(10)
    ]
    retriever.add_chunks_directly(chunks)

    res_3 = retriever.retrieve("term frequency test", top_k=3)
    assert len(res_3) == 3

    res_5 = retriever.retrieve("term frequency test", top_k=5)
    assert len(res_5) == 5


def test_bm25_deterministic_ranking_and_ties():
    retriever = BM25Retriever()
    # Identical text in three chunks -> identical raw BM25 score
    chunks = [
        {"chunk_id": "chunk_C", "doc_name": "a.txt", "text": "Exact same content for all chunks."},
        {"chunk_id": "chunk_A", "doc_name": "b.txt", "text": "Exact same content for all chunks."},
        {"chunk_id": "chunk_B", "doc_name": "c.txt", "text": "Exact same content for all chunks."},
    ]
    retriever.add_chunks_directly(chunks)

    res1 = retriever.retrieve("Exact same content", top_k=3)
    res2 = retriever.retrieve("Exact same content", top_k=3)

    # Identical results
    assert [r["chunk_id"] for r in res1] == [r["chunk_id"] for r in res2]
    # Tie-breaking by chunk_id ascending: chunk_A, chunk_B, chunk_C
    assert [r["chunk_id"] for r in res1] == ["chunk_A", "chunk_B", "chunk_C"]


def test_bm25_empty_query_and_empty_index():
    retriever = BM25Retriever()
    # Empty index
    assert retriever.retrieve("test query", top_k=3) == []

    # Add content
    retriever.add_document("doc.txt", "Valid document content.")
    # Empty string query
    assert retriever.retrieve("", top_k=3) == []
    # Whitespace only query
    assert retriever.retrieve("   \t  \n ", top_k=3) == []


def test_bm25_clear():
    retriever = BM25Retriever()
    retriever.add_document("doc.txt", "Some content to index.")
    assert len(retriever.chunks) == 1
    retriever.clear()
    assert len(retriever.chunks) == 0
    assert retriever.retrieve("content", top_k=1) == []


def test_fallback_bm25_explicit():
    # Test SimpleBM25Fallback directly to ensure mathematical correctness
    corpus = [
        ["the", "quick", "brown", "fox"],
        ["jumped", "over", "the", "lazy", "dog"],
        ["the", "fox", "was", "very", "quick"],
    ]
    fb = SimpleBM25Fallback(corpus, k1=1.5, b=0.75)
    scores = fb.get_scores(["quick", "fox"])
    assert len(scores) == 3
    # Doc 0 and Doc 2 have both "quick" and "fox", Doc 1 has neither
    assert scores[0] > 0.0
    assert scores[2] > 0.0
    assert scores[1] == 0.0
