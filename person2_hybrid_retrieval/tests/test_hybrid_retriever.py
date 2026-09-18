"""
Unit & Integration tests for HybridRetriever (Person 2 Workstream).
Tests:
1. End-to-end hybrid retrieval with Mock Semantic provider
2. Candidate union and deduplication
3. Decoupled candidate sizes (bm25_top_k, semantic_top_k) vs final_top_k
4. RRF vs Weighted fusion selection
5. Explainability metadata preservation
6. Deterministic repeated queries
7. Adaptive weighting is OFF by default (mandatory invariant)
8. Adaptive weighting when explicitly enabled
9. Document addition and chunking
10. Index clearing and state management
11. Real Person 1 compatibility test (if person1_dense_retrieval is present)
"""

import os
import sys
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from hybrid_retrieval.hybrid_retriever import HybridRetriever
from hybrid_retrieval.bm25_retriever import BM25Retriever
from hybrid_retrieval.semantic_adapter import (
    MockSemanticRetriever,
    Person1SemanticAdapter,
)


def test_hybrid_retriever_initialization_defaults():
    hr = HybridRetriever()
    assert hr.fusion_strategy == "rrf"
    assert hr.alpha == 0.5
    assert hr.rrf_k == 60
    assert hr.bm25_top_k == 5
    assert hr.semantic_top_k == 5
    assert hr.final_top_k == 3
    # MANDATORY: adaptive weighting MUST be False by default
    assert hr.enable_adaptive_weighting is False


def test_hybrid_retriever_end_to_end_mock():
    mock_sem = MockSemanticRetriever()
    hr = HybridRetriever(
        semantic_retriever=mock_sem,
        fusion_strategy="rrf",
        final_top_k=2
    )

    corpus = [
        {"chunk_id": "chunk_1", "doc_name": "billing.txt", "text": "The invoice must be remitted in thirty days."},
        {"chunk_id": "chunk_2", "doc_name": "warranty.txt", "text": "Hardware coverage valid for twenty-four months."},
        {"chunk_id": "chunk_3", "doc_name": "hr.txt", "text": "Vacation requests submitted through HR portal."},
    ]
    hr.add_chunks_directly(corpus)

    results = hr.retrieve("invoice remit thirty days", top_k=2)
    assert len(results) <= 2
    assert len(results) > 0
    # chunk_1 should rank #1 due to strong lexical and semantic relevance
    assert results[0]["chunk_id"] == "chunk_1"
    assert "score" in results[0]
    assert "bm25_rank" in results[0]


def test_candidate_union_promotes_out_of_top_items():
    # Demonstrates separate candidate top-k vs final top-k
    bm25 = BM25Retriever()
    mock_sem = MockSemanticRetriever()
    hr = HybridRetriever(
        bm25_retriever=bm25,
        semantic_retriever=mock_sem,
        bm25_top_k=5,
        semantic_top_k=5,
        final_top_k=3
    )

    chunks = [
        {"chunk_id": f"chunk_{i}", "doc_name": "doc", "text": f"Document chunk text number {i}."}
        for i in range(1, 10)
    ]
    hr.add_chunks_directly(chunks)

    res = hr.retrieve("chunk text 1", top_k=3)
    assert len(res) == 3


def test_weighted_vs_rrf_strategies():
    mock_sem = MockSemanticRetriever()
    hr_rrf = HybridRetriever(semantic_retriever=mock_sem, fusion_strategy="rrf")
    hr_wt = HybridRetriever(semantic_retriever=mock_sem, fusion_strategy="weighted", alpha=0.7)

    doc = "Intel Core i5-12450H CPU hardware latency under 500 milliseconds."
    hr_rrf.add_document("spec.txt", doc)
    hr_wt.add_document("spec.txt", doc)

    res_rrf = hr_rrf.retrieve("Intel Core CPU", top_k=1)
    res_wt = hr_wt.retrieve("Intel Core CPU", top_k=1)

    assert res_rrf[0]["fusion_strategy"] == "rrf"
    assert res_wt[0]["fusion_strategy"] == "weighted"
    assert res_wt[0]["alpha"] == 0.7


def test_deterministic_repeated_queries():
    hr = HybridRetriever()
    hr.add_document("doc.txt", "Repeated query determinism verification text sequence.")

    q = "determinism verification"
    res1 = hr.retrieve(q, top_k=2)
    res2 = hr.retrieve(q, top_k=2)

    assert [r["chunk_id"] for r in res1] == [r["chunk_id"] for r in res2]
    assert [r["score"] for r in res1] == [r["score"] for r in res2]


def test_adaptive_weighting_off_by_default_and_on_explicitly():
    # 1. Verify OFF by default
    hr_default = HybridRetriever(fusion_strategy="weighted", alpha=0.5)
    assert hr_default.enable_adaptive_weighting is False

    hr_default.add_document("policy.txt", "Section 104 code POL-9999 specifies tax.")
    res_def = hr_default.retrieve("What is POL-9999?", top_k=1)
    assert res_def[0]["alpha"] == 0.5

    # 2. Verify ON explicitly
    hr_adaptive = HybridRetriever(
        fusion_strategy="weighted",
        alpha=0.5,
        enable_adaptive_weighting=True
    )
    hr_adaptive.add_document("policy.txt", "Section 104 code POL-9999 specifies tax.")

    # Query with structured ID 'POL-9999' should shift alpha toward BM25 (> 0.50)
    res_id = hr_adaptive.retrieve("What is POL-9999?", top_k=1)
    assert res_id[0]["adaptive_alpha_applied"] > 0.50

    # Conversational open-ended query should shift alpha toward Semantic (< 0.50)
    res_conv = hr_adaptive.retrieve("Under what circumstances does customer pay tax?", top_k=1)
    assert res_conv[0]["adaptive_alpha_applied"] < 0.50


def test_clear_resets_retriever_state():
    hr = HybridRetriever()
    hr.add_document("doc.txt", "Some content to be wiped.")
    assert hr.stats()["bm25"]["indexed_chunks"] == 1

    hr.clear()
    assert hr.stats()["bm25"]["indexed_chunks"] == 0
    assert hr.retrieve("content", top_k=1) == []


def test_real_person1_integration_compatibility():
    """
    Attempts to connect with Person 1's real SemanticRetriever if present.
    If unavailable, gracefully skips without failing.
    """
    person1_root = os.path.abspath(os.path.join(current_dir, "..", "..", "person1_dense_retrieval", "src"))
    if not os.path.isdir(person1_root):
        pytest.skip("Real Person 1 directory not found. Skipping compatibility test.")

    if person1_root not in sys.path:
        sys.path.insert(0, person1_root)

    try:
        from dense_retrieval.semantic_retriever import SemanticRetriever as P1SemanticRetriever
    except ImportError:
        pytest.skip("Could not import Person 1 SemanticRetriever. Skipping compatibility test.")

    # Instantiate Person 1 real retriever (uses fallback mock embedder if sentence-transformers is uninstalled)
    p1_retriever = P1SemanticRetriever(batch_size=16, allow_fallback=True)
    adapter = Person1SemanticAdapter(p1_retriever)
    assert adapter.is_available is True

    # Connect to Person 2 HybridRetriever
    hr = HybridRetriever(semantic_retriever=adapter, fusion_strategy="rrf")
    chunks = hr.add_document(
        "real_p1_test.txt",
        "The purchaser shall remit the payment within thirty calendar days of invoice."
    )
    assert len(chunks) >= 1

    fused = hr.retrieve("remit payment thirty days", top_k=1)
    assert len(fused) == 1
    assert fused[0]["chunk_id"] == "chunk_1"
    assert fused[0]["semantic_rank"] is not None
    assert fused[0]["bm25_rank"] is not None
