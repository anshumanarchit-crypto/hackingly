"""
Integration and functional tests for SemanticRetriever.
Validates ProofMesh compatibility, deterministic ranking, filtering, and edge cases.
"""

import sys
import os
import pytest

# Ensure person1_dense_retrieval/src is in sys.path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

from dense_retrieval.models import SemanticChunk, ChunkRecord
from dense_retrieval.embeddings import SemanticEmbedder
from dense_retrieval.semantic_retriever import SemanticRetriever


@pytest.fixture
def retriever():
    """Provides a fresh retriever using the deterministic mock embedder."""
    embedder = SemanticEmbedder(model_name="mock", device="cpu")
    return SemanticRetriever(embedder=embedder)


def test_add_document_chunking(retriever):
    """Test that add_document splits text into overlapping chunks and indexes them."""
    long_text = " ".join([f"word{i}" for i in range(120)])
    created_chunks = retriever.add_document("doc_alpha", long_text)

    assert len(created_chunks) > 1
    assert retriever.stats()["indexed_chunks"] == len(created_chunks)
    assert created_chunks[0].chunk_id == "chunk_1"
    assert created_chunks[0].doc_name == "doc_alpha"


def test_add_chunks_directly_preserves_identities(retriever):
    """Test indexing pre-chunked items preserves chunk_id, doc_name, text, and metadata."""
    chunks = [
        SemanticChunk(
            chunk_id="chunk_custom_101",
            doc_name="policy_contract.txt",
            text="The purchaser shall remit payment within 30 days.",
            metadata={"priority": "high"},
        ),
        {
            "chunk_id": "chunk_custom_102",
            "doc_name": "warranty.pdf",
            "text": "The device warranty is valid for 24 months.",
        },
        ChunkRecord(
            chunk_id="chunk_custom_103",
            source="hr_rules.txt",
            text="Employees must submit vacation requests via the portal.",
        ),
    ]

    retriever.add_chunks_directly(chunks)
    assert len(retriever.chunks) == 3

    # Check first chunk
    c1 = retriever.chunks[0]
    assert c1.chunk_id == "chunk_custom_101"
    assert c1.doc_name == "policy_contract.txt"
    assert c1.text == "The purchaser shall remit payment within 30 days."
    assert c1.metadata == {"priority": "high"}

    # Check dict chunk
    c2 = retriever.chunks[1]
    assert c2.chunk_id == "chunk_custom_102"
    assert c2.doc_name == "warranty.pdf"

    # Check ChunkRecord
    c3 = retriever.chunks[2]
    assert c3.chunk_id == "chunk_custom_103"
    assert c3.doc_name == "hr_rules.txt"


def test_top_k_retrieval_and_result_schema(retriever):
    """Verify top_k limiting and output dictionary schema."""
    chunks = [
        {"chunk_id": f"chunk_{i}", "doc_name": "test_doc", "text": f"Text sample number {i}"}
        for i in range(10)
    ]
    retriever.add_chunks_directly(chunks)

    results = retriever.retrieve("sample number 3", top_k=3)
    assert len(results) == 3

    # Verify ProofMesh schema compatibility
    for r in results:
        assert "chunk_id" in r
        assert "doc_name" in r
        assert "text" in r
        assert "score" in r
        assert isinstance(r["score"], float)


def test_score_ordering_descending(retriever):
    """Verify results are sorted by score descending."""
    chunks = [
        {"chunk_id": "chunk_a", "doc_name": "d1", "text": "solar energy panels sun light electricity"},
        {"chunk_id": "chunk_b", "doc_name": "d2", "text": "deep ocean submarine underwater aquatic fish"},
        {"chunk_id": "chunk_c", "doc_name": "d3", "text": "solar panels photovoltaic cells sunlight power"},
    ]
    retriever.add_chunks_directly(chunks)

    results = retriever.retrieve("solar panels sunlight", top_k=3)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_deterministic_tie_breaking(retriever):
    """Verify tie-breaking: when scores are identical, chunk_id ascending breaks ties."""
    # Identical texts will yield identical similarity scores
    chunks = [
        {"chunk_id": "chunk_z", "doc_name": "d1", "text": "exact duplicate text content"},
        {"chunk_id": "chunk_a", "doc_name": "d2", "text": "exact duplicate text content"},
        {"chunk_id": "chunk_m", "doc_name": "d3", "text": "exact duplicate text content"},
    ]
    retriever.add_chunks_directly(chunks)

    results = retriever.retrieve("duplicate text", top_k=3)
    assert len(results) == 3
    # Scores are equal
    assert pytest.approx(results[0]["score"], rel=1e-4) == results[1]["score"]
    assert pytest.approx(results[1]["score"], rel=1e-4) == results[2]["score"]

    # Tie break should be chunk_a, chunk_m, chunk_z
    ids = [r["chunk_id"] for r in results]
    assert ids == ["chunk_a", "chunk_m", "chunk_z"]


def test_min_score_filtering(retriever):
    """Verify that min_score filters out items below the threshold."""
    chunks = [
        {"chunk_id": "chunk_relevant", "doc_name": "d1", "text": "quantum computing cryptography physics"},
        {"chunk_id": "chunk_irrelevant", "doc_name": "d2", "text": "baking chocolate chip cookies recipe oven"},
    ]
    retriever.add_chunks_directly(chunks)

    all_results = retriever.retrieve("quantum physics", top_k=2)
    assert len(all_results) == 2

    # Filter with a threshold higher than the irrelevant chunk's score
    top_score = all_results[0]["score"]
    filtered_results = retriever.retrieve("quantum physics", top_k=2, min_score=top_score - 0.05)
    assert len(filtered_results) == 1
    assert filtered_results[0]["chunk_id"] == "chunk_relevant"


def test_empty_index_handling(retriever):
    """Verify searching an empty index returns empty list without error."""
    assert retriever.retrieve("any query", top_k=5) == []


def test_empty_query_handling(retriever):
    """Verify empty or whitespace query returns empty list without error."""
    retriever.add_chunks_directly([{"chunk_id": "c1", "doc_name": "d1", "text": "some text"}])
    assert retriever.retrieve("", top_k=3) == []
    assert retriever.retrieve("   \n\t ", top_k=3) == []


def test_invalid_top_k(retriever):
    """Verify top_k <= 0 returns empty list and non-integer raises TypeError."""
    retriever.add_chunks_directly([{"chunk_id": "c1", "doc_name": "d1", "text": "some text"}])
    assert retriever.retrieve("text", top_k=0) == []
    assert retriever.retrieve("text", top_k=-2) == []

    with pytest.raises(TypeError):
        retriever.retrieve("text", top_k="three")  # type: ignore


def test_pii_placeholder_preservation(retriever):
    """Verify that protected PII tokens such as [PERSON_001] remain intact."""
    pii_text = "[PERSON_001] submitted invoice [ORG_99] on [DATE_01] for medical procedure."
    retriever.add_chunks_directly([
        {"chunk_id": "pii_chunk", "doc_name": "masked_records.txt", "text": pii_text}
    ])

    results = retriever.retrieve("medical procedure invoice", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "pii_chunk"
    assert "[PERSON_001]" in results[0]["text"]
    assert "[ORG_99]" in results[0]["text"]
    assert results[0]["text"] == pii_text


def test_repeated_retrieval_determinism(retriever):
    """Verify identical repeated queries against the same index produce strictly identical output."""
    chunks = [
        {"chunk_id": f"chunk_{i}", "doc_name": "d", "text": f"Unique content sentence {i} with key data"}
        for i in range(5)
    ]
    retriever.add_chunks_directly(chunks)

    run_1 = retriever.retrieve("content sentence key data", top_k=3)
    run_2 = retriever.retrieve("content sentence key data", top_k=3)

    assert run_1 == run_2


def test_clear_and_stats(retriever):
    """Verify clear() resets the index and stats() provides accurate diagnostics."""
    retriever.add_document("doc1", "Alpha beta gamma delta epsilon zeta eta theta")
    assert retriever.stats()["indexed_chunks"] > 0

    retriever.clear()
    stats = retriever.stats()
    assert stats["indexed_chunks"] == 0
    assert stats["embedding_matrix_shape"] == [0, 384]
    assert retriever.retrieve("Alpha", top_k=1) == []


def test_aliases_search_and_index(retriever):
    """Verify search() and index() aliases behave identically to retrieve() and add_chunks_directly()."""
    retriever.index([{"chunk_id": "alias_1", "doc_name": "d", "text": "Testing alias method."}])
    results = retriever.search("Testing alias", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "alias_1"


def test_paraphrase_and_semantic_retrieval():
    """
    Demonstrate paraphrase retrieval.
    If real sentence-transformers model is available, verify semantic matching over lexical mismatch.
    If mock backend is active, verify that retrieval completes deterministically with valid scores.
    """
    retriever = SemanticRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        allow_fallback=True,
    )
    chunks = [
        {
            "chunk_id": "chunk_payment",
            "doc_name": "contract.pdf",
            "text": "The purchaser shall remit the outstanding invoice within thirty calendar days.",
        },
        {
            "chunk_id": "chunk_warranty",
            "doc_name": "warranty.pdf",
            "text": "The hardware warranty remains valid for twenty-four months from the original purchase date.",
        },
        {
            "chunk_id": "chunk_leave",
            "doc_name": "hr.pdf",
            "text": "Employees must submit vacation requests through the HR portal.",
        },
    ]
    retriever.add_chunks_directly(chunks)

    # Paraphrase query: zero exact word overlap with chunk_payment
    query = "When does the customer need to pay?"
    results = retriever.retrieve(query, top_k=1)

    assert len(results) == 1
    assert "score" in results[0]
    assert isinstance(results[0]["score"], float)

    # Under real sentence-transformers neural model, chunk_payment must rank #1
    if not retriever.embedder.is_mock:
        assert results[0]["chunk_id"] == "chunk_payment"


def test_unrelated_query_filtering():
    """Verify unrelated queries receive low scores and can be rejected by min_score."""
    retriever = SemanticRetriever(model_name="mock")
    chunks = [
        {"chunk_id": "c1", "doc_name": "it.txt", "text": "Server maintenance and Linux cloud deployment procedures."},
        {"chunk_id": "c2", "doc_name": "fin.txt", "text": "Quarterly earnings report and balance sheet analysis."},
    ]
    retriever.add_chunks_directly(chunks)

    # Unrelated medical query
    unrelated_results = retriever.retrieve("What is the patient's blood type?", top_k=2, min_score=0.90)
    assert len(unrelated_results) == 0

