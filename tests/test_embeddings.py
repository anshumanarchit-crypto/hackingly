"""
Unit tests for SemanticEmbedder and embedding operations.
"""

import sys
import os
import pytest
import numpy as np

try:
    from proofmesh.dense_retrieval.embeddings import SemanticEmbedder, DeterministicMockBackend
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "person1_dense_retrieval", "src")))
    from dense_retrieval.embeddings import SemanticEmbedder, DeterministicMockBackend


def test_embedder_initialization_lazy_loading():
    """Verify embedder initializes without immediately loading the model (lazy loading)."""
    embedder = SemanticEmbedder(model_name="mock", device="cpu")
    assert not embedder.is_loaded
    assert embedder.model_name == "mock"
    assert embedder.device == "cpu"

    # Accessing dimension or encoding triggers load
    dim = embedder.get_dimension()
    assert embedder.is_loaded
    assert dim == 384


def test_single_text_embedding():
    """Verify single text encoding returns a 1D float32 normalized array."""
    embedder = SemanticEmbedder(model_name="mock", device="cpu", normalize_embeddings=True)
    text = "The invoice must be settled within 30 days."
    vec = embedder.encode_text(text)

    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape[0] == 384
    assert vec.dtype == np.float32

    # Verify L2 normalization
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, rel=1e-4) == 1.0


def test_batch_embeddings():
    """Verify batch encoding returns a 2D float32 array with correct shape."""
    embedder = SemanticEmbedder(model_name="mock", device="cpu", batch_size=2)
    texts = [
        "First document sentence.",
        "Second document sentence.",
        "Third document sentence.",
    ]
    vectors = embedder.encode_texts(texts, batch_size=2)

    assert isinstance(vectors, np.ndarray)
    assert vectors.ndim == 2
    assert vectors.shape == (3, 384)
    assert vectors.dtype == np.float32

    # Check normalization across all rows
    for i in range(3):
        norm = np.linalg.norm(vectors[i])
        assert pytest.approx(norm, rel=1e-4) == 1.0


def test_embedding_dimension_consistency():
    """Verify query and document embeddings have identical dimensions."""
    embedder = SemanticEmbedder(model_name="mock")
    q_vec = embedder.encode_text("Query text")
    docs_vec = embedder.encode_texts(["Doc 1", "Doc 2"])

    assert q_vec.shape[0] == docs_vec.shape[1] == embedder.get_dimension()


def test_model_reuse():
    """Verify that multiple encode calls reuse the same underlying model instance."""
    embedder = SemanticEmbedder(model_name="mock")
    embedder.encode_text("First query")
    model_ref_1 = embedder._model

    embedder.encode_text("Second query")
    model_ref_2 = embedder._model

    assert model_ref_1 is model_ref_2
    assert embedder.stats()["encode_call_count"] == 2


def test_empty_and_whitespace_input():
    """Verify empty or whitespace strings do not raise crashes."""
    embedder = SemanticEmbedder(model_name="mock")
    empty_vec = embedder.encode_text("")
    ws_vec = embedder.encode_text("   \n\t  ")

    assert empty_vec.shape == (384,)
    assert ws_vec.shape == (384,)

    empty_batch = embedder.encode_texts([])
    assert empty_batch.shape == (0, 384)


def test_mock_backend_explicit_identification():
    """Verify that mock mode is clearly identifiable and never disguised as production model."""
    embedder = SemanticEmbedder(model_name="mock")
    stats = embedder.stats()

    assert stats["is_mock"] is True
    assert stats["backend"] == "mock"
    assert embedder.is_mock is True


def test_real_backend_when_available():
    """
    Test real SentenceTransformer backend if sentence-transformers is installed.
    If not installed, test documents that mock fallback is active.
    """
    try:
        import sentence_transformers
        has_st = True
    except ImportError:
        has_st = False

    if not has_st:
        # Verify fallback triggers cleanly
        embedder = SemanticEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            allow_fallback=True,
        )
        assert embedder.is_mock is True
        pytest.skip("sentence-transformers not installed; mock fallback verified.")
    else:
        embedder = SemanticEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            allow_fallback=False,
        )
        assert embedder.backend_type == "sentence-transformers"
        assert not embedder.is_mock
        vec = embedder.encode_text("Testing real model")
        assert vec.shape == (384,)
