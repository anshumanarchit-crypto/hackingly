# Person 1: Dense Semantic Retrieval Subsystem

Standalone, zero-dependency-conflict dense semantic vector retrieval component for the **ProofMesh** project.

---

## 1. Purpose & Relationship to ProofMesh

The ProofMesh architecture currently utilizes lexical BM25 retrieval within `proofmesh/retriever.py`. While BM25 excels at keyword matching and exact token lookups, it suffers from the vocabulary mismatch problem—it fails when users query with synonyms, natural phrasing, or paraphrases.

This module provides **Dense Semantic Retrieval (Feature 2 - Person 1)**:
- Encodes document chunks and queries into 384-dimensional dense vector embeddings.
- Computes vectorized cosine similarity in memory using NumPy.
- Ranks candidate chunks based on deep conceptual semantic similarity rather than token overlap.
- Designed as a completely isolated subsystem under `person1_dense_retrieval/` to ensure zero merge conflicts with parallel teammates working on `proofmesh/`.

---

## 2. Architecture & Data Flow

```
Document Chunks ──► SentenceTransformer ──► Dense Vectors (Normalized) ──► In-Memory Matrix Index
                                                                                   │
User Query      ──► SentenceTransformer ──► Query Vector (Normalized)   ──► Cosine Similarity (Dot Product)
                                                                                   │
Ranked Results  ◄── ProofMesh Schema    ◄── Min-Score Filter & Top-K    ◄── Deterministic Tie-Breaker
```

---

## 3. Embedding Model Selection

### Production / Evaluation Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding Dimension**: 384
- **Parameters**: ~22.7M (extremely lightweight)
- **Target Device**: CPU (`device="cpu"`)
- **Latency**: ~10–25ms per query on modern x86_64 CPUs
- **Offline Capability**: Once weights are cached locally, inference requires zero internet access.

### Development & Test Fallback Mode
- **Backend**: `DeterministicMockBackend` (`model_name="mock"`)
- **Purpose**: Enables rapid offline unit testing and development in environments where `sentence-transformers` is not pre-installed.
- **Implementation**: Uses deterministic feature hashing over token and character n-grams into a normalized 384-dimensional vector.
- **Reporting**: The embedder clearly reports `backend: "mock"` and `is_mock: True` in diagnostic stats and benchmarks.

---

## 4. Installation & Setup

All dependencies are defined locally in `person1_dense_retrieval/requirements.txt`. Do NOT install dependencies globally or modify root-level dependency files.

To install dependencies in a dedicated local environment:
```powershell
pip install -r person1_dense_retrieval/requirements.txt
```

---

## 5. Public API Reference

The primary class is `SemanticRetriever` from `dense_retrieval`:

```python
from dense_retrieval import SemanticRetriever, SemanticChunk

# 1. Initialize retriever
retriever = SemanticRetriever(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
    batch_size=32,
    min_score=None
)

# 2. Add raw document text (automatically chunked ~60 words, 15 word overlap)
retriever.add_document(
    doc_name="terms.pdf",
    text="The purchaser shall remit the outstanding invoice within thirty calendar days."
)

# 3. Or index pre-chunked items directly
retriever.add_chunks_directly([
    {
        "chunk_id": "chunk_custom_1",
        "doc_name": "warranty.pdf",
        "text": "The hardware warranty remains valid for twenty-four months."
    }
])

# 4. Semantic Search / Retrieve
results = retriever.retrieve(
    query="When does the customer have to pay?",
    top_k=3,
    min_score=0.20
)

# 5. Output Format (ProofMesh schema compatible)
for res in results:
    print(res["chunk_id"], res["score"], res["text"])
```

### Retrieval Result Format
```json
[
  {
    "chunk_id": "chunk_1",
    "doc_name": "terms.pdf",
    "text": "The purchaser shall remit the outstanding invoice within thirty calendar days.",
    "score": 0.8421
  }
]
```

### Deterministic Ranking
- Primary sort: `score` descending.
- Secondary tie-break: `chunk_id` lexicographically ascending.

---

## 6. Verification & Testing

### Running Tests
Execute the pytest suite:
```powershell
python -m pytest person1_dense_retrieval/tests -v
```

All 20 test specifications are validated:
- Lazy loading & model caching
- Single and batch embeddings
- Dimension consistency (384-d)
- Vector L2 normalization
- Model reuse verification
- Document chunking & pre-chunked indexing
- Top-k limiting & ProofMesh schema preservation
- Score ordering & deterministic tie-breaking
- Min-score filtering
- Handling empty index, empty queries, and invalid `top_k`
- Preservation of PII placeholders (e.g. `[PERSON_001]`)
- Paraphrase and semantic discrimination

### Running the Demonstration
```powershell
python person1_dense_retrieval/examples/semantic_demo.py
```

### Running the Benchmark
```powershell
python person1_dense_retrieval/benchmarks/benchmark_semantic_retrieval.py
```

---

## 7. Integration Contract for Person 2 (Hybrid Fusion)

When Person 2 integrates Dense Semantic Retrieval into ProofMesh's `HybridRetriever`:
1. `SemanticRetriever` produces standardized dictionaries:
   `{"chunk_id": str, "doc_name": str, "text": str, "score": float}`.
2. Person 2 can retrieve BM25 lexical scores and Semantic similarity scores concurrently.
3. Person 2 fuses the rankings via Reciprocal Rank Fusion (RRF) or convex score combination:
   $$\text{Score}_{\text{hybrid}} = \alpha \cdot \text{Score}_{\text{BM25}} + (1 - \alpha) \cdot \text{Score}_{\text{semantic}}$$
4. No modification of `person1_dense_retrieval` internals is needed.

---

## 8. What is Intentionally NOT Implemented

To maintain strict workstream boundaries and prevent merge conflicts, this module intentionally excludes:
- **BM25 or Lexical Search** (owned by existing ProofMesh retriever).
- **Hybrid Score Fusion** (owned by Person 2).
- **LLM Prompting or Generation** (owned by downstream ProofMesh pipeline).
- **PII Detection or Masking** (already owned by ProofMesh `pii_masker.py`).
- **Evidence Gate or Claim Verification** (owned by ProofMesh core).
- **External Vector DBs (Chroma/FAISS)** (in-memory NumPy vector indexing satisfies offline requirements).
