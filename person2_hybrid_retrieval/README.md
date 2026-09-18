# ProofMesh Hybrid Retrieval Subsystem (Person 2 Workstream)

An independent, zero-conflict hybrid retrieval and score fusion engine for the ProofMesh Evidence-Bound RAG system.

---

## 1. Overview & Purpose

In retrieval-augmented generation (RAG), lexical (BM25) and dense semantic retrieval models offer complementary strengths:
- **BM25 Lexical Retrieval**: Exceptional precision for exact terminology, structured identifiers (e.g. `POL-4092`), numbers, dates, and rare tokens. Struggles with paraphrasing, conceptual synonyms, and vocabulary mismatch.
- **Dense Semantic Retrieval**: Strong conceptual understanding, cross-vocabulary matching, and semantic paraphrasing. Struggles with exact codes, rare identifiers, and numerical precision.

**Person 2's Responsibility**: Intelligent combination of these two independent candidate streams using mathematically sound score normalization, Reciprocal Rank Fusion (RRF), weighted convex fusion, candidate union, deduplication, and deterministic ranking.

---

## 2. Directory Layout

```
person2_hybrid_retrieval/
├── src/
│   └── hybrid_retrieval/
│       ├── __init__.py               # Package exports
│       ├── models.py                 # HybridChunk, ScoredChunk, HybridResult
│       ├── bm25_retriever.py         # Standalone BM25 (60/15 chunking, rank_bm25 + fallback)
│       ├── semantic_adapter.py       # Protocol, MockSemanticRetriever, Person1Adapter
│       ├── score_normalizer.py       # Min-Max, Rank, and Z-Score normalizers
│       ├── fusion.py                 # Weighted Normalized & RRF Fusion Engine
│       └── hybrid_retriever.py       # Orchestrator & optional Query-Adaptive weighting
├── tests/
│   ├── test_bm25_retriever.py        # 9 unit tests for BM25 retriever
│   ├── test_fusion.py                # 12 unit tests for normalizers and fusion
│   └── test_hybrid_retriever.py      # 9 integration tests for HybridRetriever
├── examples/
│   └── hybrid_demo.py                # Prescribed 4-chunk demo on 3 query types
├── benchmarks/
│   ├── evaluation_dataset.py         # 15-chunk corpus, 12 multi-category test queries
│   └── benchmark_fusion.py           # Multi-strategy benchmark (Recall, MRR, Latency)
├── docs/
│   └── architecture.md               # Detailed query flow diagrams and math formulas
├── requirements.txt                  # rank-bm25, numpy, pytest
├── README.md                         # This documentation
└── .gitignore                        # Python cache ignore rules
```

---

## 3. Core Technical Principles

### 3.1 Why Raw Score Addition is Invalid
BM25 scores are unbounded positive frequencies ($0.5$ to $25.0+$) based on TF-IDF heuristics. Dense semantic scores are bounded cosine values ($0.0$ to $1.0$). Adding raw scores without normalization causes BM25 to completely overpower the semantic signal.

### 3.2 Score Normalization Methods
1. **Min-Max Normalization**:
   $$\tilde{s}_i = \frac{s_i - \min(S)}{\max(S) - \min(S)}$$
   Guarded against division-by-zero when all scores are equal ($\max = \min$).
2. **Rank Normalization**: Percentile rank mapping: $\text{norm\_rank} = (N - \text{rank} + 1) / N$.
3. **Z-Score Normalization**: Zero-mean unit-variance scaling with zero-variance safety.

### 3.3 Fusion Strategies
- **Strategy A: Weighted Normalized Fusion**:
  $$S_{\text{hybrid}} = \alpha \cdot \tilde{S}_{\text{BM25}} + (1 - \alpha) \cdot \tilde{S}_{\text{semantic}}$$
  Configurable $\alpha \in [0.0, 1.0]$.
- **Strategy B: Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF}(c) = \sum_{m} \frac{1}{k + \text{rank}_m(c)}$$
  Configurable smoothing constant $k > 0$ (default $k = 60$). RRF is completely score-scale invariant.

### 3.4 Candidate Union & Deduplication
- Operates on $\text{Union}(\text{BM25 candidates}, \text{Semantic candidates})$, never intersection.
- Chunks present in only one source are handled gracefully (missing score treated as $0.0$).
- Chunks appearing in both sources receive combined fusion scores and are deduplicated by `chunk_id`.

### 3.5 Deterministic Ranking
- Primary sort: `score` descending.
- Secondary tie-breaker: `chunk_id` ascending lexicographically.

---

## 4. Query-Adaptive Weighting (Optional Stage 2)

- **Mandatory Default**: `enable_adaptive_weighting = False`.
- When explicitly enabled (`enable_adaptive_weighting=True`), uses a fast, deterministic rule-based heuristic:
  - BM25 weight boosted for: Structured IDs (`POL-4092`), numbers, dates (`24 months`, `2025`), quoted terms (`"invoice reference"`).
  - Semantic weight boosted for: Open conversational questions (`"under what circumstances"`, `"how long"`), descriptive phrasing.
- Does **NOT** use an LLM or neural classifier.

---

## 5. Benchmark Results (Controlled Evaluation Dataset)

Evaluated across 15 reference document chunks and 12 multi-category queries:

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | Average Query Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. BM25 Only** | 100.0% | 100.0% | 100.0% | 1.0000 | 0.036 ms |
| **B. Dense Semantic Only** | 90.9% | 100.0% | 100.0% | 0.9394 | 0.108 ms |
| **C. Weighted Fusion ($\alpha=0.5$)** | 100.0% | 100.0% | 100.0% | 1.0000 | 0.267 ms |
| **D. RRF Fusion ($k=60$)** | 100.0% | 100.0% | 100.0% | 1.0000 | 0.180 ms |
| **E. Query-Adaptive Fusion** | 100.0% | 100.0% | 100.0% | 1.0000 | 0.221 ms |

*Note: Latency measured on local Intel CPU. Fusion adds under 0.2 ms per query.*

---

## 6. Usage Example

```python
from hybrid_retrieval import HybridRetriever, BM25Retriever, MockSemanticRetriever

# 1. Initialize Hybrid Retriever
retriever = HybridRetriever(
    fusion_strategy="rrf",
    rrf_k=60,
    bm25_top_k=5,
    semantic_top_k=5,
    final_top_k=3,
    enable_adaptive_weighting=False
)

# 2. Add documents or chunks
corpus = [
    {"chunk_id": "chunk_1", "doc_name": "terms.pdf", "text": "The purchaser shall remit invoice in 30 days."},
    {"chunk_id": "chunk_2", "doc_name": "warranty.pdf", "text": "Hardware warranty valid for 24 months."}
]
retriever.add_chunks_directly(corpus)

# 3. Retrieve
results = retriever.retrieve("When does the customer need to pay?", top_k=2)
for r in results:
    print(f"Ranked: {r['chunk_id']} | Score: {r['score']:.4f} | BM25 Rank: {r.get('bm25_rank')}, Sem Rank: {r.get('semantic_rank')}")
```

---

## 7. Verification & Testing

Run all unit and integration tests:
```bash
python -m pytest person2_hybrid_retrieval/tests -v
```

Run the interactive side-by-side demonstration:
```bash
python person2_hybrid_retrieval/examples/hybrid_demo.py
```

Run the benchmark suite:
```bash
python person2_hybrid_retrieval/benchmarks/benchmark_fusion.py
```

---

## 8. Limitations & Future Integration

- **Subsystem Scope**: Operates strictly within `person2_hybrid_retrieval/`. Does not modify `proofmesh/` or `person1_dense_retrieval/`.
- **Intentionally Omitted**: No neural cross-encoders, no LLM-based rerankers, and no cloud dependencies were introduced to maintain ultra-fast sub-millisecond CPU execution.
- **Future ProofMesh Integration**: The output schema strictly mirrors `proofmesh/retriever.py`'s `DocumentChunk`, enabling drop-in integration into ProofMesh's `EvidenceGate` when team integration begins.
