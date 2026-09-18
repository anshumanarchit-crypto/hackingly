# ProofMesh Hybrid Retrieval Architecture (Person 2 Workstream)

## 1. System Overview

The **Hybrid Retrieval and Retrieval Optimization** subsystem (Person 2 Workstream) bridges lexical precision and dense semantic recall into a unified, deterministic, and explainable evidence retrieval pipeline for ProofMesh.

```
                    USER QUERY
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
   BM25 RETRIEVER             SEMANTIC RETRIEVER
 (60/15 chunking, Okapi)   (Person 1 Adapter / Mock)
         │                             │
         ▼                             ▼
  BM25 Raw Candidates        Semantic Raw Candidates
   (top bm25_top_k)           (top semantic_top_k)
         │                             │
         └──────────────┬──────────────┘
                        ▼
               SCORE NORMALIZATION
             (Min-Max, Rank, Z-Score)
                        │
                        ▼
                  FUSION ENGINE
                 /             \
      Weighted Fusion       Reciprocal Rank Fusion
    alpha * S_b + (1-a)*S_s       sum(1 / (k + rank))
                 \             /
                  ▼           ▼
                   CANDIDATE UNION
             union(BM25, Semantic Candidates)
                          │
                          ▼
                    DEDUPLICATION
              (Merged by unique chunk_id)
                          │
                          ▼
                DETERMINISTIC RANKING
            (Score desc, chunk_id asc tie-break)
                          │
                          ▼
                     FINAL TOP-K
               (Explainable Hybrid Results)
```

---

## 2. Core Problem: Disparate Score Spaces

BM25 scores and dense semantic vector similarities operate in fundamentally incompatible numerical spaces:
- **BM25 Lexical Scores**: Unbounded non-negative numbers ($s \in [0, \infty)$), often ranging from $0.5$ to $25.0+$ based on term frequency (TF), document length normalization, and inverse document frequency (IDF).
- **Dense Cosine Scores**: Bounded angular similarities ($s \in [-1.0, 1.0]$ or $[0.0, 1.0]$).

> [!WARNING]
> **Prohibition of Raw Addition**:
> Directly computing `bm25_score + semantic_score` is mathematically invalid. Without normalization or rank-based fusion, the unbounded BM25 score completely dominates and distorts the dense signal.

---

## 3. Mathematical Foundations

### 3.1 Min-Max Normalization

Linear scaling to the standard unit interval $[0.0, 1.0]$:

$$\tilde{s}_i = \frac{s_i - \min(S)}{\max(S) - \min(S)}$$

#### Constant-Score Safety
When all candidates in a retrieval list share identical scores ($\max(S) = \min(S)$), division-by-zero is prevented:
- If $\max(S) > 0$: $\tilde{s}_i = 1.0$ (all items equally relevant).
- If $\max(S) \le 0$: $\tilde{s}_i = 0.0$.

---

### 3.2 Strategy A: Weighted Normalized Fusion

Linear convex combination of normalized scores:

$$S_{\text{hybrid}}(c) = \alpha \cdot \tilde{S}_{\text{BM25}}(c) + (1 - \alpha) \cdot \tilde{S}_{\text{semantic}}(c)$$

- $\alpha \in [0.0, 1.0]$ controls the balance:
  - $\alpha = 1.0$: Pure lexical BM25 ranking.
  - $\alpha = 0.0$: Pure dense semantic ranking.
  - $\alpha = 0.5$: Balanced 50/50 weighting.

---

### 3.3 Strategy B: Reciprocal Rank Fusion (RRF)

Score-independent, rank-based aggregation:

$$\text{RRF}(c) = \sum_{m \in \{\text{BM25}, \text{Semantic}\}} \frac{1}{k + \text{rank}_m(c)}$$

Where:
- $k$ is a smoothing constant (default $k = 60$).
- $\text{rank}_m(c)$ is the 1-based ordinal rank of chunk $c$ in provider $m$.
- If a chunk $c$ does not appear in the top candidate list of provider $m$, its contribution from that provider is $0$.

**Why RRF is Robust**: RRF bypasses score calibration issues entirely because it depends only on relative orderings, not raw score magnitudes or distributions.

---

## 4. Candidate Union & Deduplication

Rather than taking the intersection of BM25 and semantic results, the system operates on the **Candidate Union**:

$$\mathcal{C} = \text{Top}_{K_{\text{bm25}}}(\text{BM25}) \cup \text{Top}_{K_{\text{sem}}}(\text{Semantic})$$

### Deduplication Rules:
1. Every chunk is identified uniquely by its `chunk_id`.
2. When a chunk appears in both retrieval streams, its ranks and raw scores are preserved in explainability fields, and its fusion score reflects contributions from both sources.
3. When a chunk appears in only one stream (e.g. found only by semantic paraphrase), the missing stream's contribution is treated as 0 without throwing an error.
4. Each chunk appears **at most once** in the final result list.

---

## 5. Decoupled Candidate Sizing

To ensure high recall, candidate retrieval sizes are separated from final presentation top-k:
- `bm25_top_k`: Number of lexical candidates fetched (default: 5).
- `semantic_top_k`: Number of dense candidates fetched (default: 5).
- `final_top_k`: Number of final fused results returned (default: 3).

This decoupling ensures that a chunk ranked at position 4 or 5 in one retriever can be combined with a signal from the other retriever and promoted into the final top-3.

---

## 6. Deterministic Final Ordering

To guarantee idempotent, testable behavior:
1. **Primary Sort Key**: `hybrid_score` descending.
2. **Secondary Tie-Breaker**: `chunk_id` ascending lexicographically.

Repeated identical queries against identical index states will always produce identical output orders.

---

## 7. Optional Stage 2: Query-Adaptive Weighting

- **Status**: Optional feature, **OFF by default** (`enable_adaptive_weighting = False`).
- **Heuristic**:
  - BM25 signals: Structured alphanumeric IDs (`POL-4092`), numbers/dates (`24 months`, `2025`), quoted exact phrases (`"invoice reference"`).
  - Semantic signals: Conversational question openers (`"under what circumstances"`, `"how long"`), descriptive conceptual phrasing.
  - Dynamically shifts $\alpha$ in $[0.10, 0.90]$ using deterministic rules with zero LLM or neural classifier overhead.

---

## 8. Eventual Shared ProofMesh Integration Plan

This subsystem operates completely in isolation inside `person2_hybrid_retrieval/`. During the final team integration phase:
1. `proofmesh/retriever.py` will import and wrap `person2_hybrid_retrieval.src.hybrid_retrieval.HybridRetriever`.
2. Person 1's real `SemanticRetriever` will be supplied via `Person1SemanticAdapter`.
3. Output dicts from `HybridRetriever.retrieve()` match the existing ProofMesh contract (`chunk_id`, `doc_name`, `text`, `score`), ensuring zero breaking changes to `evidence_gate.py` or `answering_engine.py`.
