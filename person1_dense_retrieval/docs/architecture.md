# Dense Semantic Retrieval Subsystem - Architecture

## 1. Overview & Workstream Boundary

This module forms **Person 1's Workstream: Dense Semantic Retrieval** for the ProofMesh project.

It provides a standalone, offline dense vector retrieval subsystem designed to complement the existing BM25 lexical retriever in ProofMesh. To prevent merge conflicts in a parallel team workflow, this subsystem operates in complete isolation within `person1_dense_retrieval/` and does not import or alter any files in `proofmesh/`.

---

## 2. Data Flow Architecture

The internal dense semantic retrieval pipeline operates as follows:

```
                  ┌───────────────────────────────┐
                  │ Document Chunks (Text Input)  │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │   SentenceTransformer Model   │
                  │ (all-MiniLM-L6-v2 / Mock CPU) │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │    Dense Vectors (float32)    │
                  │   L2-Normalized (D = 384)     │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │   In-Memory Matrix Index      │
                  │       (N x D matrix)          │
                  └───────────────┬───────────────┘
                                  │
      ┌────────────────┐          │
      │   User Query   │          │
      └───────┬────────┘          │
              │                   │
              ▼                   │
      ┌────────────────┐          │
      │ Query Vector   │          │
      │  (1 x D float) │          │
      └───────┬────────┘          │
              │                   │
              ▼                   ▼
      ┌───────────────────────────────────────────┐
      │         Vectorized Cosine Similarity      │
      │          scores = Matrix @ q_vec          │
      └───────────────────┬───────────────────────┘
                          │
                          ▼
      ┌───────────────────────────────────────────┐
      │           Deterministic Ranking           │
      │   Primary: score DESC | Tie: chunk_id ASC │
      └───────────────────┬───────────────────────┘
                          │
                          ▼
      ┌───────────────────────────────────────────┐
      │          Threshold & Top-K Slicing        │
      │      (min_score filter, limit top_k)      │
      └───────────────────┬───────────────────────┘
                          │
                          ▼
      ┌───────────────────────────────────────────┐
      │   ProofMesh-Compatible Ranked Results     │
      │  [{"chunk_id", "doc_name", "text", ...}] │
      └───────────────────────────────────────────┘
```

---

## 3. Mathematical Operations & Index Design

### Vector Normalization
Embeddings produced by the model (both query vectors and document chunk vectors) are L2 unit-normalized:

$$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\max(\|\mathbf{v}\|_2, 10^{-12})}$$

### Vectorized Cosine Similarity
Because both document vectors $\mathbf{D} \in \mathbb{R}^{N \times D}$ and the query vector $\mathbf{q} \in \mathbb{R}^D$ are L2 unit-normalized, cosine similarity simplifies to an efficient single matrix-vector multiplication in NumPy:

$$\mathbf{s} = \mathbf{D} \mathbf{q} \in \mathbb{R}^N$$

This avoids any Python `for` loops across chunks during similarity computation, ensuring sub-millisecond search latency on standard CPU hardware.

### Deterministic Tie-Breaking
To ensure strictly reproducible retrieval across repeated executions:
- Primary key: Semantic similarity score descending ($-\text{score}$).
- Secondary key: `chunk_id` lexicographically ascending.

---

## 4. Why Dense Semantic Retrieval is Decoupled from BM25

| Feature / Dimension | Lexical Retrieval (BM25) | Dense Semantic Retrieval (Person 1) |
| :--- | :--- | :--- |
| **Matching Mechanism** | Exact token frequency & inverse document frequency | Deep contextual embedding vector proximity |
| **Strengths** | Exact acronyms, part numbers, rare proper nouns | Synonyms, paraphrases, conceptual intent |
| **Weakness** | Vocabulary mismatch (fails when query uses different words) | Can miss rare exact alphanumeric identifiers |
| **Example Match** | Fails on: "When does buyer pay?" vs "Purchaser shall remit invoice" | Successfully scores high semantic similarity |

By developing this subsystem independently, **Person 1** delivers clean semantic rankings with zero BM25 entanglements.

---

## 5. Future Integration Contract for Person 2 (Hybrid Fusion)

**Person 2** will later unify lexical BM25 and dense semantic retrieval into a single hybrid ranker.

```
                         USER QUERY
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   ┌─────────────────┐               ┌─────────────────┐
   │  ProofMesh BM25 │               │ Person 1 Dense  │
   │  Lexical Search │               │ Semantic Search │
   └────────┬────────┘               └────────┬────────┘
            │                                 │
            │ lexical_scores                  │ semantic_scores
            └────────────────┬────────────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │   Person 2: Hybrid Fusion     │
             │   (RRF / Convex Score Fusion) │
             │  α · BM25 + (1 - α) · Semantic│
             └───────────────┬───────────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │       Evidence Gate           │
             │ (Downstream ProofMesh Engine) │
             └───────────────────────────────┘
```

### Integration Handoff
Person 1's `retrieve()` returns an identical schema to `HybridRetriever.retrieve()`:
```python
[
    {
        "chunk_id": "chunk_1",
        "doc_name": "contract.pdf",
        "text": "The purchaser shall remit the outstanding invoice within thirty calendar days.",
        "score": 0.8421
    }
]
```

Person 2 can readily consume these dictionaries, map `chunk_id -> semantic_score`, and combine them with BM25 scores without requiring any internal changes to `SemanticRetriever`.
