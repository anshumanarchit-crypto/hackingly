# ProofMesh: Offline Evidence-Bound RAG Pipeline

> *"The model cannot decide what is true. Our evidence layer decides what the model is allowed to say."*

ProofMesh is an **offline-first, evidence-bound Retrieval-Augmented Generation (RAG)** engine built for high-stakes environments. It enforces strict factual grounding, deterministic contradiction detection, and post-generation claim verification, ensuring that local LLMs never hallucinate or extrapolate beyond retrieved source evidence.

---

## 🏛️ Architecture & Multi-Tier Defense

```
USER QUERY
    ↓
PII MASKING ([PERSON_001], [EMAIL_001], [AADHAAR_001])
    ↓
TRUE HYBRID RETRIEVAL (BM25 Lexical + Dense Semantic Retrieval with Score Fusion)
    ├── Lexical BM25 Scoring
    ├── Dense Vector Embedding (all-MiniLM-L6-v2 / Mock CPU)
    └── Hybrid Fusion: α · Dense + (1 - α) · BM25
    ↓
DETERMINISTIC CONTRADICTION DETECTOR (Pre-LLM Numerical & Polarity Gate)
    ├── [Conflict Found] ──► Immediate Output: "CONTRADICTION DETECTED" (Early Halt)
    └── [No Conflict]   ──► Proceed to Evidence Gate
    ↓
EVIDENCE GATE (Prompt-Injection Neutralizer & Lexical Sufficiency Check)
    ├── [Insufficient]  ──► Immediate Output: "NOT ENOUGH EVIDENCE" (Early Halt)
    └── [Sufficient]    ──► Forward Sanitized Data Chunks
    ↓
EVIDENCE-BOUND LOCAL LLM (llama3.2:latest @ 100% Offline Intel CPU)
    ↓
CITATION-ID VALIDATOR (Verifies that all [chunk_id] citations exist)
    ↓
POST-GENERATION CLAIM SUPPORT VERIFICATION GATE
    ├── [All Claims SUPPORTED] ──────────────► FINAL EVIDENCE-BOUND RESPONSE
    └── [Any Claim UNSUPPORTED] ─────────────► CONTROLLED 1-STEP REGENERATION
                                                    ↓
                                               RE-VERIFICATION GATE
                                               ├── [SUPPORTED]   ──► FINAL RESPONSE
                                               └── [UNSUPPORTED] ──► "NOT ENOUGH EVIDENCE"
```

---

## ✨ Core Features

1. **100% Offline CPU Inference**:
   - Runs locally on Intel Core i5-12450H CPU (8 threads).
   - Zero cloud API or external network dependencies.
2. **Prompt Injection Immunity**:
   - Retrieved chunks are strictly treated as passive **DATA**, not instructions.
   - Adversarial commands embedded in source documents cannot hijack the answering or verification engine.
3. **Deterministic Pre-LLM Contradiction Gate**:
   - Catches numerical and polarity conflicts across evidence chunks before invoking the LLM, halting immediately with formatted conflicting statements.
4. **Post-Generation Claim Support Verification**:
   - Verifies that cited excerpts actually establish every generated factual claim.
   - Deterministic JSON verdict validation (`SUPPORTED` vs `UNSUPPORTED`) with single-retry error recovery.
5. **Controlled 1-Step Regeneration**:
   - If an unsupported claim is detected, triggers at most one controlled rewrite attempt.
   - If unsupported claims persist, safely fails closed with exact `NOT ENOUGH EVIDENCE`.
6. **PII Masking & Privacy Preservation**:
   - Automatically masks names, emails, phone numbers, and Aadhaar numbers into opaque tokens (`[PERSON_001]`, `[EMAIL_001]`, `[AADHAAR_001]`).
   - Reversible mappings are retained securely offline for authorized consumers.
7. **Qualcomm AI Hub Model Validation**:
   - Validated Qualcomm AI Hub (`qai-hub` v0.55.0) integration path for hosted Snapdragon hardware (`Samsung Galaxy S25` / Snapdragon 8 Elite `sm8750-ac` and `Snapdragon X Elite CRD`).

---

## 📁 Project Structure

```
.
├── proofmesh/                           # Core ProofMesh package
│   ├── __init__.py                      # Unified exports (Pipeline, Gate, HybridRetriever, SemanticRetriever)
│   ├── prompt_templates.py              # Evidence-Bound system prompts & input templates
│   ├── pii_masker.py                    # PII detection & pseudonymization engine
│   ├── contradiction_detector.py        # Deterministic numerical & polarity conflict engine
│   ├── evidence_gate.py                 # Sanitizer against prompt injection & evidence gate
│   ├── retriever.py                     # [UPGRADED] True HybridRetriever (BM25 + Dense Semantic)
│   ├── dense_retrieval/                 # [NEW] Integrated dense semantic retrieval subsystem
│   │   ├── __init__.py                  # Exports SemanticRetriever, SemanticEmbedder, SemanticChunk
│   │   ├── models.py                    # Data models (SemanticChunk, ChunkRecord, RetrievalResult)
│   │   ├── embeddings.py                # SemanticEmbedder (all-MiniLM-L6-v2 with deterministic mock fallback)
│   │   └── semantic_retriever.py        # In-memory vector matrix & vectorized cosine retrieval
│   ├── claim_verifier.py                # ClaimExtractor & ClaimSupportVerifier gate
│   ├── answering_engine.py              # Local LLM runner with citation validation & regeneration
│   └── pipeline.py                      # Unified end-to-end multi-tier pipeline
├── person1_dense_retrieval/             # Standalone Person 1 Workstream (Preserved)
│   ├── src/dense_retrieval/             # Standalone source package
│   ├── tests/                           # Standalone embedding & retrieval tests
│   ├── examples/                        # Standalone semantic demonstration
│   ├── benchmarks/                      # Standalone latency & throughput benchmark
│   ├── docs/architecture.md             # Subsystem architecture
│   ├── requirements.txt                 # Standalone requirements
│   └── README.md                        # Standalone Person 1 documentation
├── tests/                               # Unified test suite
│   ├── test_embeddings.py               # Embedding subsystem unit tests
│   ├── test_semantic_retriever.py       # Dense retriever unit tests
│   └── test_hybrid_retriever.py         # True HybridRetriever unit tests
├── offline_inference.py                 # Standalone local CPU inference benchmark harness
├── test_proofmesh.py                    # 12-scenario automated verification test suite
├── requirements.txt                     # Unified project dependencies
├── NOTES.md                             # Comprehensive environment, benchmark & validation log
└── README.md
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) with `llama3.2:latest` (or `llama3.2:1b`)

### Running the Test Suite
```bash
python test_proofmesh.py
```

### Python API Usage
```python
from proofmesh.pipeline import ProofMeshPipeline

# Initialize offline pipeline
pipeline = ProofMeshPipeline(model_name="llama3.2:latest")

# Index documents
pipeline.index_document(
    doc_name="warranty_policy.txt",
    content="The hardware warranty period is 24 months from the purchase date."
)

# Query with strict evidence bounds
result = pipeline.query("What is the hardware warranty period?")

print("Answer:", result["answer"])
# Output: "The hardware warranty period is 24 months. [chunk_1]"

print("Verification Metadata:", result["claim_verification"])
# Output: {'total_claims': 1, 'supported_claims': 1, 'unsupported_claims': 0, 'regenerated': False}
```

---

## 🧪 Verification Matrix (`test_proofmesh.py`)

| Test ID | Scenario | Result |
| :--- | :--- | :--- |
| **Test 1** | Grounded Q&A with Inline Citations | **PASSED** |
| **Test 2** | Deterministic Contradiction Detection (Numerical Conflict) | **PASSED** |
| **Test 3** | Deterministic Contradiction Detection (Polarity Flip) | **PASSED** |
| **Test 4** | Insufficient Evidence Exact Fallback (`NOT ENOUGH EVIDENCE`) | **PASSED** |
| **Test 5** | Adversarial Document Prompt Injection Defense | **PASSED** |
| **Test 6** | PII Pseudonymization & Placeholder Preservation | **PASSED** |
| **Test 7** | Claim Verifier: Correct Supported Claim | **PASSED** |
| **Test 8** | Claim Verifier: Unsupported Detail -> Controlled Regeneration | **PASSED** |
| **Test 9** | Claim Verifier: Completely Unsupported Answer -> Rejection | **PASSED** |
| **Test 10** | Claim Verifier: Invalid Citation ID (`[chunk_999]`) -> Rejection | **PASSED** |
| **Test 11** | Claim Verifier: Prompt Injection Inside Evidence Chunk | **PASSED** |
| **Test 12** | Claim Verifier: PII Placeholder Support | **PASSED** |

---

## 📄 License
MIT License.
