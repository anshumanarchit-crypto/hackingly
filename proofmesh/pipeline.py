"""
ProofMesh Unified End-to-End Pipeline
-------------------------------------
Orchestrates:
Retriever -> PII Masking -> Deterministic Contradiction Detector -> Evidence Gate -> LLM -> Citation-ID Validation -> Claim Support Verification -> Evidence-Bound Response.

"The model cannot decide what is true. Our evidence layer decides what the model is allowed to say."
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional

try:
    from .retriever import HybridRetriever
    from .pii_masker import PIIMasker
    from .evidence_gate import EvidenceGate, EvidenceGateDecision
    from .answering_engine import EvidenceBoundAnsweringEngine
except (ImportError, ValueError):
    _curr_dir = os.path.dirname(os.path.abspath(__file__))
    _root_dir = os.path.dirname(_curr_dir)
    for _p in [_curr_dir, _root_dir]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    try:
        from proofmesh.retriever import HybridRetriever
        from proofmesh.pii_masker import PIIMasker
        from proofmesh.evidence_gate import EvidenceGate, EvidenceGateDecision
        from proofmesh.answering_engine import EvidenceBoundAnsweringEngine
    except ImportError:
        from retriever import HybridRetriever  # type: ignore[import-not-found]
        from pii_masker import PIIMasker  # type: ignore[import-not-found]
        from evidence_gate import EvidenceGate, EvidenceGateDecision  # type: ignore[import-not-found]
        from answering_engine import EvidenceBoundAnsweringEngine  # type: ignore[import-not-found]

class ProofMeshPipeline:
    """
    End-to-End ProofMesh RAG Pipeline with Deterministic Evidence Gating
    and Post-Generation Claim Support Verification.
    """

    def __init__(self, model_name: str = "llama3.2:latest", top_k: int = 3):
        self.retriever = HybridRetriever()
        self.pii_masker = PIIMasker()
        self.evidence_gate = EvidenceGate()
        self.answering_engine = EvidenceBoundAnsweringEngine(model_name=model_name)
        self.top_k = top_k

    def index_document(self, doc_name: str, content: str, mask_pii: bool = True):
        """Indexes a document after optional PII pseudonymization."""
        processed_content = self.pii_masker.mask_text(content) if mask_pii else content
        return self.retriever.add_document(doc_name, processed_content)

    def query(
        self,
        user_query: str,
        custom_chunks: Optional[List[Dict[str, Any]]] = None,
        mask_query_pii: bool = True,
        verify_claims: bool = True,
        redact_active: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes query through the multi-tier ProofMesh defense pipeline.
        Privacy protection is enforced by default (redact_active=True).
        Local de-redaction is only performed when explicitly requested (redact_active=False).
        """
        t0 = time.time()
        
        # 1. Mask query PII if enabled
        safe_query = self.pii_masker.mask_text(user_query) if mask_query_pii else user_query

        # 2. Retrieve evidence chunks or use supplied custom chunks
        if custom_chunks is not None:
            raw_chunks = custom_chunks
        else:
            raw_chunks = self.retriever.retrieve(safe_query, top_k=self.top_k)

        # 3. Evidence Gate (Sanitization + Deterministic Contradiction Check)
        gate_decision: EvidenceGateDecision = self.evidence_gate.evaluate(raw_chunks, safe_query)

        # If gate halts execution (insufficient evidence or contradiction detected deterministically)
        if not gate_decision.allow_llm:
            t1 = time.time()
            halt_answer = gate_decision.immediate_response
            if not redact_active and halt_answer:
                halt_answer = self.pii_masker.unmask_text(halt_answer)
            contradiction_info = None
            if gate_decision.contradiction_result and gate_decision.contradiction_result.is_contradiction:
                contradiction_info = {
                    "is_contradiction": True,
                    "conflicting_statements": gate_decision.contradiction_result.conflicting_statements,
                    "details": gate_decision.contradiction_result.details,
                }
            return {
                "answer": halt_answer,
                "status": gate_decision.status,
                "gating_reason": gate_decision.reason,
                "gated_by": "Deterministic Evidence Gate",
                "retrieved_chunks": gate_decision.sanitized_chunks,
                "contradiction_details": contradiction_info,
                "citations": [],
                "is_citation_valid": True,
                "claim_verification": {
                    "total_claims": 0,
                    "supported_claims": 0,
                    "unsupported_claims": 0,
                    "regenerated": False
                },
                "tokens_per_second": None,
                "total_pipeline_latency_seconds": round(t1 - t0, 3),
                "offline": True,
                "llm_invoked": False,
                "redact_active": redact_active,
            }

        # 4. LLM Answering Engine (Evidence-Bound Prompt Execution + Claim Verification Gate)
        llm_result = self.answering_engine.generate_answer(
            chunks=gate_decision.sanitized_chunks,
            user_query=safe_query,
            verify_claims=verify_claims
        )
        t1 = time.time()

        final_answer = llm_result["response"]
        retrieved_chunks = gate_decision.sanitized_chunks
        if not redact_active:
            final_answer = self.pii_masker.unmask_text(final_answer)
            retrieved_chunks = [
                {**c, "text": self.pii_masker.unmask_text(c.get("text", ""))}
                for c in retrieved_chunks
            ]

        return {
            "answer": final_answer,
            "status": "ANSWER_GENERATED",
            "gating_reason": gate_decision.reason,
            "gated_by": "LLM with Evidence-Bound Verification & Claim Support Gate",
            "retrieved_chunks": retrieved_chunks,
            "contradiction_details": None,
            "citations": llm_result["citations"],
            "is_citation_valid": llm_result["is_citation_valid"],
            "claim_verification": llm_result.get("claim_verification", {}),
            "claim_verification_details": llm_result.get("claim_verification_details", []),
            "tokens_per_second": llm_result["tokens_per_second"],
            "llm_latency_seconds": llm_result["latency_seconds"],
            "verification_latency_seconds": llm_result.get("verification_latency_seconds", 0.0),
            "total_pipeline_latency_seconds": round(t1 - t0, 3),
            "offline": True,
            "llm_invoked": True,
            "redact_active": redact_active,
        }


if __name__ == "__main__":
    print("Testing ProofMeshPipeline standalone...")
    pipeline = ProofMeshPipeline()
    sample_doc = (
        "Project Hackingly utilizes local CPU execution on Intel Core i5-12450H. "
        "The standard system response latency is bounded under 500 milliseconds. "
        "Customer support can be contacted at support@proofmesh.ai."
    )
    pipeline.index_document("overview.txt", sample_doc)
    res = pipeline.query("What processor does Project Hackingly utilize?")
    print("Answer:", res["answer"])
    print("Status:", res["status"])
    print("Gated By:", res["gated_by"])
    print("Success: ProofMeshPipeline executed cleanly.")
