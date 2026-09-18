"""
ProofMesh: Offline Evidence-Bound RAG Pipeline
----------------------------------------------
"The model cannot decide what is true. Our evidence layer decides what the model is allowed to say."
"""

from .prompt_templates import (
    EVIDENCE_BOUND_SYSTEM_PROMPT,
    CLAIM_VERIFIER_SYSTEM_PROMPT,
    format_prompt,
    format_user_prompt,
    format_verifier_input,
    REGENERATION_PROMPT_TEMPLATE,
    format_regeneration_user_prompt
)
from .pii_masker import (
    PIIMasker,
    mask_pii,
    unmask_pii,
    get_vault,
    redact_pii,
    deredact_pii,
    extract_entities,
    process_payload,
)
from .pii.vault import PIIVault
from .pii.redactor import redact_text, deredact_text
from .contradiction_detector import DeterministicContradictionDetector, ContradictionResult
from .evidence_gate import EvidenceGate, EvidenceGateDecision
from .retriever import HybridRetriever, DocumentChunk
from .dense_retrieval import (
    SemanticRetriever,
    SemanticChunk,
    ChunkRecord,
    RetrievalResult,
    SemanticEmbedder,
)
from .claim_verifier import ClaimSupportVerifier, ClaimExtractor, ClaimVerificationReport, ExtractedClaim
from .answering_engine import EvidenceBoundAnsweringEngine
from .pipeline import ProofMeshPipeline

__all__ = [
    "EVIDENCE_BOUND_SYSTEM_PROMPT",
    "CLAIM_VERIFIER_SYSTEM_PROMPT",
    "format_prompt",
    "format_user_prompt",
    "format_verifier_input",
    "REGENERATION_PROMPT_TEMPLATE",
    "format_regeneration_user_prompt",
    "PIIMasker",
    "mask_pii",
    "unmask_pii",
    "get_vault",
    "redact_pii",
    "deredact_pii",
    "extract_entities",
    "process_payload",
    "PIIVault",
    "redact_text",
    "deredact_text",
    "DeterministicContradictionDetector",
    "ContradictionResult",
    "EvidenceGate",
    "EvidenceGateDecision",
    "HybridRetriever",
    "DocumentChunk",
    "SemanticRetriever",
    "SemanticChunk",
    "ChunkRecord",
    "RetrievalResult",
    "SemanticEmbedder",
    "ClaimSupportVerifier",
    "ClaimExtractor",
    "ClaimVerificationReport",
    "ExtractedClaim",
    "EvidenceBoundAnsweringEngine",
    "ProofMeshPipeline",
]
