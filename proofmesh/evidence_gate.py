"""
ProofMesh Evidence Gate
-----------------------
Enforces the Evidence-First invariant:
1. Filters & sanitizes candidate chunks (neutralizes prompt injection payloads).
2. Runs the Deterministic Contradiction Detector.
3. Evaluates evidence sufficiency thresholds before LLM invocation.

If evidence is insufficient, halts early with 'NOT ENOUGH EVIDENCE'.
If contradiction is detected, halts early with 'CONTRADICTION DETECTED'.
"""

import os
import sys
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

try:
    from .contradiction_detector import DeterministicContradictionDetector, ContradictionResult
except (ImportError, ValueError):
    _curr_dir = os.path.dirname(os.path.abspath(__file__))
    _root_dir = os.path.dirname(_curr_dir)
    for _p in [_curr_dir, _root_dir]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    try:
        from proofmesh.contradiction_detector import DeterministicContradictionDetector, ContradictionResult
    except ImportError:
        from contradiction_detector import DeterministicContradictionDetector, ContradictionResult  # type: ignore[import-not-found]

@dataclass
class EvidenceGateDecision:
    allow_llm: bool
    status: str  # "PASS", "CONTRADICTION_DETECTED", "NOT_ENOUGH_EVIDENCE"
    immediate_response: Optional[str]
    sanitized_chunks: List[Dict[str, Any]]
    reason: str
    contradiction_result: Optional[ContradictionResult] = None


class EvidenceGate:
    """
    Gates LLM execution by strictly validating evidence sufficiency,
    consistency, and prompt injection safety.
    """

    ADVERSARIAL_INJECTION_PATTERNS = [
        r'(?i)\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b',
        r'(?i)\byou\s+are\s+now\s+(?:a|an)\b',
        r'(?i)\bsystem\s+prompt\s*:\s*',
        r'(?i)\bdisregard\s+(?:all\s+)?(?:rules|instructions)\b',
        r'(?i)\boverride\s+(?:the\s+)?(?:system|rules)\b'
    ]

    def __init__(self, min_keyword_overlap: int = 1):
        self.contradiction_detector = DeterministicContradictionDetector()
        self.min_keyword_overlap = min_keyword_overlap

    def sanitize_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Neutralizes potential prompt injection text while preserving factual payload.
        """
        text = chunk.get("text", "")
        # Neutralize injection triggers by prefixing them as raw passive data
        for pattern in self.ADVERSARIAL_INJECTION_PATTERNS:
            text = re.sub(pattern, r'[QUOTED_TEXT_DATA: \g<0>]', text)
        
        sanitized = dict(chunk)
        sanitized["text"] = text
        return sanitized

    def evaluate(self, chunks: List[Dict[str, Any]], user_query: str) -> EvidenceGateDecision:
        """
        Evaluates chunks against query. Returns decision.
        """
        if not chunks:
            return EvidenceGateDecision(
                allow_llm=False,
                status="NOT_ENOUGH_EVIDENCE",
                immediate_response="NOT ENOUGH EVIDENCE",
                sanitized_chunks=[],
                reason="Zero evidence chunks retrieved."
            )

        # 1. Sanitize all chunks
        sanitized_chunks = [self.sanitize_chunk(c) for c in chunks]

        # 2. Check for deterministic contradiction across evidence
        contradiction_res: ContradictionResult = self.contradiction_detector.check_chunks(sanitized_chunks, user_query)
        if contradiction_res.is_contradiction:
            formatted_contradiction = contradiction_res.format_output()
            return EvidenceGateDecision(
                allow_llm=False,
                status="CONTRADICTION_DETECTED",
                immediate_response=formatted_contradiction,
                sanitized_chunks=sanitized_chunks,
                reason=f"Deterministic contradiction detected: {contradiction_res.details}",
                contradiction_result=contradiction_res,
            )

        # 3. Check for query-evidence lexical / keyword relevance overlap
        query_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', user_query.lower()))
        # Remove common stop words
        stop_words = {
            "what", "when", "where", "which", "who", "whom", "this", "that", "these",
            "those", "have", "with", "from", "does", "about", "the", "and", "for",
            "are", "was", "were", "can", "how", "has", "had", "its", "not", "but"
        }
        query_words = query_words - stop_words

        if query_words:
            all_evidence_text = " ".join(c.get("text", "") for c in sanitized_chunks).lower()
            overlap = [w for w in query_words if w in all_evidence_text]
            if len(overlap) < self.min_keyword_overlap:
                return EvidenceGateDecision(
                    allow_llm=False,
                    status="NOT_ENOUGH_EVIDENCE",
                    immediate_response="NOT ENOUGH EVIDENCE",
                    sanitized_chunks=sanitized_chunks,
                    reason=f"Insufficient lexical overlap between query {query_words} and evidence."
                )

        # Passed all gate checks -> forward to LLM
        return EvidenceGateDecision(
            allow_llm=True,
            status="PASS",
            immediate_response=None,
            sanitized_chunks=sanitized_chunks,
            reason="Passed contradiction and relevance checks."
        )


if __name__ == "__main__":
    print("Testing EvidenceGate standalone...")
    gate = EvidenceGate()
    chunks = [
        {"chunk_id": "chunk_1", "text": "The hardware warranty period is 24 months."}
    ]
    query = "What is the warranty period?"
    decision = gate.evaluate(chunks, query)
    print("Decision status:", decision.status)
    print("Allow LLM:", decision.allow_llm)
    print("Success: EvidenceGate executed cleanly.")
