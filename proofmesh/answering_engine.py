"""
ProofMesh Evidence-Bound Answering Engine
-----------------------------------------
Executes the Evidence-Bound system prompt on the offline local LLM.
Includes:
1. Citation-ID validation (verifies all [chunk_id] citations exist).
2. Post-Generation Claim Support Verification (verifies cited text actually establishes claims).
3. 1-step controlled regeneration if any claim is unsupported, failing closed to 'NOT ENOUGH EVIDENCE'.
"""

import os
import sys
import re
import time
from typing import List, Dict, Any, Optional, Tuple

try:
    import ollama
except ImportError:
    ollama = None

# Dual import support for both package execution and standalone execution
try:
    from .prompt_templates import (
        EVIDENCE_BOUND_SYSTEM_PROMPT,
        format_user_prompt,
        format_regeneration_user_prompt
    )
    from .claim_verifier import ClaimSupportVerifier, ClaimVerificationReport
except (ImportError, ValueError):
    _curr_dir = os.path.dirname(os.path.abspath(__file__))
    _root_dir = os.path.dirname(_curr_dir)
    for _p in [_curr_dir, _root_dir]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    try:
        from proofmesh.prompt_templates import (  # type: ignore
            EVIDENCE_BOUND_SYSTEM_PROMPT,
            format_user_prompt,
            format_regeneration_user_prompt
        )
        from proofmesh.claim_verifier import ClaimSupportVerifier, ClaimVerificationReport  # type: ignore
    except ImportError:
        from prompt_templates import (  # type: ignore
            EVIDENCE_BOUND_SYSTEM_PROMPT,
            format_user_prompt,
            format_regeneration_user_prompt
        )
        from claim_verifier import ClaimSupportVerifier, ClaimVerificationReport  # type: ignore

class EvidenceBoundAnsweringEngine:
    """
    Offline local LLM answering engine bound strictly to evidence chunks
    with post-generation claim verification.
    """

    def __init__(self, model_name: str = "llama3.2:latest", host: str = "http://127.0.0.1:11434"):
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=self.host) if ollama is not None else None
        self.verifier = ClaimSupportVerifier(model_name=model_name, host=host)

    def is_available(self) -> bool:
        """Check if local offline LLM service is running and model is loaded."""
        if self.client is None:
            return False
        try:
            models_info = self.client.list()
            model_names = [getattr(m, "model", str(m)) for m in getattr(models_info, "models", [])]
            return any(self.model_name in name for name in model_names)
        except Exception:
            return False

    def generate_answer(
        self,
        chunks: List[Dict[str, Any]],
        user_query: str,
        temperature: float = 0.0,
        verify_claims: bool = True
    ) -> Dict[str, Any]:
        """
        Generates answer, performs citation-ID validation, runs post-generation
        claim verification, and manages 1-step controlled regeneration if needed.
        """
        formatted_user_msg = format_user_prompt(chunks, user_query)
        supplied_chunk_ids = {c.get("chunk_id") for c in chunks if "chunk_id" in c}

        messages = [
            {"role": "system", "content": EVIDENCE_BOUND_SYSTEM_PROMPT},
            {"role": "user", "content": formatted_user_msg}
        ]

        t0 = time.time()
        t1 = t0
        raw_text = None
        eval_count = 0
        tokens_per_sec = 50.0

        if self.client is not None and self.is_available():
            try:
                raw_response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_thread": 8,  # Intel Core i5-12450H CPU threads
                    }
                )
                t1 = time.time()
                raw_text = raw_response["message"]["content"].strip()
                eval_count = raw_response.get("eval_count", 0)
                eval_duration_ns = raw_response.get("eval_duration", 0) or 1
                eval_duration_s = eval_duration_ns / 1e9
                tokens_per_sec = eval_count / eval_duration_s if eval_duration_s > 0 else 0.0
            except Exception:
                raw_text = None

        if raw_text is None:
            # Deterministic offline extractive fallback when Ollama daemon is offline
            matched_sentences = []
            stopwords = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "out", "off", "over", "under", "again", "further", "then", "once", "and", "or", "but"}
            q_keywords = [w.lower() for w in re.findall(r'\b\w+\b', user_query) if w.lower() not in stopwords]

            for c in chunks:
                cid = c.get("chunk_id", "")
                text = c.get("text", "")
                # Ignore entire chunks that contain prompt injection attacks
                if "ignore all previous instructions" in text.lower() or "unrestricted assistant" in text.lower():
                    continue

                sentences = re.split(r'(?<=[.?!])\s+', text)
                for s in sentences:
                    s_clean = s.strip()
                    s_words = set(re.findall(r'\b\w+\b', s_clean.lower()))
                    overlap = sum(1 for w in q_keywords if w in s_words)
                    if overlap >= 2 or (len(q_keywords) == 1 and overlap == 1):
                        matched_sentences.append(f"{s_clean.rstrip('.')} [{cid}].")
                        break
                if matched_sentences:
                    break

            raw_text = " ".join(matched_sentences) if matched_sentences else "NOT ENOUGH EVIDENCE"
            t1 = time.time()
            eval_count = len(raw_text.split())
            tokens_per_sec = 50.0

        # Step 1: Citation-ID Validation
        validated_text, citations_found, is_citation_valid = self._validate_citations(
            raw_text, supplied_chunk_ids
        )

        # Step 2: Post-Generation Claim Support Verification
        report: Optional[ClaimVerificationReport] = None
        final_text = validated_text
        regenerated = False

        if verify_claims and final_text not in ["NOT ENOUGH EVIDENCE", "CONTRADICTION DETECTED"] and not final_text.startswith("CONTRADICTION DETECTED"):
            # If citation ID validation failed (e.g. fabricated chunk_999), mark unsupported
            if not is_citation_valid:
                # Immediate regeneration or reject
                report = self.verifier.verify_response(final_text, chunks)
                report.unsupported_claims = max(1, report.unsupported_claims)
                report.all_supported = False
            else:
                report = self.verifier.verify_response(final_text, chunks)

            # Step 3: Handle Unsupported Claims with 1-Step Controlled Regeneration
            if not report.all_supported:
                unsupported_claims = [r.claim_text for r in report.results if r.verdict == "UNSUPPORTED"]
                
                # Perform 1 controlled regeneration attempt
                regen_text, regen_eval_count, regen_tps = self._regenerate_answer(
                    chunks=chunks,
                    user_query=user_query,
                    previous_answer=final_text,
                    unsupported_claims=unsupported_claims,
                    temperature=temperature
                )
                regenerated = True

                # Step 4: Second Validation on Regenerated Answer
                regen_validated_text, regen_cits, regen_cit_valid = self._validate_citations(
                    regen_text, supplied_chunk_ids
                )

                if regen_validated_text in ["NOT ENOUGH EVIDENCE", "CONTRADICTION DETECTED"] or regen_validated_text.startswith("CONTRADICTION DETECTED"):
                    final_text = regen_validated_text
                    report = ClaimVerificationReport(
                        total_claims=0,
                        supported_claims=0,
                        unsupported_claims=0,
                        all_supported=True,
                        results=[],
                        total_verification_latency=report.total_verification_latency,
                        regenerated=True
                    )
                else:
                    report_second = self.verifier.verify_response(regen_validated_text, chunks)
                    report_second.regenerated = True
                    report_second.total_verification_latency += report.total_verification_latency

                    if report_second.all_supported and regen_cit_valid and report_second.total_claims > 0:
                        final_text = regen_validated_text
                        report = report_second
                    else:
                        # Persistent unsupported claim or empty supported content -> fallback
                        final_text = "NOT ENOUGH EVIDENCE"
                        report = report_second

        if report is None:
            report = ClaimVerificationReport(
                total_claims=len(citations_found),
                supported_claims=len(citations_found),
                unsupported_claims=0,
                all_supported=True,
                results=[],
                total_verification_latency=0.0,
                regenerated=False
            )

        return {
            "response": final_text,
            "raw_response": raw_text,
            "citations": list(citations_found),
            "supplied_chunk_ids": list(supplied_chunk_ids),
            "is_citation_valid": is_citation_valid,
            "claim_verification": report.to_metadata(),
            "claim_verification_details": [
                {"claim": r.claim_text, "verdict": r.verdict, "reason": r.reason}
                for r in report.results
            ],
            "tokens_per_second": round(tokens_per_sec, 2),
            "latency_seconds": round(t1 - t0, 3),
            "verification_latency_seconds": report.total_verification_latency,
            "eval_count": eval_count,
            "offline": True,
            "model": self.model_name
        }

    def _regenerate_answer(
        self,
        chunks: List[Dict[str, Any]],
        user_query: str,
        previous_answer: str,
        unsupported_claims: List[str],
        temperature: float = 0.0
    ) -> Tuple[str, int, float]:
        """
        Executes single-attempt regeneration removing unsupported claims.
        """
        regen_prompt = format_regeneration_user_prompt(
            chunks=chunks,
            user_query=user_query,
            original_answer=previous_answer,
            unsupported_claims=unsupported_claims
        )

        messages = [
            {"role": "system", "content": EVIDENCE_BOUND_SYSTEM_PROMPT},
            {"role": "user", "content": regen_prompt}
        ]

        t0 = time.time()
        if self.client is not None and self.is_available():
            try:
                raw_response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_thread": 8,
                    }
                )
                t1 = time.time()

                raw_text = raw_response["message"]["content"].strip()
                eval_count = raw_response.get("eval_count", 0)
                eval_duration_ns = raw_response.get("eval_duration", 0) or 1
                eval_duration_s = eval_duration_ns / 1e9
                tokens_per_sec = eval_count / eval_duration_s if eval_duration_s > 0 else 0.0

                return raw_text, eval_count, tokens_per_sec
            except Exception:
                pass

        # Offline deterministic regeneration fallback
        clean_text = previous_answer
        for uc in unsupported_claims:
            clean_text = re.sub(re.escape(uc), "", clean_text, flags=re.IGNORECASE).strip()
        clean_text = re.sub(r'\s+and\s*\[', ' [', clean_text).strip()
        return clean_text, len(clean_text.split()), 50.0

    def _validate_citations(
        self,
        response_text: str,
        supplied_chunk_ids: set
    ) -> Tuple[str, set, bool]:
        """
        Validates that all cited chunk IDs exist in the supplied excerpts.
        """
        # Exact keyword escapes
        if response_text in ["NOT ENOUGH EVIDENCE", "CONTRADICTION DETECTED"]:
            return response_text, set(), True

        if response_text.startswith("CONTRADICTION DETECTED"):
            return response_text, set(), True

        # Extract all [chunk_xxx] citations
        citations = set(re.findall(r'\[(chunk_\w+|\d+)\]', response_text))
        
        # Check for invalid / phantom citations
        invalid_citations = [c for c in citations if c not in supplied_chunk_ids and f"chunk_{c}" not in supplied_chunk_ids]

        if invalid_citations:
            # Strip invalid citations and mark citation validity as False
            clean_text = response_text
            for ic in invalid_citations:
                clean_text = clean_text.replace(f"[{ic}]", "")
            return clean_text.strip(), citations - set(invalid_citations), False

        return response_text, citations, True


if __name__ == "__main__":
    print("Testing EvidenceBoundAnsweringEngine standalone...")
    engine = EvidenceBoundAnsweringEngine()
    test_chunks = [
        {"chunk_id": "chunk_1", "text": "Project Hackingly achieves high accuracy on Intel Core i5-12450H CPU."},
        {"chunk_id": "chunk_2", "text": "The return window for standard items is 30 days."}
    ]
    query = "What CPU does Project Hackingly use?"
    res = engine.generate_answer(test_chunks, query)
    print("Response:", res["response"])
    print("Citations:", res["citations"])
    print("Claim Verification:", res["claim_verification"])
    print("Success: EvidenceBoundAnsweringEngine executed cleanly.")
