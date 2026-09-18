"""
ProofMesh Post-Generation Claim Support Verifier
------------------------------------------------
Verifies that every factual claim in a generated answer is directly and explicitly
supported by the cited source excerpts.

Rules:
- Never trust the verifier LLM blindly: validate JSON and verdict deterministically.
- Accepts ONLY verdict == "SUPPORTED" or verdict == "UNSUPPORTED".
- Single retry on malformed JSON; defaults to UNSUPPORTED on failure.
- Excerpts treated as passive DATA (prompt injection immune).
- PII placeholders preserved.
"""

import os
import sys
import re
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
try:
    import ollama
except ImportError:
    ollama = None

# Dual import support for both package execution and standalone execution
try:
    from .prompt_templates import CLAIM_VERIFIER_SYSTEM_PROMPT, format_verifier_input
except (ImportError, ValueError):
    _curr_dir = os.path.dirname(os.path.abspath(__file__))
    _root_dir = os.path.dirname(_curr_dir)
    for _p in [_curr_dir, _root_dir]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    try:
        from proofmesh.prompt_templates import CLAIM_VERIFIER_SYSTEM_PROMPT, format_verifier_input
    except ImportError:
        from prompt_templates import CLAIM_VERIFIER_SYSTEM_PROMPT, format_verifier_input  # type: ignore

@dataclass
class ExtractedClaim:
    claim_text: str
    citations: List[str]
    has_invalid_citations: bool = False
    invalid_citations: Optional[List[str]] = None


@dataclass
class VerificationItemResult:
    claim_text: str
    citations: List[str]
    verdict: str  # "SUPPORTED" or "UNSUPPORTED"
    reason: str
    latency_seconds: float = 0.0


@dataclass
class ClaimVerificationReport:
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    all_supported: bool
    results: List[VerificationItemResult]
    total_verification_latency: float
    regenerated: bool = False

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "regenerated": self.regenerated
        }


class ClaimExtractor:
    """
    Extracts factual claims and associated inline citations from generated text.
    Handles compound sentences and validates citation IDs against available excerpts.
    """

    def __init__(self):
        pass

    def extract_claims(self, text: str, supplied_chunks: List[Dict[str, Any]]) -> List[ExtractedClaim]:
        clean_text = text.strip()
        if not clean_text:
            return []

        # Check for exact terminal sentinel outputs
        if clean_text in ["NOT ENOUGH EVIDENCE", "CONTRADICTION DETECTED"] or clean_text.startswith("CONTRADICTION DETECTED"):
            return []

        valid_chunk_ids = {c.get("chunk_id") for c in supplied_chunks if "chunk_id" in c}

        # Sentence extraction that keeps citations attached to the sentence
        # and avoids splitting on PII brackets like [PERSON_001]
        raw_sentences = [
            s.strip() for s in re.findall(r'[^.!?]+(?:[.!?]+(?:\s*\[(?:chunk_\w+|\d+)\])*|$)', clean_text)
            if s.strip()
        ]
        
        claims: List[ExtractedClaim] = []

        for sent in raw_sentences:
            sent_str = sent.strip()
            if not sent_str:
                continue

            # Extract all citations [chunk_xxx] in this sentence
            raw_cits = re.findall(r'\[(chunk_\w+|\d+)\]', sent_str)
            normalized_citations = []
            for c in raw_cits:
                if c in valid_chunk_ids:
                    normalized_citations.append(c)
                elif f"chunk_{c}" in valid_chunk_ids:
                    normalized_citations.append(f"chunk_{c}")
                else:
                    normalized_citations.append(c)

            invalid_cids = [c for c in normalized_citations if c not in valid_chunk_ids]

            # Strip citation tags from claim text for clean factual inspection
            claim_body = re.sub(r'\[(chunk_\w+|\d+)\]', '', sent_str).strip()
            claim_body = re.sub(r'\s+', ' ', claim_body).strip()
            
            # Check for compound conjunctions (e.g., "claim A and covers B")
            sub_clauses = self._split_compound_claims(claim_body)

            for clause in sub_clauses:
                clean_clause = clause.strip().rstrip('.!,;')
                if len(clean_clause) > 3:
                    claims.append(ExtractedClaim(
                        claim_text=clean_clause,
                        citations=normalized_citations,
                        has_invalid_citations=bool(invalid_cids) or (not normalized_citations),
                        invalid_citations=invalid_cids if invalid_cids else ([] if normalized_citations else ["No citations"])
                    ))

        return claims

    def _split_compound_claims(self, sentence: str) -> List[str]:
        """
        Splits compound assertions joined by conjunctions where separate factual claims exist.
        e.g. 'The warranty is 24 months and covers accidental damage'
        """
        # Match ' and ' followed by active verb phrases
        compound_pattern = r'\s+and\s+(?=(?:covers?|includes?|requires?|costs?|provides?|features?)\b)'
        parts = re.split(compound_pattern, sentence, flags=re.IGNORECASE)
        if len(parts) > 1 and all(len(p.strip()) > 5 for p in parts):
            return [p.strip() for p in parts if p.strip()]
        return [sentence]


class ClaimSupportVerifier:
    """
    Validates claim support using local LLM with deterministic JSON parsing.
    """

    def __init__(self, model_name: str = "llama3.2:latest", host: str = "http://127.0.0.1:11434"):
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=self.host) if ollama is not None else None
        self.extractor = ClaimExtractor()

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

    def verify_response(
        self,
        generated_text: str,
        supplied_chunks: List[Dict[str, Any]]
    ) -> ClaimVerificationReport:
        """
        Extracts claims from generated response and verifies each claim against cited chunks.
        """
        t_start = time.time()
        claims = self.extractor.extract_claims(generated_text, supplied_chunks)

        if not claims:
            t_end = time.time()
            return ClaimVerificationReport(
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                all_supported=True,
                results=[],
                total_verification_latency=round(t_end - t_start, 3),
                regenerated=False
            )

        # Create chunk lookup dictionary
        chunk_map = {c.get("chunk_id"): c.get("text", "") for c in supplied_chunks}
        results: List[VerificationItemResult] = []

        for claim in claims:
            # 1. Deterministic check: Non-existent / invalid citations are automatically UNSUPPORTED
            if claim.has_invalid_citations or not claim.citations:
                results.append(VerificationItemResult(
                    claim_text=claim.claim_text,
                    citations=claim.citations,
                    verdict="UNSUPPORTED",
                    reason=f"Invalid or missing citation IDs: {claim.invalid_citations}",
                    latency_seconds=0.0
                ))
                continue

            # Gather cited excerpt text
            cited_texts = []
            for cid in claim.citations:
                if cid in chunk_map:
                    cited_texts.append(f"[{cid}]\n{chunk_map[cid]}")

            if not cited_texts:
                results.append(VerificationItemResult(
                    claim_text=claim.claim_text,
                    citations=claim.citations,
                    verdict="UNSUPPORTED",
                    reason="No valid cited excerpt text found.",
                    latency_seconds=0.0
                ))
                continue

            cited_excerpts_combined = "\n\n".join(cited_texts)

            # 2. Run LLM verifier with deterministic validation
            verdict, lat, reason = self._verify_single_claim(claim.claim_text, cited_excerpts_combined)
            results.append(VerificationItemResult(
                claim_text=claim.claim_text,
                citations=claim.citations,
                verdict=verdict,
                reason=reason,
                latency_seconds=lat
            ))

        supported_count = sum(1 for r in results if r.verdict == "SUPPORTED")
        unsupported_count = sum(1 for r in results if r.verdict == "UNSUPPORTED")
        t_end = time.time()

        return ClaimVerificationReport(
            total_claims=len(results),
            supported_claims=supported_count,
            unsupported_claims=unsupported_count,
            all_supported=(unsupported_count == 0),
            results=results,
            total_verification_latency=round(t_end - t_start, 3),
            regenerated=False
        )

    def _verify_single_claim(self, claim_text: str, cited_excerpts: str) -> Tuple[str, float, str]:
        """
        Executes the verifier prompt and deterministically validates the JSON verdict.
        Retries once on malformed response before failing closed to UNSUPPORTED.
        """
        user_input = format_verifier_input(claim=claim_text, cited_excerpts=cited_excerpts)
        messages = [
            {"role": "system", "content": CLAIM_VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]

        if self.client is None or not self.is_available():
            # Deterministic offline claim verification fallback
            claim_tokens = set(re.findall(r'\[[A-Z0-9_]+\]|\b\w+\b', claim_text.lower()))
            excerpt_tokens = set(re.findall(r'\[[A-Z0-9_]+\]|\b\w+\b', cited_excerpts.lower()))
            stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "out", "off", "over", "under", "again", "further", "then", "once", "and", "or", "but"}
            content_tokens = claim_tokens - stopwords
            if content_tokens and content_tokens.issubset(excerpt_tokens):
                return "SUPPORTED", 0.001, "Deterministic offline claim support verification."
            return "UNSUPPORTED", 0.001, "Claim contains facts or terms not established in cited excerpts."

        t1: float = 0.0  # pre-initialized so it's always bound after the loop
        for attempt in range(2):
            t0 = time.time()
            t1 = t0  # reset at start of each attempt
            try:
                raw_response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    options={"temperature": 0.0}
                )
                t1 = time.time()
                content = raw_response["message"]["content"].strip()
                
                # Deterministic JSON extraction
                verdict = self._parse_verdict(content)
                if verdict in ["SUPPORTED", "UNSUPPORTED"]:
                    return verdict, round(t1 - t0, 3), f"Verified in attempt {attempt+1}"

            except Exception as e:
                t1 = time.time()
                if attempt == 1:
                    return "UNSUPPORTED", round(t1 - t0, 3), f"Verifier exception: {str(e)}"

        return "UNSUPPORTED", round(t1 - t0, 3), "Malformed verifier JSON; failed closed to UNSUPPORTED."

    def _parse_verdict(self, text: str) -> Optional[str]:
        """
        Strictly parses JSON or exact verdict tokens.
        Accepts ONLY 'SUPPORTED' or 'UNSUPPORTED'.
        """
        # Try direct JSON parsing
        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)
            
            json_match = re.search(r'\{[^{}]*\}', cleaned)
            if json_match:
                parsed = json.loads(json_match.group(0))
                verdict = parsed.get("verdict", "").strip().upper()
                if verdict in ["SUPPORTED", "UNSUPPORTED"]:
                    return verdict
        except Exception:
            pass

        # Strict token fallback only if exact JSON structure failed
        if re.search(r'\b"verdict"\s*:\s*"SUPPORTED"\b', text, re.IGNORECASE):
            return "SUPPORTED"
        if re.search(r'\b"verdict"\s*:\s*"UNSUPPORTED"\b', text, re.IGNORECASE):
            return "UNSUPPORTED"

        return None


if __name__ == "__main__":
    print("Testing ClaimSupportVerifier standalone...")
    verifier = ClaimSupportVerifier()
    chunks = [
        {"chunk_id": "chunk_1", "text": "The hardware warranty period is 24 months from the date of purchase."}
    ]
    test_claim = "The hardware warranty period is 24 months. [chunk_1]"
    rep = verifier.verify_response(test_claim, chunks)
    print(f"Total claims: {rep.total_claims}, Supported: {rep.supported_claims}, All Supported: {rep.all_supported}")
    for res in rep.results:
        print(f"  Claim: '{res.claim_text}' -> Verdict: {res.verdict} ({res.reason})")
    print("Success: ClaimSupportVerifier executed cleanly.")
