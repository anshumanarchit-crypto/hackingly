"""
ProofMesh Deterministic Contradiction Detector
-----------------------------------------------
First line of defense against hallucinations and conflicting evidence.
Analyzes retrieved chunks deterministically for numerical, polarity, temporal,
and status contradictions BEFORE passing data to the LLM.

Philosophy: "The model cannot decide what is true. Our evidence layer decides what the model is allowed to say."
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

@dataclass
class ContradictionResult:
    is_contradiction: bool
    conflicting_statements: List[Tuple[str, str]]  # List of (statement_text, chunk_id)
    details: str

    def format_output(self) -> str:
        """Formats the output according to the strict ProofMesh contract."""
        if not self.is_contradiction or not self.conflicting_statements:
            return ""
        
        lines = ["CONTRADICTION DETECTED"]
        for idx, (stmt, chunk_id) in enumerate(self.conflicting_statements, start=1):
            clean_stmt = stmt.strip()
            # Ensure chunk id formatting
            cid = chunk_id if chunk_id.startswith("[") else f"[{chunk_id}]"
            lines.append(f"Conflicting statement {idx}: {clean_stmt} {cid}")
        
        return "\n".join(lines)


class DeterministicContradictionDetector:
    """
    Scans candidate chunks for explicit, irreconcilable factual conflicts.
    """

    POLARITY_PAIRS = [
        (r'\b(approved|authorized|accepted|granted|permitted|allowed)\b',
         r'\b(rejected|denied|declined|prohibited|forbidden|disapproved|not approved|unauthorized)\b'),
        (r'\b(eligible|qualified|entitled)\b',
         r'\b(ineligible|disqualified|not eligible|unqualified)\b'),
        (r'\b(valid|active|effective|enabled)\b',
         r'\b(invalid|inactive|expired|terminated|disabled|void)\b'),
        (r'\b(mandatory|required|compulsory|obligatory)\b',
         r'\b(optional|voluntary|exempt|not required)\b'),
        (r'\b(passed|succeeded)\b',
         r'\b(failed|unsuccessful)\b'),
        (r'\b(included|covered)\b',
         r'\b(excluded|not included|not covered)\b')
    ]

    def __init__(self):
        pass

    def check_chunks(self, chunks: List[Dict[str, Any]], user_query: Optional[str] = None) -> ContradictionResult:
        """
        Compares all pairs of chunks for direct, explicit factual contradictions.
        """
        if len(chunks) < 2:
            return ContradictionResult(is_contradiction=False, conflicting_statements=[], details="Less than 2 chunks")

        # 1. Check for polarity / antonym contradictions on matching subjects
        polarity_conflict = self._check_polarity_conflicts(chunks)
        if polarity_conflict.is_contradiction:
            return polarity_conflict

        # 2. Check for numerical / quantitative conflicts on identical attributes
        numerical_conflict = self._check_numerical_conflicts(chunks)
        if numerical_conflict.is_contradiction:
            return numerical_conflict

        return ContradictionResult(is_contradiction=False, conflicting_statements=[], details="No deterministic contradiction found")

    def _extract_sentences(self, text: str) -> List[str]:
        """Splits chunk text into individual sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def _check_polarity_conflicts(self, chunks: List[Dict[str, Any]]) -> ContradictionResult:
        """Checks for direct negation / polarity flips across chunks."""
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                c1, c2 = chunks[i], chunks[j]
                c1_id = c1.get("chunk_id", f"chunk_{i}")
                c2_id = c2.get("chunk_id", f"chunk_{j}")
                
                sents1 = self._extract_sentences(c1.get("text", ""))
                sents2 = self._extract_sentences(c2.get("text", ""))

                for s1 in sents1:
                    for s2 in sents2:
                        for pos_pat, neg_pat in self.POLARITY_PAIRS:
                            pos_in_s1 = bool(re.search(pos_pat, s1, re.IGNORECASE))
                            neg_in_s2 = bool(re.search(neg_pat, s2, re.IGNORECASE))
                            
                            neg_in_s1 = bool(re.search(neg_pat, s1, re.IGNORECASE))
                            pos_in_s2 = bool(re.search(pos_pat, s2, re.IGNORECASE))

                            if (pos_in_s1 and neg_in_s2) or (neg_in_s1 and pos_in_s2):
                                # Verify they share significant keyword/entity overlap
                                words1 = set(re.findall(r'\b[a-zA-Z]{4,}\b', s1.lower()))
                                words2 = set(re.findall(r'\b[a-zA-Z]{4,}\b', s2.lower()))
                                shared = words1.intersection(words2)
                                
                                # Ignore generic words
                                ignore_words = {"this", "that", "with", "from", "have", "were", "been", "will", "shall", "must"}
                                shared = shared - ignore_words

                                if len(shared) >= 2:
                                    return ContradictionResult(
                                        is_contradiction=True,
                                        conflicting_statements=[(s1, c1_id), (s2, c2_id)],
                                        details=f"Polarity conflict detected between {c1_id} and {c2_id} on shared entities: {shared}"
                                    )

        return ContradictionResult(is_contradiction=False, conflicting_statements=[], details="")

    def _check_numerical_conflicts(self, chunks: List[Dict[str, Any]]) -> ContradictionResult:
        """Checks for conflicting numbers/dates/durations for identical attributes."""
        # Pattern for numeric facts: e.g. "warranty is 24 months", "fee is $500", "price is 100 USD"
        patterns = [
            r'(\b\w+\s*(?:period|duration|limit|warranty|price|cost|fee|term|amount|rate)\b[^\.\,\;]*?(\d+(?:\.\d+)?\s*(?:months|years|days|hours|usd|dollars|inr|rupees|percent|%)))',
            r'(\b(?:founded|established|released|born|started|commenced)\s*(?:in|on)?\s*(\b\d{4}\b))'
        ]

        facts: List[Tuple[str, str, str, str]] = []  # (attribute_key, full_sentence, value, chunk_id)

        for chunk in chunks:
            cid = chunk.get("chunk_id", "chunk_x")
            text = chunk.get("text", "")
            sentences = self._extract_sentences(text)

            for s in sentences:
                # Check for "warranty is X months", "period is X days", etc.
                match = re.search(r'\b(\w+)\s+(?:is|was|of|shall be)\s+(\d+(?:\.\d+)?\s*(?:months?|years?|days?|hours?|%|percent|\$|usd|inr|rupees))\b', s, re.IGNORECASE)
                if match:
                    attr = match.group(1).lower()
                    val = match.group(2).lower().strip()
                    facts.append((attr, s, val, cid))

        # Compare extracted numerical facts
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                attr1, s1, val1, cid1 = facts[i]
                attr2, s2, val2, cid2 = facts[j]

                if cid1 != cid2 and attr1 == attr2 and val1 != val2:
                    return ContradictionResult(
                        is_contradiction=True,
                        conflicting_statements=[(s1, cid1), (s2, cid2)],
                        details=f"Numerical conflict on '{attr1}': '{val1}' ({cid1}) vs '{val2}' ({cid2})"
                    )

        return ContradictionResult(is_contradiction=False, conflicting_statements=[], details="")


if __name__ == "__main__":
    print("Testing DeterministicContradictionDetector standalone...")
    detector = DeterministicContradictionDetector()
    chunks = [
        {"chunk_id": "chunk_1", "text": "The project warranty is 24 months."},
        {"chunk_id": "chunk_2", "text": "The project warranty is 12 months."}
    ]
    res = detector.check_chunks(chunks)
    print("Is contradiction:", res.is_contradiction)
    print("Details:", res.details)
    print("Formatted:\n" + res.format_output())
    print("Success: DeterministicContradictionDetector executed cleanly.")
