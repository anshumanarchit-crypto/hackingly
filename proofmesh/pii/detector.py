"""
Dynamic PII Detection Module combining Regex patterns and spaCy NER with fallback heuristics.
"""

import re
import logging
from typing import Any, Dict, List, Optional
from .schemas import PIISpan

logger = logging.getLogger("proofmesh.pii.detector")

_NLP = None
_SPACY_AVAILABLE = True


def get_spacy_nlp():
    """
    Attempts to load the spaCy en_core_web_sm model.
    If spaCy fails to load (e.g. environment incompatibility), flags fallback mode.
    """
    global _NLP, _SPACY_AVAILABLE
    if _NLP is None and _SPACY_AVAILABLE:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"spaCy load failed ({str(e)}). Falling back to regex NER heuristics.")
            _SPACY_AVAILABLE = False
            _NLP = None
    return _NLP if _SPACY_AVAILABLE else None


# 1. EMAIL Pattern
EMAIL_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
)

# 2. AADHAAR Pattern (12 digits, often formatted as 4-4-4 or 12 continuous digits)
AADHAAR_PATTERN = re.compile(
    r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b|\b\d{12}\b"
)

# 3. ID Patterns (SSN, 9-digit / Alphanumeric IDs with digits inside token)
SSN_PATTERN = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
ALPHA_ID_PATTERN = re.compile(r"\b(?=[A-Za-z0-9]{9}\b)[A-Za-z0-9]*\d[A-Za-z0-9]*\b")
GENERIC_9DIGIT_PATTERN = re.compile(r"\b\d{9}\b")

# 4. PHONE Pattern (International / US / Indian standard phone formats)
PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{3,5}"
)

# 5. PERSON & ORG Patterns
PERSON_TITLE_PATTERN = re.compile(
    r"\b(?:[Pp]atient|Dr\.|Dr|Mr\.|Mr|Mrs\.|Mrs|Ms\.|Ms|Prof\.|Prof|[Dd]octor|[Rr]esearcher|Officer|Agent|Engineer|Director|Manager|Lead|Author|Client)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)
FALLBACK_PERSON_NAME_PATTERN = re.compile(
    r"\b[A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,20}\b"
)
FALLBACK_ORG_PATTERN = re.compile(
    r"\b[A-Z][a-zA-Z0-9&.\s]{1,30}\s+(?:Hospital|Clinic|Healthcare|Bank|University|Corp|Inc|LLC|Foundation|Labs|Center|Services|Reserve)\b"
)

NON_PERSON_WORDS = {
    "North America", "South America", "New York", "United States", "Jan", "Feb", "Mar",
    "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Monday", "Tuesday",
    "Federal Reserve", "Reserve Bank"
}


def _spans_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    """Returns True if [s1, e1) intersects with [s2, e2)."""
    return max(s1, s2) < min(e1, e2)


def extract_entities(text: str) -> List[PIISpan]:
    """
    Extracts PII entities dynamically from input text.
    Categories: EMAIL, AADHAAR, ID, PHONE, PERSON, ORG.

    Order of execution & Overlap Resolution:
    1. Regex detectors run first (EMAIL, AADHAAR, ID, PHONE, PERSON titles).
    2. spaCy NER (or Fallback Heuristics) runs second for PERSON & ORG.
    3. Any match that intersects with an established match is discarded.
    """
    if not text:
        return []

    regex_spans: List[Dict[str, Any]] = []

    # 1. EMAIL
    for match in EMAIL_PATTERN.finditer(text):
        regex_spans.append(
            {
                "raw_value": match.group(0),
                "entity_type": "EMAIL",
                "start": match.start(),
                "end": match.end(),
            }
        )

    # 2. AADHAAR (Distinct entity category to preserve [AADHAAR_NNN] tokens)
    for match in AADHAAR_PATTERN.finditer(text):
        raw = match.group(0)
        digits_only = re.sub(r"\D", "", raw)
        if len(digits_only) == 12:
            start, end = match.start(), match.end()
            if not any(_spans_overlap(start, end, e["start"], e["end"]) for e in regex_spans):
                regex_spans.append(
                    {
                        "raw_value": raw,
                        "entity_type": "AADHAAR",
                        "start": start,
                        "end": end,
                    }
                )

    # 3. ID (SSN, 9-digit, alphanumeric IDs)
    id_matches = []
    for pattern in (SSN_PATTERN, ALPHA_ID_PATTERN, GENERIC_9DIGIT_PATTERN):
        for match in pattern.finditer(text):
            id_matches.append((match.start(), match.end(), match.group(0)))

    for start, end, raw in id_matches:
        if not any(_spans_overlap(start, end, e["start"], e["end"]) for e in regex_spans):
            regex_spans.append(
                {
                    "raw_value": raw,
                    "entity_type": "ID",
                    "start": start,
                    "end": end,
                }
            )

    # 4. PHONE
    for match in PHONE_PATTERN.finditer(text):
        start, end = match.start(), match.end()
        raw = match.group(0)
        digits_only = re.sub(r"\D", "", raw)
        if len(digits_only) < 7:
            continue
        if not any(_spans_overlap(start, end, e["start"], e["end"]) for e in regex_spans):
            regex_spans.append(
                {
                    "raw_value": raw,
                    "entity_type": "PHONE",
                    "start": start,
                    "end": end,
                }
            )

    # 5. PERSON Title Pattern
    for match in PERSON_TITLE_PATTERN.finditer(text):
        name_raw = match.group(1)
        name_start = match.start(1)
        name_end = match.end(1)
        if not any(_spans_overlap(name_start, name_end, e["start"], e["end"]) for e in regex_spans):
            regex_spans.append(
                {
                    "raw_value": name_raw,
                    "entity_type": "PERSON",
                    "start": name_start,
                    "end": name_end,
                }
            )

    # 6. NER Extraction (spaCy or Fallback)
    nlp = get_spacy_nlp()
    ner_spans: List[Dict[str, Any]] = []

    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG"):
                start, end = ent.start_char, ent.end_char
                raw = ent.text

                # Clean possessive 's or trailing punctuation
                while raw.endswith("'s") or raw.endswith("’s"):
                    raw = raw[:-2]
                    end -= 2

                raw_stripped = raw.rstrip(".,;:'\"")
                if len(raw_stripped) < len(raw):
                    end -= (len(raw) - len(raw_stripped))
                    raw = raw_stripped

                if not raw or raw in NON_PERSON_WORDS:
                    continue

                if any(_spans_overlap(start, end, re_ent["start"], re_ent["end"]) for re_ent in regex_spans):
                    continue

                ner_spans.append(
                    {
                        "raw_value": raw,
                        "entity_type": ent.label_,
                        "start": start,
                        "end": end,
                    }
                )

    # Fallback / Supplemental Org Matcher
    for match in FALLBACK_ORG_PATTERN.finditer(text):
        start, end = match.start(), match.end()
        raw = match.group(0).strip()
        if not any(_spans_overlap(start, end, e["start"], e["end"]) for e in (regex_spans + ner_spans)):
            ner_spans.append(
                {
                    "raw_value": raw,
                    "entity_type": "ORG",
                    "start": start,
                    "end": end,
                }
            )

    # Fallback / Supplemental Title Case Person matcher
    for match in FALLBACK_PERSON_NAME_PATTERN.finditer(text):
        start, end = match.start(), match.end()
        raw = match.group(0)
        if raw not in NON_PERSON_WORDS and not any(
            _spans_overlap(start, end, e["start"], e["end"]) for e in (regex_spans + ner_spans)
        ):
            ner_spans.append(
                {
                    "raw_value": raw,
                    "entity_type": "PERSON",
                    "start": start,
                    "end": end,
                }
            )

    # Merge adjacent NER entities of same type separated only by whitespace
    merged_ner: List[Dict[str, Any]] = []
    for ent in ner_spans:
        if not merged_ner:
            merged_ner.append(ent)
            continue
        prev = merged_ner[-1]
        inter_text = text[prev["end"]:ent["start"]]
        if prev["entity_type"] == ent["entity_type"] and inter_text.strip() == "":
            prev["end"] = ent["end"]
            prev["raw_value"] = text[prev["start"]:prev["end"]]
        else:
            merged_ner.append(ent)

    # Combine and sort by start index
    all_spans = regex_spans + merged_ner
    all_spans.sort(key=lambda x: x["start"])

    return [
        PIISpan(
            raw_value=item["raw_value"],
            entity_type=item["entity_type"],
            start=item["start"],
            end=item["end"],
        )
        for item in all_spans
    ]
