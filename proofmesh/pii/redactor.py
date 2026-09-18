"""
Bidirectional Privacy Engine (Redaction & Local De-redaction).
Sanitizes raw text into tokens before leaving local host environment,
and de-sanitizes for local display upon explicit toggle action.
"""

import re
from typing import Any, Dict, List, Optional, Union
from .schemas import PayloadFilterRequest, PIISpan
from .detector import extract_entities
from .vault import PIIVault

_DEFAULT_VAULT: Optional[PIIVault] = None


def get_default_vault() -> PIIVault:
    global _DEFAULT_VAULT
    if _DEFAULT_VAULT is None:
        _DEFAULT_VAULT = PIIVault()
    return _DEFAULT_VAULT


# Regex to match token placeholders, e.g. [PERSON_001], [EMAIL_002], [PHONE_003], [AADHAAR_001], [ID_001]
TOKEN_PATTERN = re.compile(r"\[([A-Z]+_\d+)\]")


def _spans_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    return max(s1, s2) < min(e1, e2)


def redact_text(text: str, vault: Optional[PIIVault] = None) -> str:
    """
    Sanitizes text by replacing raw PII spans with Vault tokens.

    CRITICAL ALGORITHM STEP:
    Sorts detected spans descending by 'start' index before string splicing
    so length changes do not invalidate remaining indices.
    """
    if not text:
        return text

    if vault is None:
        vault = get_default_vault()

    spans = extract_entities(text)
    span_dicts = [
        {
            "raw_value": span.raw_value,
            "entity_type": span.entity_type,
            "start": span.start,
            "end": span.end,
        }
        for span in spans
    ]

    # Multi-turn persistence check against existing vault mappings
    known_entries = vault.get_all_entries()
    if known_entries:
        for raw_val, ent_type in known_entries.items():
            if not raw_val:
                continue
            pattern = re.compile(re.escape(raw_val), re.IGNORECASE)
            for m in pattern.finditer(text):
                s, e = m.start(), m.end()
                if not any(_spans_overlap(s, e, sp["start"], sp["end"]) for sp in span_dicts):
                    span_dicts.append(
                        {
                            "raw_value": m.group(0),
                            "entity_type": ent_type,
                            "start": s,
                            "end": e,
                        }
                    )

    if not span_dicts:
        return text

    # Sort descending by start index to maintain index alignment
    sorted_spans = sorted(span_dicts, key=lambda x: x["start"], reverse=True)

    redacted_text = text
    for item in sorted_spans:
        start = item["start"]
        end = item["end"]
        raw = item["raw_value"]
        ent_type = item["entity_type"]

        token = vault.get_or_create_token(raw, ent_type)
        redacted_text = redacted_text[:start] + token + redacted_text[end:]

    return redacted_text


def deredact_text(redacted_text: str, vault: Optional[PIIVault] = None) -> str:
    """
    Replaces token placeholders like [PERSON_001] with raw values retrieved from Vault.
    If a token is missing from the Vault (e.g. [PERSON_999]), leaves it unchanged.
    """
    if not redacted_text or not isinstance(redacted_text, str):
        return redacted_text

    if vault is None:
        vault = get_default_vault()

    def replace_token(match: re.Match) -> str:
        token = match.group(0)
        raw_val = vault.get_raw_value(token)
        return raw_val if raw_val is not None else token

    return TOKEN_PATTERN.sub(replace_token, redacted_text)


def process_payload(
    payload: Union[PayloadFilterRequest, Dict[str, Any]],
    vault: Optional[PIIVault] = None,
) -> Dict[str, Any]:
    """
    Intercepts response payload from inference engine.

    If redact_active is True (default safe state), returns payload unmodified.
    If redact_active is False, de-redacts answer and citations strings for local UI display.
    """
    if isinstance(payload, PayloadFilterRequest):
        redact_active = payload.redact_active
        data_dict = {
            "answer": payload.answer,
            "citations": payload.citations,
            "redact_active": redact_active,
        }
    else:
        redact_active = payload.get("redact_active", True)
        data_dict = dict(payload)

    if redact_active:
        return data_dict

    if vault is None:
        vault = get_default_vault()

    def _recursive_deredact(val: Any) -> Any:
        if isinstance(val, str):
            return deredact_text(val, vault=vault)
        elif isinstance(val, dict):
            return {k: _recursive_deredact(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [_recursive_deredact(item) for item in val]
        return val

    return _recursive_deredact(data_dict)


# Alias
deredact_payload = process_payload
