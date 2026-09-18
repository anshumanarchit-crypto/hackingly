"""
ProofMesh PII Masker & Vault Interface
---------------------------------------
Masks personal identifiable information into opaque tokens ([PERSON_001], [EMAIL_001], etc.)
to guarantee privacy preservation before text enters the evidence retrieval & LLM pipeline.
Provides dynamic PII detection, tokenization, and reverse de-redaction
with SQLite disk persistence.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

# Ensure root workspace is in sys.path for standalone and package execution
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_curr_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from proofmesh.pii.vault import PIIVault
from proofmesh.pii.detector import extract_entities
from proofmesh.pii.redactor import (
    redact_text,
    deredact_text,
    process_payload,
    deredact_payload,
)

_SHARED_VAULT: Optional[PIIVault] = None


def get_vault(db_path: Union[str, Path] = "pii_vault.db") -> PIIVault:
    """
    Returns a persistent SQLite PIIVault instance.
    """
    global _SHARED_VAULT
    path_str = str(db_path)
    if _SHARED_VAULT is None or _SHARED_VAULT.db_path != path_str:
        _SHARED_VAULT = PIIVault(db_path=path_str)
    return _SHARED_VAULT


def mask_pii(text: str, vault: Optional[PIIVault] = None) -> str:
    """
    Sanitizes raw text by masking PII into tokens (e.g. [PERSON_001], [EMAIL_001]).
    Persists raw value to token mapping in local SQLite vault.
    """
    active_vault: PIIVault = vault if vault is not None else get_vault()
    return redact_text(text, vault=active_vault)


def unmask_pii(masked_text: str, vault: Optional[PIIVault] = None) -> str:
    """
    Reverses PII tokenization by looking up raw values in the local SQLite vault.
    """
    active_vault: PIIVault = vault if vault is not None else get_vault()
    return deredact_text(masked_text, vault=active_vault)


# Aliases for backward compatibility
redact_pii = mask_pii
deredact_pii = unmask_pii


class PIIMasker:
    """
    Canonical ProofMesh PII Masker & Pseudonymization Engine.
    Detects and replaces PII with opaque sequential placeholders backed by PIIVault.
    Preserves existing ProofMesh pipeline API contracts.
    """

    vault: PIIVault

    def __init__(
        self,
        vault: Optional[PIIVault] = None,
        db_path: Union[str, Path] = "pii_vault.db",
        reset_on_init: bool = True,
    ):
        if vault is not None:
            self.vault = vault
        else:
            self.vault = get_vault(db_path=db_path)
            if reset_on_init:
                self.reset_mappings()

    def reset_mappings(self) -> None:
        """Clears all mappings in the underlying vault."""
        self.vault.clear_vault()

    @property
    def mappings(self) -> Dict[str, str]:
        """Returns dict of raw_value -> entity_id token."""
        return self.vault.get_all_mappings()

    @property
    def reverse_mappings(self) -> Dict[str, str]:
        """Returns dict of entity_id token -> raw_value."""
        return self.vault.get_all_reverse_mappings()

    def mask_text(self, text: str) -> str:
        """
        Masks emails, phone numbers, Aadhaar numbers, IDs, and known sensitive patterns.
        """
        return redact_text(text, vault=self.vault)

    def mask_named_entity(self, text: str, person_names: List[str]) -> str:
        """
        Masks explicit person names using vault token assignment.
        """
        masked = text
        for name in sorted(person_names, key=len, reverse=True):
            clean_name = name.strip()
            if clean_name and clean_name in masked:
                ph = self.vault.get_or_create_token(clean_name, "PERSON")
                pattern = re.compile(re.escape(clean_name), re.IGNORECASE)
                masked = pattern.sub(ph, masked)
        return masked

    def unmask_text(self, text: str) -> str:
        """
        Restores original PII for authorized offline consumers.
        """
        return deredact_text(text, vault=self.vault)


__all__ = [
    "PIIMasker",
    "mask_pii",
    "unmask_pii",
    "get_vault",
    "redact_pii",
    "deredact_pii",
    "extract_entities",
    "process_payload",
    "deredact_payload",
]


if __name__ == "__main__":
    print("Testing merged PIIMasker standalone...")
    masker = PIIMasker()
    masker.reset_mappings()
    raw = "Contact Alice at alice@example.com or phone +91 9876543210."
    masked = masker.mask_text(raw)
    masked = masker.mask_named_entity(masked, ["Alice"])
    print("Raw:     ", raw)
    print("Masked:  ", masked)
    unmasked = masker.unmask_text(masked)
    print("Unmasked:", unmasked)
    assert "Alice" in unmasked
    assert "alice@example.com" in unmasked
    print("Success: Merged PIIMasker executed cleanly.")

