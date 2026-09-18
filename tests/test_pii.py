"""
Comprehensive Feature 3 Test Suite: PII Vault & Privacy Protection.
Validates zero-hardcoding, dynamic entity extractions, token persistence,
span overlap resolution, descending index replacement math, security audit logging,
REST API endpoints, and full PIIMasker/Pipeline integration.
"""

import os
import tempfile
import pytest
from fastapi.testclient import TestClient

from proofmesh.pii.schemas import PayloadFilterRequest, RedactRequest, PIISpan
from proofmesh.pii.vault import PIIVault
from proofmesh.pii.detector import extract_entities, _spans_overlap
from proofmesh.pii.redactor import redact_text, deredact_text, process_payload
from proofmesh.pii.export import generate_export
from proofmesh.pii_masker import PIIMasker, mask_pii, unmask_pii, get_vault
from proofmesh.pipeline import ProofMeshPipeline
from proofmesh.api import app as fastapi_app


@pytest.fixture
def temp_vault():
    """Fixture for clean temporary SQLite PIIVault instance."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    vault = PIIVault(db_path=db_path)
    yield vault
    vault.clear_vault()
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass


@pytest.fixture
def client():
    return TestClient(fastapi_app)


# ----------------------------------------------------------------------
# 1. DYNAMIC DETECTION & EXTRACTION TESTS
# ----------------------------------------------------------------------

def test_dynamic_entity_extraction_all_types():
    """Validates dynamic extraction of EMAIL, PHONE, ID, AADHAAR, PERSON, and ORG."""
    text = (
        "Contact researcher Alexander Hamilton at alexander.h@treasury.gov or +1-202-555-0143. "
        "Case ID 123-45-6789 and Aadhaar card number 9876-5432-1098 presented at Federal Reserve Bank."
    )
    spans = extract_entities(text)
    types_found = {span.entity_type for span in spans}

    assert "EMAIL" in types_found, "EMAIL entity not detected"
    assert "PHONE" in types_found, "PHONE entity not detected"
    assert "ID" in types_found, "ID entity not detected"
    assert "AADHAAR" in types_found, "AADHAAR entity not detected"
    assert "PERSON" in types_found, "PERSON entity not detected"
    assert "ORG" in types_found, "ORG entity not detected"


def test_aadhaar_token_format_convention(temp_vault):
    """
    CRITICAL INVARIANT: Aadhaar numbers must receive [AADHAAR_NNN] tokens,
    while generic IDs receive [ID_NNN] tokens.
    """
    text = "Patient Aadhaar 5432 1098 7654 and SSN 123-45-6789."
    masked = redact_text(text, vault=temp_vault)

    assert "[AADHAAR_001]" in masked, f"Expected [AADHAAR_001] in '{masked}'"
    assert "[ID_001]" in masked, f"Expected [ID_001] in '{masked}'"


def test_overlapping_span_resolution():
    """Ensures that no two extracted entity spans overlap in character indices."""
    text = "Please reach out to john.smith@company.org regarding case number 123-45-6789."
    spans = extract_entities(text)

    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            s1, e1 = spans[i].start, spans[i].end
            s2, e2 = spans[j].start, spans[j].end
            assert not _spans_overlap(s1, e1, s2, e2), f"Spans overlap: {spans[i]} vs {spans[j]}"


# ----------------------------------------------------------------------
# 2. TOKEN STABILITY & REPEATED VALUE TESTS
# ----------------------------------------------------------------------

def test_repeated_entity_token_reuse(temp_vault):
    """Identical raw values must always receive the exact same token."""
    raw_name = "Rahul Sharma"
    tok1 = temp_vault.get_or_create_token(raw_name, "PERSON")
    tok2 = temp_vault.get_or_create_token(raw_name, "PERSON")

    assert tok1 == tok2 == "[PERSON_001]"
    assert temp_vault.get_raw_value(tok1) == raw_name


def test_distinct_values_get_different_tokens(temp_vault):
    """Different values of the same entity type receive different sequential tokens."""
    tok1 = temp_vault.get_or_create_token("Alice Smith", "PERSON")
    tok2 = temp_vault.get_or_create_token("Bob Jones", "PERSON")

    assert tok1 != tok2
    assert tok1 == "[PERSON_001]"
    assert tok2 == "[PERSON_002]"


def test_vault_persistence_across_instances():
    """Verifies that mappings persist on disk and reload across separate PIIVault instances."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Instance 1: write
        v1 = PIIVault(db_path=db_path)
        tok_v1 = v1.get_or_create_token("Dr. Clara Oswald", "PERSON")
        assert tok_v1 == "[PERSON_001]"

        # Instance 2: read from existing file
        v2 = PIIVault(db_path=db_path)
        raw_val = v2.get_raw_value("[PERSON_001]")
        assert raw_val == "Dr. Clara Oswald"

        # Re-requesting the same value in v2 must return the existing token
        tok_v2 = v2.get_or_create_token("Dr. Clara Oswald", "PERSON")
        assert tok_v2 == "[PERSON_001]"
    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except OSError:
                pass


# ----------------------------------------------------------------------
# 3. REDACTION, DE-REDACTION & ROUND-TRIP TESTS
# ----------------------------------------------------------------------

def test_descending_index_redaction_and_roundtrip(temp_vault):
    """Tests descending index string splicing and exact round-trip de-redaction."""
    raw_text = "Dr. Clara Oswald called patient Amy Pond at 555-123-4567."
    redacted = redact_text(raw_text, vault=temp_vault)

    assert "Clara Oswald" not in redacted
    assert "Amy Pond" not in redacted
    assert "555-123-4567" not in redacted
    assert "[" in redacted and "]" in redacted

    deredacted = deredact_text(redacted, vault=temp_vault)
    assert deredacted == raw_text


def test_missing_token_remains_unchanged(temp_vault):
    """An unknown token not in the vault (e.g. [PERSON_999]) remains unchanged."""
    text_with_unknown = "The report was signed by [PERSON_999]."
    result = deredact_text(text_with_unknown, vault=temp_vault)
    assert result == text_with_unknown


def test_process_payload_toggle(temp_vault):
    """Tests payload filtering: unmodified when redact_active=True, de-redacted when False."""
    tok = temp_vault.get_or_create_token("Sherlock Holmes", "PERSON")

    payload = PayloadFilterRequest(
        answer=f"The investigation was led by {tok}.",
        citations=[{"doc_id": "c1", "text": f"Consulting detective {tok} was present."}],
        redact_active=True,
    )

    # Safe state (True)
    safe = process_payload(payload, vault=temp_vault)
    assert safe["answer"] == f"The investigation was led by {tok}."

    # Raw state (False)
    payload_raw = PayloadFilterRequest(
        answer=f"The investigation was led by {tok}.",
        citations=[{"doc_id": "c1", "text": f"Consulting detective {tok} was present."}],
        redact_active=False,
    )
    unmasked = process_payload(payload_raw, vault=temp_vault)
    assert "Sherlock Holmes" in unmasked["answer"]
    assert "Sherlock Holmes" in unmasked["citations"][0]["text"]


# ----------------------------------------------------------------------
# 4. EXPORT & SECURITY AUDIT WARNING TESTS
# ----------------------------------------------------------------------

def test_export_security_audit_logging(temp_vault, capsys):
    """TXT export works and unredacted raw export triggers security audit log warning to stderr."""
    history = [{"role": "user", "content": "Sensitive note for Bruce Wayne."}]

    # Redacted export
    b1, m1, f1 = generate_export("sess_01", "txt", True, history, vault=temp_vault)
    assert "Bruce Wayne" not in b1.decode("utf-8")
    assert f1.endswith(".txt")

    # Unredacted export triggers security audit warning
    b2, m2, f2 = generate_export("sess_02", "txt", False, history, vault=temp_vault)
    captured = capsys.readouterr()
    assert "[SECURITY AUDIT]" in captured.err
    assert "sess_02" in captured.err
    assert "Bruce Wayne" in b2.decode("utf-8")


# ----------------------------------------------------------------------
# 5. REST API ENDPOINTS
# ----------------------------------------------------------------------

def test_api_redact_endpoint(client):
    res = client.post("/api/pii/redact", json={"text": "Contact Arthur Dent at arthur@galaxy.org"})
    assert res.status_code == 200
    data = res.json()
    assert "redacted_text" in data
    assert "Arthur Dent" not in data["redacted_text"]


def test_api_deredact_endpoint(client):
    r1 = client.post("/api/pii/redact", json={"text": "Patient James Bond ID 007"})
    redacted_text = r1.json()["redacted_text"]

    payload = {
        "text": redacted_text
    }
    r2 = client.post("/api/pii/deredact", json=payload)
    assert r2.status_code == 200
    assert "James Bond" in r2.json()["unmasked_text"]


def test_api_toggle_state_endpoint(client):
    res = client.get("/api/pii/toggle-state")
    assert res.status_code == 200
    assert res.json()["redact_active"] is True


# ----------------------------------------------------------------------
# 6. PIIMasker CLASS COMPATIBILITY & ROUND-TRIP
# ----------------------------------------------------------------------

def test_piimasker_api_compatibility(temp_vault):
    """Validates PIIMasker class with mask_text, mask_named_entity, unmask_text, and mappings."""
    masker = PIIMasker(vault=temp_vault)
    raw = "Lead researcher John Doe can be reached at john.doe@example.com or Aadhaar 5432 1098 7654."
    
    masked = masker.mask_text(raw)
    masked = masker.mask_named_entity(masked, ["John Doe"])

    assert "[EMAIL_001]" in masked
    assert "[AADHAAR_001]" in masked
    assert "[PERSON_001]" in masked
    assert "John Doe" not in masked

    # Check mappings dict properties
    assert "john.doe@example.com" in masker.mappings
    assert "[EMAIL_001]" in masker.reverse_mappings

    # Round-trip unmask
    unmasked = masker.unmask_text(masked)
    assert unmasked == raw


# ----------------------------------------------------------------------
# 7. PROOFMESH PIPELINE INTEGRATION
# ----------------------------------------------------------------------

def test_pipeline_instantiation_and_document_indexing():
    """Verifies ProofMeshPipeline instantiates, indexes document with PII, and queries safely."""
    pipeline = ProofMeshPipeline()
    sample_doc = "Patient Rahul Sharma (phone +91 9876543210, Aadhaar 9876 5432 1098) was treated at Clinic."
    chunks = pipeline.index_document("patient_record.txt", sample_doc, mask_pii=True)

    # Chunks in index must have PII masked
    indexed_text = " ".join(c.text for c in chunks)
    assert "Rahul Sharma" not in indexed_text
    assert "9876543210" not in indexed_text

    # Safe query test
    query = "What is the patient's phone number?"
    res = pipeline.query(query, mask_query_pii=True, redact_active=True)
    assert res["status"] in ("ANSWER_GENERATED", "CONTRADICTION_HALT", "EVIDENCE_GATE_HALT")
    # Raw phone must NOT be in answer
    assert "9876543210" not in res["answer"]
