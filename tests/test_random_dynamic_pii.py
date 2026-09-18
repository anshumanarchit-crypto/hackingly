"""
Comprehensive Random Dynamic Test Suite for ProofMesh PII Engine (`test_random_dynamic_pii.py`).
Generates completely randomized PII entities, text structures, and edge cases to prove
the redaction, vault persistence, and de-redaction logic is 100% dynamic (not hardcoded).
"""

import os
import sys
import random
import string
import tempfile
import pytest
from fastapi.testclient import TestClient

# Ensure workspace root is in sys.path for direct script invocation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from proofmesh.pii_masker import mask_pii, unmask_pii, get_vault
from proofmesh.pii.vault import PIIVault
from proofmesh.pii.redactor import redact_text, deredact_text, process_payload
from proofmesh.api import app as fastapi_app

client = TestClient(fastapi_app)

FIRST_NAMES = [
    "Alexander", "Beatrix", "Carlos", "Dmitri", "Elena", "Fatima", "Giovanni", "Hannah",
    "Ibrahim", "Jasmin", "Kavya", "Liam", "Mei", "Nikolai", "Olivia", "Pranav",
    "Quentin", "Roxanne", "Siddharth", "Tariq", "Uma", "Vikram", "Winston", "Xavier",
    "Yara", "Zackary", "Aarav", "Priya", "Chen", "Sophia"
]

LAST_NAMES = [
    "Hamilton", "Kidman", "Santoro", "Volkov", "Rostova", "Khan", "Rossi", "Abbott",
    "Al-Mansoor", "Roy", "Sharma", "Nasser", "Lin", "Ivanov", "Rodriguez", "Patel",
    "Tarantino", "Vance", "Mehta", "Bakshi", "Thakur", "Rao", "Churchill", "Mendes",
    "Zhang", "Taylor", "Sengupta", "Deshmukh", "Chowdhury", "Kowalski"
]

TITLES = ["Patient", "Dr.", "Doctor", "Mr.", "Mrs.", "Ms.", "Prof.", "Officer", "Agent", "Lead", "Researcher"]

ORGS = [
    "Johns Hopkins Hospital", "Fortis Healthcare", "Apollo Hospitals", "Mayo Clinic",
    "Cyberdyne Systems", "Pemberley Labs", "HDFC Bank", "Saint Jude Hospital",
    "Stanford Medical Center", "Global Health Foundation", "Acme Corporation"
]


def generate_random_email():
    u = "".join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(6, 12)))
    d = "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 9)))
    tld = random.choice(["com", "org", "net", "edu", "io", "co.in", "gov"])
    return f"{u}@{d}.{tld}"


def generate_random_phone():
    fmt = random.choice([
        "+1-{a}-{b}-{c}",
        "+91-{a:05d}{b:05d}",
        "({a}) {b}-{c}",
        "{a}-{b}-{c}"
    ])
    a = random.randint(200, 999)
    b = random.randint(100, 999)
    c = random.randint(1000, 9999)
    return fmt.format(a=a, b=b, c=c)


def generate_random_ssn():
    return f"{random.randint(100, 999):03d}-{random.randint(10, 99):02d}-{random.randint(1000, 9999):04d}"


def generate_random_aadhaar():
    return f"{random.randint(1000, 9999):04d}-{random.randint(1000, 9999):04d}-{random.randint(1000, 9999):04d}"


def generate_random_person():
    title = random.choice(TITLES)
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    return title, f"{first} {last}"


def test_randomized_pii_redaction_and_roundtrip():
    """
    Generates 20 random PII scenarios with dynamically generated names, emails,
    phones, SSNs, and Aadhaar numbers. Tests that redact + unmask produces 100% exact match.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    vault = PIIVault(db_path=db_path)

    for i in range(20):
        title, name = generate_random_person()
        email = generate_random_email()
        phone = generate_random_phone()
        ssn = generate_random_ssn()
        aadhaar = generate_random_aadhaar()
        org = random.choice(ORGS)

        sentence = (
            f"{title} {name} (SSN: {ssn}, Aadhaar: {aadhaar}) visited {org}. "
            f"Contact via phone {phone} or email {email}."
        )

        masked = redact_text(sentence, vault=vault)

        # 1. Raw PII must NOT exist in masked text
        assert name not in masked, f"Name '{name}' leaked in sample {i}!"
        assert ssn not in masked, f"SSN '{ssn}' leaked in sample {i}!"
        assert aadhaar not in masked, f"Aadhaar '{aadhaar}' leaked in sample {i}!"
        assert email not in masked, f"Email '{email}' leaked in sample {i}!"
        assert phone not in masked, f"Phone '{phone}' leaked in sample {i}!"

        # 2. Placeholders must be inserted
        assert "[" in masked and "]" in masked, f"No tokens created in sample {i}!"

        # 3. Roundtrip unmasking must restore EXACT original string
        unmasked = deredact_text(masked, vault=vault)
        assert unmasked == sentence, f"Roundtrip mismatch in sample {i}!\nOriginal: {sentence}\nUnmasked: {unmasked}"

    vault.clear_vault()
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass


def test_randomized_token_persistence():
    """
    Verifies that when the exact same randomly generated entity is passed in multiple distinct sentences,
    the vault reuses the exact same assigned token identifier.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    vault = PIIVault(db_path=db_path)

    random_email = generate_random_email()
    random_phone = generate_random_phone()

    text1 = f"Please contact candidate at {random_email} or {random_phone} immediately."
    text2 = f"Reminder: notification sent to {random_email}, backup phone is {random_phone}."

    masked1 = redact_text(text1, vault=vault)
    masked2 = redact_text(text2, vault=vault)

    import re
    email_token1 = re.search(r"\[EMAIL_\d+\]", masked1).group(0)
    email_token2 = re.search(r"\[EMAIL_\d+\]", masked2).group(0)

    assert email_token1 == email_token2, f"Token mismatch for email! {email_token1} != {email_token2}"

    vault.clear_vault()
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass


def test_randomized_api_endpoints():
    """
    Tests the REST API endpoints using completely randomized input data.
    """
    rand_name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    rand_email = generate_random_email()
    rand_phone = generate_random_phone()

    text = f"Consultant {rand_name} email {rand_email} phone {rand_phone}"

    # Test /api/pii/mask
    r1 = client.post("/api/pii/mask", json={"text": text})
    assert r1.status_code == 200
    masked = r1.json()["masked_text"]
    assert rand_name not in masked
    assert rand_email not in masked

    # Test /api/pii/deredact
    r2 = client.post("/api/pii/deredact", json={"text": masked})
    assert r2.status_code == 200
    unmasked = r2.json()["unmasked_text"]
    assert unmasked == text


if __name__ == "__main__":
    print("Running randomized dynamic test suite...")
    test_randomized_pii_redaction_and_roundtrip()
    print("[PASS] Randomized PII Redaction & Roundtrip: PASSED (20/20 random iterations)")
    test_randomized_token_persistence()
    print("[PASS] Randomized Token Persistence: PASSED")
    test_randomized_api_endpoints()
    print("[PASS] Randomized REST API Endpoints: PASSED")
    print("ALL RANDOM DYNAMIC TESTS PASSED!")
