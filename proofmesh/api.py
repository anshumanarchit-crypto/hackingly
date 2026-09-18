"""
ProofMesh REST API Entrypoint (`proofmesh/api.py`).
Exposes PII masking, local de-redaction, and privacy-safe export endpoints.
"""

from fastapi import FastAPI
from .pii.routes import router as pii_router

app = FastAPI(
    title="ProofMesh API",
    description="Privacy-First PII Vault & Redaction REST API for ProofMesh RAG",
    version="1.0.0",
)

app.include_router(pii_router)


@app.get("/")
def root():
    return {
        "status": "online",
        "system": "ProofMesh API",
        "service": "PII Vault & Evidence Gating",
    }
