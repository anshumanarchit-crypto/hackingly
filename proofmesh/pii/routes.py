"""
FastAPI Routes for ProofMesh PII Vault & Redaction System.
"""

from fastapi import APIRouter, HTTPException, Response
from typing import Any, Dict, Optional
from .schemas import (
    RedactRequest,
    RedactResponse,
    PayloadFilterRequest,
    ExportRequest,
    ToggleStateResponse,
    PIISpan,
)
from .detector import extract_entities
from .redactor import redact_text, process_payload, get_default_vault
from .export import generate_export

router = APIRouter(prefix="/api/pii", tags=["PII Vault & Redaction"])

_GLOBAL_PRIVACY_STATE = {"redact_active": True}


@router.post("/redact", response_model=RedactResponse)
@router.post("/mask")
def redact_endpoint(request: RedactRequest):
    """
    Ingestion Endpoint.
    Sanitizes raw text into tokens before sending to retrieval / model context.
    """
    vault = get_default_vault()
    spans = extract_entities(request.text)
    redacted = redact_text(request.text, vault=vault)
    return {
        "redacted_text": redacted,
        "masked_text": redacted,
        "entities": [span.model_dump() for span in spans],
    }


@router.post("/deredact")
@router.post("/unmask")
def deredact_endpoint(payload: Dict[str, Any]):
    """
    Output Display Endpoint.
    Handles raw text unmasking or structured payload de-redaction.
    """
    vault = get_default_vault()
    if "text" in payload and payload["text"] is not None:
        from .redactor import deredact_text
        unmasked = deredact_text(payload["text"], vault=vault)
        return {"unmasked_text": unmasked, "raw_text": unmasked}

    if "answer" in payload:
        processed = process_payload(payload, vault=vault)
        return {"payload": processed}

    if "payload" in payload and isinstance(payload["payload"], dict):
        processed = process_payload(payload["payload"], vault=vault)
        return {"payload": processed}

    raise HTTPException(status_code=400, detail="Must provide 'text', 'answer', or 'payload'.")


@router.post("/process-payload")
def process_payload_endpoint(request: PayloadFilterRequest):
    """
    Endpoint for PayloadFilterRequest structured payload processing.
    """
    vault = get_default_vault()
    processed = process_payload(request, vault=vault)
    return {"payload": processed}


@router.post("/export")
def export_endpoint(request: ExportRequest):
    """
    Export Endpoint.
    Generates downloadable .txt or .pdf file of conversation history.
    Enforces privacy overrides and logs '[SECURITY AUDIT]' if unredacted.
    """
    vault = get_default_vault()
    try:
        file_bytes, media_type, filename = generate_export(
            session_id=request.session_id,
            format_type=request.format,
            redact=request.redact,
            conversation_history=request.conversation_history,
            vault=vault,
        )
        return Response(
            content=file_bytes,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/toggle-state", response_model=ToggleStateResponse)
def get_toggle_state():
    """
    Returns current global privacy redaction default state.
    """
    return ToggleStateResponse(redact_active=_GLOBAL_PRIVACY_STATE["redact_active"])
