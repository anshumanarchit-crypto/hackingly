"""
Pydantic v2 Data Contracts for ProofMesh PII Module.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PIISpan(BaseModel):
    raw_value: str = Field(..., description="Raw text substring extracted.")
    entity_type: str = Field(..., description="Entity category (PERSON, ORG, PHONE, EMAIL, ID, AADHAAR).")
    start: int = Field(..., description="Character start index in raw text.")
    end: int = Field(..., description="Character end index in raw text.")


class RedactRequest(BaseModel):
    text: str = Field(..., description="Raw text containing potential PII.")


class RedactResponse(BaseModel):
    redacted_text: str = Field(..., description="Sanitized text with tokens.")
    entities: List[PIISpan] = Field(default_factory=list, description="Extracted PII spans.")


class Citation(BaseModel):
    doc_id: str
    text: str


class PayloadFilterRequest(BaseModel):
    answer: str = Field(..., description="Generated LLM response answer text.")
    citations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Retrieved citation objects."
    )
    redact_active: bool = Field(
        default=True,
        description="If True, payload remains redacted. If False, de-redact locally for UI display.",
    )


class ExportRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier.")
    format: str = Field(default="txt", description="Export format ('txt' or 'pdf').")
    redact: bool = Field(
        default=True,
        description="Whether to sanitize conversation history before export.",
    )
    conversation_history: List[Dict[str, Any]] = Field(
        ..., description="List of conversation message dicts."
    )


class ToggleStateResponse(BaseModel):
    redact_active: bool = Field(..., description="Current system privacy toggle status.")
