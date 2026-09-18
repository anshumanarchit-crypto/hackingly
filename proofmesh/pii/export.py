"""
Privacy-First Export Sanitizer Module.
Generates TXT and PDF export streams for conversation history with security audit logging.
"""

import io
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple
from .redactor import redact_text
from .vault import PIIVault

logger = logging.getLogger("proofmesh.pii.export")


def generate_export(
    session_id: str,
    format_type: str,
    redact: bool,
    conversation_history: List[Dict[str, Any]],
    vault: Optional[PIIVault] = None,
) -> Tuple[bytes, str, str]:
    """
    Generates downloadable conversation export (TXT or PDF).

    Security Audit Requirement:
    If redact is False, logs [SECURITY AUDIT] warning to console before writing raw output file.
    Note: Does not print or log raw PII contents.
    """
    fmt = format_type.lower().strip()
    if fmt not in ("txt", "pdf"):
        fmt = "txt"

    if not redact:
        audit_msg = (
            f"[SECURITY AUDIT] Raw export requested for session '{session_id}'. "
            "Sensitive unmasked data exported."
        )
        sys.stderr.write(f"\n{audit_msg}\n")
        logger.warning(audit_msg)

    processed_history = []
    for msg in conversation_history:
        role = str(msg.get("role", "User")).capitalize()
        raw_content = str(msg.get("content", ""))

        content = redact_text(raw_content, vault=vault) if redact else raw_content
        processed_history.append({"role": role, "content": content})

    if fmt == "pdf":
        return _generate_pdf_export(session_id, redact, processed_history)
    else:
        return _generate_txt_export(session_id, redact, processed_history)


def _generate_txt_export(
    session_id: str, redact: bool, history: List[Dict[str, str]]
) -> Tuple[bytes, str, str]:
    status_label = "REDACTED (PII Protected)" if redact else "RAW DATA (PII Unredacted)"
    lines = [
        "=" * 60,
        "PROOFMESH SESSION EXPORT",
        f"Session ID: {session_id}",
        f"Privacy Mode: {status_label}",
        "=" * 60,
        "",
    ]

    for item in history:
        lines.append(f"[{item['role']}]:")
        lines.append(f"{item['content']}")
        lines.append("")

    text_content = "\n".join(lines)
    filename = f"proofmesh_session_{session_id}.txt"
    return text_content.encode("utf-8"), "text/plain", filename


def _generate_pdf_export(
    session_id: str, redact: bool, history: List[Dict[str, str]]
) -> Tuple[bytes, str, str]:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
    except ImportError:
        raise RuntimeError(
            "PDF export requires the 'reportlab' package. "
            "Please install reportlab (`pip install reportlab`) or export using format='txt'."
        )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#1e293b"),
    )
    meta_style = ParagraphStyle(
        "MetaStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
    )
    role_style = ParagraphStyle(
        "RoleStyle",
        parent=styles["Heading3"],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
    )
    content_style = ParagraphStyle(
        "ContentStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )

    elements = []
    status_label = "REDACTED (PII Protected)" if redact else "RAW DATA (PII Unredacted)"

    elements.append(Paragraph("ProofMesh Conversation Export", title_style))
    elements.append(Paragraph(f"<b>Session ID:</b> {session_id}", meta_style))
    elements.append(Paragraph(f"<b>Privacy Mode:</b> {status_label}", meta_style))
    elements.append(Spacer(1, 15))

    for item in history:
        elements.append(Paragraph(f"[{item['role']}]", role_style))
        safe_text = (
            item["content"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )
        elements.append(Paragraph(safe_text, content_style))
        elements.append(Spacer(1, 10))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    filename = f"proofmesh_session_{session_id}.pdf"
    return pdf_bytes, "application/pdf", filename
