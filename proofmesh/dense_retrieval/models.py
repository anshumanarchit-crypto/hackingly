"""
Data models for ProofMesh Dense Semantic Retrieval subsystem.
Completely isolated and decoupled from existing ProofMesh modules.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class SemanticChunk:
    """
    Standard chunk representation for dense semantic retrieval.
    Designed to be 100% compatible with the ProofMesh DocumentChunk data contract:
    chunk_id, doc_name, text, score.
    """
    chunk_id: str
    doc_name: str
    text: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk into a standard ProofMesh dictionary."""
        result = {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "text": self.text,
            "score": float(self.score),
        }
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


# ChunkRecord is provided as an interchangeable alias for adapter flexibility
@dataclass
class ChunkRecord:
    """
    Generic chunk record representation for universal integration adapters.
    """
    chunk_id: str
    text: str
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_semantic_chunk(self) -> SemanticChunk:
        return SemanticChunk(
            chunk_id=self.chunk_id,
            doc_name=self.source,
            text=self.text,
            score=0.0,
            metadata=self.metadata,
        )


@dataclass
class RetrievalResult:
    """
    Structured container for semantic retrieval output.
    """
    chunk_id: str
    doc_name: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "text": self.text,
            "score": float(self.score),
        }
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result
