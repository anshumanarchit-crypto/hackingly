"""
Dense Semantic Retrieval module for ProofMesh (Person 1 Workstream).
Completely isolated, standalone subsystem.
"""

from .models import SemanticChunk, ChunkRecord, RetrievalResult
from .embeddings import SemanticEmbedder
from .semantic_retriever import SemanticRetriever

__all__ = [
    "SemanticChunk",
    "ChunkRecord",
    "RetrievalResult",
    "SemanticEmbedder",
    "SemanticRetriever",
]
