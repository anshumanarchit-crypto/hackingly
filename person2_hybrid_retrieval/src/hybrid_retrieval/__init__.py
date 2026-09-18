"""
ProofMesh Hybrid Retrieval Subsystem (Person 2 Workstream).
Independent, isolated lexical-semantic fusion and retrieval optimization package.
"""

from .models import HybridChunk, ScoredChunk, HybridResult
from .score_normalizer import ScoreNormalizer
from .bm25_retriever import BM25Retriever
from .semantic_adapter import (
    SemanticRetrieverProtocol,
    MockSemanticRetriever,
    Person1SemanticAdapter,
)
from .fusion import FusionEngine
from .hybrid_retriever import HybridRetriever

__all__ = [
    "HybridChunk",
    "ScoredChunk",
    "HybridResult",
    "ScoreNormalizer",
    "BM25Retriever",
    "SemanticRetrieverProtocol",
    "MockSemanticRetriever",
    "Person1SemanticAdapter",
    "FusionEngine",
    "HybridRetriever",
]
