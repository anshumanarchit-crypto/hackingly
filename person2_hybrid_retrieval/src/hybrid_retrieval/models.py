"""
Data models for ProofMesh Hybrid Retrieval (Person 2 Workstream).
Defines standard contracts for chunks, single-source scored candidates,
and unified hybrid results with explainability metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class HybridChunk:
    """Standard chunk representation prior to scoring."""
    chunk_id: str
    doc_name: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "text": self.text,
        }
        if self.metadata:
            res["metadata"] = dict(self.metadata)
        return res


@dataclass
class ScoredChunk:
    """Candidate chunk scored by an individual retrieval mechanism (BM25 or Dense)."""
    chunk_id: str
    doc_name: str
    text: str
    score: float
    rank: Optional[int] = None
    source: str = "unknown"  # "bm25" or "semantic"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "text": self.text,
            "score": float(self.score),
        }
        if self.rank is not None:
            res["rank"] = self.rank
        if self.source:
            res["source"] = self.source
        if self.metadata:
            res["metadata"] = dict(self.metadata)
        return res


@dataclass
class HybridResult:
    """
    Unified hybrid retrieval result combining lexical (BM25) and dense semantic signals.
    Carries complete explainability metadata.
    """
    chunk_id: str
    doc_name: str
    text: str
    score: float  # Final fused hybrid score
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    semantic_score: Optional[float] = None
    semantic_rank: Optional[int] = None
    normalized_bm25_score: Optional[float] = None
    normalized_semantic_score: Optional[float] = None
    fusion_strategy: str = ""
    alpha: Optional[float] = None
    rrf_k: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_explainability: bool = True) -> Dict[str, Any]:
        """Convert to standard dictionary matching ProofMesh schema with optional explainability."""
        res: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "text": self.text,
            "score": float(self.score),
        }
        if include_explainability:
            if self.bm25_score is not None:
                res["bm25_score"] = float(self.bm25_score)
            if self.bm25_rank is not None:
                res["bm25_rank"] = int(self.bm25_rank)
            if self.semantic_score is not None:
                res["semantic_score"] = float(self.semantic_score)
            if self.semantic_rank is not None:
                res["semantic_rank"] = int(self.semantic_rank)
            if self.normalized_bm25_score is not None:
                res["normalized_bm25_score"] = float(self.normalized_bm25_score)
            if self.normalized_semantic_score is not None:
                res["normalized_semantic_score"] = float(self.normalized_semantic_score)
            if self.fusion_strategy:
                res["fusion_strategy"] = self.fusion_strategy
            if self.alpha is not None:
                res["alpha"] = float(self.alpha)
            if self.rrf_k is not None:
                res["rrf_k"] = int(self.rrf_k)

        if self.metadata:
            res["metadata"] = dict(self.metadata)
        return res

    def explain(self) -> str:
        """Human-readable explanation of why this chunk was ranked."""
        parts = [f"Chunk: {self.chunk_id} | Final Score: {self.score:.4f} ({self.fusion_strategy})"]
        if self.bm25_rank is not None:
            parts.append(f"BM25 Rank: {self.bm25_rank} (raw: {self.bm25_score:.3f}, norm: {self.normalized_bm25_score})")
        else:
            parts.append("BM25: Not retrieved")
        if self.semantic_rank is not None:
            parts.append(f"Semantic Rank: {self.semantic_rank} (raw: {self.semantic_score:.3f}, norm: {self.normalized_semantic_score})")
        else:
            parts.append("Semantic: Not retrieved")
        return " | ".join(parts)
