"""
ProofMesh Hybrid Retriever (Person 2 Workstream).
Orchestrates standalone BM25 lexical retrieval and dense semantic retrieval streams,
performing score normalization, candidate union, reciprocal rank or weighted score fusion,
deduplication, and deterministic final ranking.

Key Features:
- Decoupled Candidate Sizing: bm25_top_k and semantic_top_k are decoupled from final_top_k,
  enabling candidates outside the top-3 of a single source to be promoted after fusion.
- Optional Query-Adaptive Weighting: Deterministic rule-based adjustment of alpha
  based on query features (numeric, IDs, quotes vs conceptual phrasing).
  MANDATORY DEFAULT: enable_adaptive_weighting = False.
- Full Explainability: Every result carries source rankings and normalized contributions.
"""

import re
import logging
from typing import List, Dict, Any, Optional

from .models import HybridResult
from .bm25_retriever import BM25Retriever
from .semantic_adapter import (
    SemanticRetrieverProtocol,
    MockSemanticRetriever,
    Person1SemanticAdapter,
)
from .fusion import FusionEngine

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Main entry point for Person 2 Hybrid Retrieval.
    Coordinates lexical and semantic retrieval pipelines without coupling to ProofMesh shared files.
    """

    def __init__(
        self,
        bm25_retriever: Optional[BM25Retriever] = None,
        semantic_retriever: Optional[Any] = None,
        fusion_strategy: str = "rrf",
        alpha: float = 0.5,
        rrf_k: int = 60,
        bm25_top_k: int = 5,
        semantic_top_k: int = 5,
        final_top_k: int = 3,
        enable_adaptive_weighting: bool = False,
    ):
        """
        Initialize the HybridRetriever.

        Args:
            bm25_retriever: BM25Retriever instance (creates default if None).
            semantic_retriever: Semantic retriever satisfying SemanticRetrieverProtocol,
                                or Person 1 SemanticRetriever, or MockSemanticRetriever.
            fusion_strategy: "rrf" (default) or "weighted".
            alpha: Base weighting for weighted fusion (default: 0.5).
            rrf_k: Smoothing constant for RRF (default: 60).
            bm25_top_k: Number of candidate chunks to fetch from BM25.
            semantic_top_k: Number of candidate chunks to fetch from semantic retriever.
            final_top_k: Number of fused results to return.
            enable_adaptive_weighting: Whether to dynamically adjust alpha based on query signals.
                                       MANDATORY: Defaults to False.
        """
        self.bm25 = bm25_retriever or BM25Retriever()

        # Wrap semantic retriever in Person1 adapter if not already an adapter/mock
        if semantic_retriever is None:
            self.semantic = MockSemanticRetriever()
        elif isinstance(semantic_retriever, (MockSemanticRetriever, Person1SemanticAdapter)):
            self.semantic = semantic_retriever
        else:
            self.semantic = Person1SemanticAdapter(semantic_retriever)

        self.fusion_strategy = fusion_strategy.strip().lower()
        self.alpha = max(0.0, min(1.0, alpha))
        self.rrf_k = max(1, rrf_k)
        self.bm25_top_k = max(1, bm25_top_k)
        self.semantic_top_k = max(1, semantic_top_k)
        self.final_top_k = max(1, final_top_k)
        self.enable_adaptive_weighting = enable_adaptive_weighting

        self.fusion_engine = FusionEngine()

    def add_document(self, doc_name: str, text: str) -> List[Dict[str, Any]]:
        """
        Indexes a raw document into both BM25 and Semantic indices.
        Returns the created chunks.
        """
        chunks = self.bm25.add_document(doc_name, text)
        if self.semantic is not None and hasattr(self.semantic, "add_chunks_directly"):
            self.semantic.add_chunks_directly(chunks)
        return chunks

    def add_chunks_directly(self, chunks: List[Dict[str, Any]]) -> None:
        """Indexes pre-formed chunks directly into both indices."""
        self.bm25.add_chunks_directly(chunks)
        if self.semantic is not None and hasattr(self.semantic, "add_chunks_directly"):
            self.semantic.add_chunks_directly(chunks)

    def _compute_adaptive_alpha(self, query: str) -> float:
        """
        Deterministic, rule-based query-adaptive weighting:
        - BM25 signals (+alpha towards 1.0):
          * Numbers / digits (e.g. "24", "12450")
          * Structured IDs / codes (e.g. "POL-1837", "chunk_1")
          * Quoted exact phrases (e.g. '"invoice reference"')
          * Short keyword queries (<= 3 words)
        - Semantic signals (-alpha towards 0.0):
          * Conversational question openers ("when does", "under what circumstances", "how does")
          * Long natural language descriptions (> 6 words without numbers/IDs)
        """
        base_alpha = self.alpha
        q_clean = query.strip()
        words = q_clean.split()
        num_words = len(words)

        has_numbers = bool(re.search(r'\b\d+\b', q_clean))
        has_id_pattern = bool(re.search(r'\b[A-Z]{2,}[-_]\d+\b', q_clean))
        has_quotes = '"' in q_clean or "'" in q_clean
        is_short_keyword = num_words <= 3 and not any(
            q_clean.lower().startswith(wh) for wh in ["how", "why", "when", "what", "where", "who"]
        )

        has_question_opener = any(
            q_clean.lower().startswith(p)
            for p in ["under what", "how long", "in what manner", "when does", "why is", "where do"]
        )
        is_long_conceptual = num_words >= 7 and not has_numbers and not has_id_pattern

        bm25_score_shift = 0.0
        if has_id_pattern:
            bm25_score_shift += 0.30
        if has_numbers:
            bm25_score_shift += 0.20
        if has_quotes:
            bm25_score_shift += 0.15
        if is_short_keyword:
            bm25_score_shift += 0.15

        semantic_score_shift = 0.0
        if has_question_opener:
            semantic_score_shift += 0.20
        if is_long_conceptual:
            semantic_score_shift += 0.20

        net_shift = bm25_score_shift - semantic_score_shift
        adapted = base_alpha + net_shift
        return round(max(0.10, min(0.90, adapted)), 2)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        strategy: Optional[str] = None,
        alpha: Optional[float] = None,
        rrf_k: Optional[int] = None,
        include_explainability: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval:
        1. Queries BM25 candidate stream.
        2. Queries Semantic candidate stream.
        3. Normalizes and fuses candidates via candidate union.
        4. Deduplicates and orders deterministically.
        5. Returns top_k results.
        """
        effective_top_k = top_k if top_k is not None else self.final_top_k
        effective_strategy = (strategy or self.fusion_strategy).strip().lower()
        effective_rrf_k = rrf_k if rrf_k is not None else self.rrf_k

        # Determine effective alpha
        if alpha is not None:
            effective_alpha = max(0.0, min(1.0, alpha))
        elif self.enable_adaptive_weighting:
            effective_alpha = self._compute_adaptive_alpha(query)
        else:
            effective_alpha = self.alpha

        # 1. Fetch BM25 candidates
        bm25_candidates = self.bm25.retrieve(query=query, top_k=self.bm25_top_k)

        # 2. Fetch Semantic candidates
        if self.semantic is not None:
            sem_candidates = self.semantic.retrieve(query=query, top_k=self.semantic_top_k)
        else:
            sem_candidates = []

        # 3. Fuse candidates
        fused_results: List[HybridResult] = self.fusion_engine.fuse(
            bm25_results=bm25_candidates,
            semantic_results=sem_candidates,
            strategy=effective_strategy,
            alpha=effective_alpha,
            rrf_k=effective_rrf_k,
            top_k=effective_top_k,
            include_explainability=include_explainability,
        )

        # 4. Filter by minimum score if specified
        output = []
        for r in fused_results:
            if min_score is not None and r.score < min_score:
                continue
            res_dict = r.to_dict(include_explainability=include_explainability)
            if self.enable_adaptive_weighting:
                res_dict["adaptive_alpha_applied"] = effective_alpha
            output.append(res_dict)

        return output

    def clear(self) -> None:
        """Clears both lexical and semantic indices."""
        self.bm25.clear()
        if self.semantic is not None and hasattr(self.semantic, "clear"):
            self.semantic.clear()

    def stats(self) -> Dict[str, Any]:
        """Returns diagnostic metadata about the hybrid retriever."""
        sem_type = type(self.semantic).__name__ if self.semantic else "None"
        return {
            "bm25": self.bm25.stats(),
            "semantic_provider_type": sem_type,
            "fusion_strategy": self.fusion_strategy,
            "alpha": self.alpha,
            "rrf_k": self.rrf_k,
            "bm25_top_k": self.bm25_top_k,
            "semantic_top_k": self.semantic_top_k,
            "final_top_k": self.final_top_k,
            "adaptive_weighting_enabled": self.enable_adaptive_weighting,
        }
