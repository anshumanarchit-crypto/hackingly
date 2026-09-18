"""
Hybrid Fusion Engine for ProofMesh (Person 2 Workstream).
Combines lexical (BM25) and dense semantic retrieval candidate streams using
Candidate Union, Deduplication, and Deterministic Scoring.

Supported Fusion Strategies:
1. Strategy A: Weighted Normalized Fusion
   hybrid_score = alpha * norm(bm25_score) + (1 - alpha) * norm(semantic_score)
   where alpha in [0.0, 1.0].

2. Strategy B: Reciprocal Rank Fusion (RRF)
   RRF_score = sum(1 / (k + rank_m)) for m in {bm25, semantic}
   where k > 0 (default: 60) and rank_m is 1-indexed.

Key Invariants:
- Candidate Union: Operates on union(BM25, Semantic), NOT intersection.
- Zero-Division & Missing Source Safety: If a chunk appears in only one source,
  missing contributions are assigned 0.0 without errors.
- Deduplication: Each chunk ID appears exactly once in final rankings.
- Deterministic Tie-Breaking: Primary sort on hybrid_score descending;
  secondary tie-breaker on chunk_id ascending.
"""

from typing import List, Dict, Any, Optional
from .models import HybridResult
from .score_normalizer import ScoreNormalizer


class FusionEngine:
    """
    Executes score normalization, candidate union, score fusion, deduplication,
    and deterministic ranking across disparate retrieval result lists.
    """

    def __init__(self, normalizer: Optional[ScoreNormalizer] = None):
        self.normalizer = normalizer or ScoreNormalizer()

    def fuse(
        self,
        bm25_results: List[Dict[str, Any]],
        semantic_results: List[Dict[str, Any]],
        strategy: str = "rrf",
        alpha: float = 0.5,
        rrf_k: int = 60,
        top_k: int = 3,
        include_explainability: bool = True,
    ) -> List[HybridResult]:
        """
        Main fusion entry point. Dispatches to selected strategy and returns
        deduplicated, deterministically ranked HybridResults.

        Args:
            bm25_results: Ranked list of lexical candidate dicts.
            semantic_results: Ranked list of semantic candidate dicts.
            strategy: "rrf" or "weighted".
            alpha: Weighting factor for weighted fusion (0.0 to 1.0).
            rrf_k: Smoothing constant for RRF (default: 60).
            top_k: Maximum number of final results to return.
            include_explainability: Whether to populate detailed explainability fields.

        Returns:
            List of HybridResult objects.
        """
        strat_lower = strategy.strip().lower()
        if strat_lower == "weighted":
            return self.weighted_normalized_fusion(
                bm25_results=bm25_results,
                semantic_results=semantic_results,
                alpha=alpha,
                top_k=top_k,
            )
        elif strat_lower == "rrf":
            return self.reciprocal_rank_fusion(
                bm25_results=bm25_results,
                semantic_results=semantic_results,
                k=rrf_k,
                top_k=top_k,
            )
        else:
            raise ValueError(
                f"Unsupported fusion strategy '{strategy}'. Supported options: 'rrf', 'weighted'."
            )

    def weighted_normalized_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        semantic_results: List[Dict[str, Any]],
        alpha: float = 0.5,
        top_k: int = 3,
    ) -> List[HybridResult]:
        """
        Executes Weighted Normalized Score Fusion:
            hybrid_score = alpha * norm_bm25 + (1 - alpha) * norm_semantic
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in range [0.0, 1.0], got {alpha}")
        if top_k <= 0:
            return []

        # 1. Normalize BM25 scores
        raw_bm25_scores = [float(r.get("score", 0.0)) for r in bm25_results]
        norm_bm25_scores = self.normalizer.min_max_normalize(raw_bm25_scores)

        # 2. Normalize Semantic scores
        raw_sem_scores = [float(r.get("score", 0.0)) for r in semantic_results]
        norm_sem_scores = self.normalizer.min_max_normalize(raw_sem_scores)

        # 3. Build candidate union lookup
        # chunk_id -> info
        candidates: Dict[str, Dict[str, Any]] = {}

        # Ingest BM25 candidates
        for idx, (r, norm_s) in enumerate(zip(bm25_results, norm_bm25_scores)):
            cid = str(r.get("chunk_id", ""))
            candidates[cid] = {
                "chunk_id": cid,
                "doc_name": r.get("doc_name", ""),
                "text": r.get("text", ""),
                "bm25_score": float(r.get("score", 0.0)),
                "bm25_rank": idx + 1,
                "norm_bm25": float(norm_s),
                "semantic_score": None,
                "semantic_rank": None,
                "norm_sem": 0.0,
                "metadata": r.get("metadata", {}),
            }

        # Ingest / merge Semantic candidates
        for idx, (r, norm_s) in enumerate(zip(semantic_results, norm_sem_scores)):
            cid = str(r.get("chunk_id", ""))
            if cid in candidates:
                # Merge into existing candidate
                candidates[cid]["semantic_score"] = float(r.get("score", 0.0))
                candidates[cid]["semantic_rank"] = idx + 1
                candidates[cid]["norm_sem"] = float(norm_s)
                # Supplement metadata if missing
                if not candidates[cid]["doc_name"]:
                    candidates[cid]["doc_name"] = r.get("doc_name", "")
                if not candidates[cid]["text"]:
                    candidates[cid]["text"] = r.get("text", "")
            else:
                candidates[cid] = {
                    "chunk_id": cid,
                    "doc_name": r.get("doc_name", ""),
                    "text": r.get("text", ""),
                    "bm25_score": None,
                    "bm25_rank": None,
                    "norm_bm25": 0.0,
                    "semantic_score": float(r.get("score", 0.0)),
                    "semantic_rank": idx + 1,
                    "norm_sem": float(norm_s),
                    "metadata": r.get("metadata", {}),
                }

        # 4. Compute combined weighted hybrid score
        scored_results: List[HybridResult] = []
        for cid, info in candidates.items():
            norm_b = info["norm_bm25"]
            norm_s = info["norm_sem"]

            if alpha >= 1.0:
                final_score = norm_b
            elif alpha <= 0.0:
                final_score = norm_s
            else:
                final_score = alpha * norm_b + (1.0 - alpha) * norm_s

            scored_results.append(
                HybridResult(
                    chunk_id=cid,
                    doc_name=info["doc_name"],
                    text=info["text"],
                    score=float(final_score),
                    bm25_score=info["bm25_score"],
                    bm25_rank=info["bm25_rank"],
                    semantic_score=info["semantic_score"],
                    semantic_rank=info["semantic_rank"],
                    normalized_bm25_score=round(norm_b, 4) if info["bm25_score"] is not None else None,
                    normalized_semantic_score=round(norm_s, 4) if info["semantic_score"] is not None else None,
                    fusion_strategy="weighted",
                    alpha=alpha,
                    rrf_k=None,
                    metadata=info["metadata"],
                )
            )

        # 5. Deterministic tie-breaking: score descending, chunk_id ascending
        scored_results.sort(key=lambda r: (-r.score, r.chunk_id))
        return scored_results[:top_k]

    def reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        semantic_results: List[Dict[str, Any]],
        k: int = 60,
        top_k: int = 3,
    ) -> List[HybridResult]:
        """
        Executes Reciprocal Rank Fusion (RRF):
            RRF_score = sum(1 / (k + rank_m))
        """
        if k <= 0:
            raise ValueError(f"RRF parameter k must be positive, got {k}")
        if top_k <= 0:
            return []

        candidates: Dict[str, Dict[str, Any]] = {}

        # 1. Process BM25 rankings
        for rank_idx, r in enumerate(bm25_results):
            cid = str(r.get("chunk_id", ""))
            rank = rank_idx + 1
            rrf_contrib = 1.0 / float(k + rank)
            candidates[cid] = {
                "chunk_id": cid,
                "doc_name": r.get("doc_name", ""),
                "text": r.get("text", ""),
                "bm25_score": float(r.get("score", 0.0)),
                "bm25_rank": rank,
                "semantic_score": None,
                "semantic_rank": None,
                "rrf_score": rrf_contrib,
                "metadata": r.get("metadata", {}),
            }

        # 2. Process Semantic rankings
        for rank_idx, r in enumerate(semantic_results):
            cid = str(r.get("chunk_id", ""))
            rank = rank_idx + 1
            rrf_contrib = 1.0 / float(k + rank)
            if cid in candidates:
                candidates[cid]["semantic_score"] = float(r.get("score", 0.0))
                candidates[cid]["semantic_rank"] = rank
                candidates[cid]["rrf_score"] += rrf_contrib
                if not candidates[cid]["doc_name"]:
                    candidates[cid]["doc_name"] = r.get("doc_name", "")
                if not candidates[cid]["text"]:
                    candidates[cid]["text"] = r.get("text", "")
            else:
                candidates[cid] = {
                    "chunk_id": cid,
                    "doc_name": r.get("doc_name", ""),
                    "text": r.get("text", ""),
                    "bm25_score": None,
                    "bm25_rank": None,
                    "semantic_score": float(r.get("score", 0.0)),
                    "semantic_rank": rank,
                    "rrf_score": rrf_contrib,
                    "metadata": r.get("metadata", {}),
                }

        # 3. Create HybridResults
        scored_results: List[HybridResult] = []
        for cid, info in candidates.items():
            scored_results.append(
                HybridResult(
                    chunk_id=cid,
                    doc_name=info["doc_name"],
                    text=info["text"],
                    score=float(info["rrf_score"]),
                    bm25_score=info["bm25_score"],
                    bm25_rank=info["bm25_rank"],
                    semantic_score=info["semantic_score"],
                    semantic_rank=info["semantic_rank"],
                    normalized_bm25_score=None,
                    normalized_semantic_score=None,
                    fusion_strategy="rrf",
                    alpha=None,
                    rrf_k=k,
                    metadata=info["metadata"],
                )
            )

        # 4. Deterministic tie-breaking: score descending, chunk_id ascending
        scored_results.sort(key=lambda r: (-r.score, r.chunk_id))
        return scored_results[:top_k]
