"""
Score Normalization subsystem for ProofMesh Hybrid Retrieval (Person 2 Workstream).
Translates disparate score spaces (unbounded BM25 scores vs bounded/cosine dense scores)
into comparable numerical scales prior to hybrid fusion.

Core Principle:
Raw score addition (bm25_score + semantic_score) is strictly mathematically invalid
because BM25 scores are unbounded positive frequencies (e.g. 0 to 25+), while dense
similarity scores are bounded cosine metrics (e.g. -1.0 to 1.0 or 0.0 to 1.0).
Score normalization maps them to commensurate [0.0, 1.0] spaces.
"""

import math
from typing import List


class ScoreNormalizer:
    """
    Independent, reusable score normalization engine supporting Min-Max,
    Rank-Percentile, and Z-Score normalization.
    """

    @staticmethod
    def min_max_normalize(scores: List[float]) -> List[float]:
        """
        Normalize a list of scores to the [0.0, 1.0] range using linear Min-Max scaling:

            normalized = (score - min_score) / (max_score - min_score)

        Edge Cases & Constant-Score Handling:
        - Empty list: Returns []
        - Single score: Returns [1.0] if score > 0 else [0.0]
        - max_score == min_score (constant scores across all candidates):
          Division by zero is guarded against. When all candidate items share identical scores:
          - If max_score > 0: All items are assigned 1.0 (indicating equal non-zero relevance).
          - If max_score <= 0: All items are assigned 0.0.

        Args:
            scores: List of raw floating point scores.

        Returns:
            List of normalized scores in [0.0, 1.0].
        """
        if not scores:
            return []

        if len(scores) == 1:
            return [1.0 if scores[0] > 0 else 0.0]

        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        if math.isclose(score_range, 0.0, abs_tol=1e-12):
            # Constant score case: avoid division by zero
            return [1.0 if max_score > 0 else 0.0 for _ in scores]

        return [(s - min_score) / score_range for s in scores]

    @staticmethod
    def rank_normalize(scores: List[float], higher_is_better: bool = True) -> List[float]:
        """
        Normalize scores using rank-percentile position:

            norm_rank = (N - rank + 1) / N

        where rank 1 corresponds to the best score (assigned 1.0) and rank N
        corresponds to the worst score (assigned 1/N).

        Tie-breaking:
        Candidates with identical scores receive identical average ranks.

        Args:
            scores: List of raw scores.
            higher_is_better: If True, highest score is rank 1.

        Returns:
            List of rank-normalized scores in (0.0, 1.0].
        """
        n = len(scores)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        # Pair scores with original indices
        indexed_scores = list(enumerate(scores))
        # Sort by score descending (if higher_is_better)
        indexed_scores.sort(key=lambda x: x[1], reverse=higher_is_better)

        # Assign ranks handling ties
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and math.isclose(indexed_scores[j][1], indexed_scores[i][1], abs_tol=1e-12):
                j += 1
            # Ranks for positions i to j-1 are 1-based (i+1 to j)
            avg_rank = (sum(range(i + 1, j + 1))) / (j - i)
            for k in range(i, j):
                orig_idx = indexed_scores[k][0]
                ranks[orig_idx] = avg_rank
            i = j

        return [(n - r + 1.0) / float(n) for r in ranks]

    @staticmethod
    def z_score_normalize(scores: List[float]) -> List[float]:
        """
        Standardize scores to zero-mean and unit variance:

            z = (score - mean) / std_dev

        Edge Cases:
        - Empty list: Returns []
        - Single score or zero variance: Returns [0.0] * N

        Args:
            scores: List of raw scores.

        Returns:
            List of z-standardized scores.
        """
        n = len(scores)
        if n <= 1:
            return [0.0] * n

        mean = sum(scores) / float(n)
        variance = sum((s - mean) ** 2 for s in scores) / float(n)
        std_dev = math.sqrt(variance)

        if math.isclose(std_dev, 0.0, abs_tol=1e-12):
            return [0.0] * n

        return [(s - mean) / std_dev for s in scores]
