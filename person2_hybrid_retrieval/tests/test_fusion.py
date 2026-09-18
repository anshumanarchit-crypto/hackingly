"""
Unit tests for Score Normalization & Fusion Engine (Person 2 Workstream).
Tests:
1. Min-max normalization
2. Constant-score normalization (division-by-zero guard)
3. Rank normalization
4. Z-score normalization
5. Weighted fusion (alpha=0.5, alpha=0.0, alpha=1.0)
6. Invalid alpha validation
7. Reciprocal Rank Fusion (RRF)
8. Different RRF k values and invalid k validation
9. Missing BM25 results (semantic only)
10. Missing Semantic results (BM25 only)
11. Candidate union and deduplication
12. Deterministic tie-breaking
13. Explainability metadata fields
14. Top-k handling
"""

import os
import sys
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from hybrid_retrieval.score_normalizer import ScoreNormalizer
from hybrid_retrieval.fusion import FusionEngine


# -------------------------------------------------------------
# Score Normalizer Tests
# -------------------------------------------------------------

def test_min_max_normalization_standard():
    scores = [10.0, 20.0, 30.0]
    norm = ScoreNormalizer.min_max_normalize(scores)
    assert norm == [0.0, 0.5, 1.0]


def test_min_max_normalization_constant_scores():
    # All identical non-zero scores -> assign 1.0
    scores = [15.0, 15.0, 15.0]
    norm = ScoreNormalizer.min_max_normalize(scores)
    assert norm == [1.0, 1.0, 1.0]

    # All zeros
    zero_scores = [0.0, 0.0]
    norm_zero = ScoreNormalizer.min_max_normalize(zero_scores)
    assert norm_zero == [0.0, 0.0]


def test_min_max_normalization_single_and_empty():
    assert ScoreNormalizer.min_max_normalize([]) == []
    assert ScoreNormalizer.min_max_normalize([5.0]) == [1.0]
    assert ScoreNormalizer.min_max_normalize([0.0]) == [0.0]


def test_rank_normalization():
    scores = [100.0, 50.0, 75.0]  # Ranks: 100 -> 1, 75 -> 2, 50 -> 3
    norm = ScoreNormalizer.rank_normalize(scores)
    # N = 3
    # 100.0 -> rank 1 -> (3 - 1 + 1)/3 = 3/3 = 1.0
    # 75.0  -> rank 2 -> (3 - 2 + 1)/3 = 2/3 = 0.6667
    # 50.0  -> rank 3 -> (3 - 3 + 1)/3 = 1/3 = 0.3333
    assert pytest.approx(norm[0], 0.001) == 1.0
    assert pytest.approx(norm[1], 0.001) == 1.0 / 3.0
    assert pytest.approx(norm[2], 0.001) == 2.0 / 3.0


def test_z_score_normalization():
    scores = [10.0, 20.0, 30.0]
    z = ScoreNormalizer.z_score_normalize(scores)
    assert len(z) == 3
    assert pytest.approx(sum(z), abs=1e-6) == 0.0  # Mean zero
    # Constant variance
    assert ScoreNormalizer.z_score_normalize([5.0, 5.0]) == [0.0, 0.0]


# -------------------------------------------------------------
# Fusion Engine Tests
# -------------------------------------------------------------

def test_weighted_fusion_alpha_half():
    engine = FusionEngine()
    bm25 = [
        {"chunk_id": "chunk_1", "doc_name": "a", "text": "...", "score": 20.0},
        {"chunk_id": "chunk_2", "doc_name": "b", "text": "...", "score": 10.0},
    ]
    semantic = [
        {"chunk_id": "chunk_1", "doc_name": "a", "text": "...", "score": 0.80},
        {"chunk_id": "chunk_2", "doc_name": "b", "text": "...", "score": 0.40},
    ]
    # BM25: chunk_1 norm=1.0, chunk_2 norm=0.0
    # Semantic: chunk_1 norm=1.0, chunk_2 norm=0.0
    results = engine.fuse(bm25, semantic, strategy="weighted", alpha=0.5, top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "chunk_1"
    assert pytest.approx(results[0].score, 0.001) == 1.0
    assert results[1].chunk_id == "chunk_2"
    assert pytest.approx(results[1].score, 0.001) == 0.0


def test_weighted_fusion_alpha_extremes():
    engine = FusionEngine()
    bm25 = [
        {"chunk_id": "chunk_1", "doc_name": "a", "text": "...", "score": 20.0},
        {"chunk_id": "chunk_2", "doc_name": "b", "text": "...", "score": 10.0},
    ]
    semantic = [
        {"chunk_id": "chunk_2", "doc_name": "b", "text": "...", "score": 0.90},
        {"chunk_id": "chunk_1", "doc_name": "a", "text": "...", "score": 0.10},
    ]

    # alpha = 1.0 (pure BM25): chunk_1 should rank #1
    res_bm25 = engine.fuse(bm25, semantic, strategy="weighted", alpha=1.0, top_k=2)
    assert res_bm25[0].chunk_id == "chunk_1"

    # alpha = 0.0 (pure Semantic): chunk_2 should rank #1
    res_sem = engine.fuse(bm25, semantic, strategy="weighted", alpha=0.0, top_k=2)
    assert res_sem[0].chunk_id == "chunk_2"


def test_invalid_alpha_raises_error():
    engine = FusionEngine()
    with pytest.raises(ValueError):
        engine.fuse([], [], strategy="weighted", alpha=-0.1)
    with pytest.raises(ValueError):
        engine.fuse([], [], strategy="weighted", alpha=1.1)


def test_rrf_scoring_and_k():
    engine = FusionEngine()
    bm25 = [
        {"chunk_id": "chunk_1", "doc_name": "a", "text": "...", "score": 10.0},  # rank 1
        {"chunk_id": "chunk_2", "doc_name": "b", "text": "...", "score": 5.0},   # rank 2
    ]
    semantic = [
        {"chunk_id": "chunk_2", "doc_name": "b", "text": "...", "score": 0.9},   # rank 1
        {"chunk_id": "chunk_1", "doc_name": "a", "text": "...", "score": 0.7},   # rank 2
    ]

    # With k=60:
    # chunk_1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    # chunk_2: 1/(60+2) + 1/(60+1) = identical score!
    res_60 = engine.fuse(bm25, semantic, strategy="rrf", rrf_k=60, top_k=2)
    assert len(res_60) == 2
    assert pytest.approx(res_60[0].score, 1e-4) == 0.032522
    assert pytest.approx(res_60[1].score, 1e-4) == 0.032522
    # Tie break by chunk_id ascending: chunk_1 before chunk_2
    assert res_60[0].chunk_id == "chunk_1"
    assert res_60[1].chunk_id == "chunk_2"


def test_invalid_rrf_k_raises_error():
    engine = FusionEngine()
    with pytest.raises(ValueError):
        engine.fuse([], [], strategy="rrf", rrf_k=0)
    with pytest.raises(ValueError):
        engine.fuse([], [], strategy="rrf", rrf_k=-10)


def test_missing_source_results_candidate_union():
    engine = FusionEngine()
    # chunk_1 only in BM25, chunk_2 only in Semantic
    bm25 = [{"chunk_id": "chunk_1", "doc_name": "a", "text": "Lexical item", "score": 10.0}]
    semantic = [{"chunk_id": "chunk_2", "doc_name": "b", "text": "Semantic item", "score": 0.85}]

    res = engine.fuse(bm25, semantic, strategy="weighted", alpha=0.5, top_k=5)
    assert len(res) == 2
    cids = {r.chunk_id for r in res}
    assert cids == {"chunk_1", "chunk_2"}

    # Verify missing fields are handled safely
    for r in res:
        if r.chunk_id == "chunk_1":
            assert r.bm25_rank == 1
            assert r.semantic_rank is None
        elif r.chunk_id == "chunk_2":
            assert r.bm25_rank is None
            assert r.semantic_rank == 1


def test_deduplication_never_outputs_duplicate_chunks():
    engine = FusionEngine()
    bm25 = [
        {"chunk_id": "chunk_1", "doc_name": "doc", "text": "Shared content", "score": 10.0},
        {"chunk_id": "chunk_2", "doc_name": "doc", "text": "Unique BM25", "score": 5.0},
    ]
    semantic = [
        {"chunk_id": "chunk_1", "doc_name": "doc", "text": "Shared content", "score": 0.90},
        {"chunk_id": "chunk_3", "doc_name": "doc", "text": "Unique Sem", "score": 0.70},
    ]

    res = engine.fuse(bm25, semantic, strategy="rrf", top_k=10)
    cids = [r.chunk_id for r in res]
    # Length should be exactly 3 (chunk_1, chunk_2, chunk_3)
    assert len(cids) == 3
    assert len(set(cids)) == 3
    # chunk_1 was in both, so it must have highest RRF score
    assert res[0].chunk_id == "chunk_1"


def test_explainability_fields():
    engine = FusionEngine()
    bm25 = [{"chunk_id": "chunk_1", "doc_name": "doc", "text": "Sample text", "score": 12.5}]
    semantic = [{"chunk_id": "chunk_1", "doc_name": "doc", "text": "Sample text", "score": 0.88}]

    res = engine.fuse(bm25, semantic, strategy="weighted", alpha=0.6, top_k=1)
    item = res[0]
    res_dict = item.to_dict(include_explainability=True)

    assert res_dict["bm25_score"] == 12.5
    assert res_dict["bm25_rank"] == 1
    assert res_dict["semantic_score"] == 0.88
    assert res_dict["semantic_rank"] == 1
    assert res_dict["fusion_strategy"] == "weighted"
    assert res_dict["alpha"] == 0.6
    assert "norm:" in item.explain()
