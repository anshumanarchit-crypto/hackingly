"""
Standalone BM25 Lexical Retriever for ProofMesh (Person 2 Workstream).
Implements Okapi BM25 scoring with exact 60-word chunking and 15-word overlap,
matching the ProofMesh document processing specification.

Dependencies:
- Uses rank_bm25.BM25Okapi when available.
- Features a self-contained Okapi BM25 fallback for resilience in offline / zero-dependency environments.
- Clearly reports backend status ('rank_bm25' vs 'fallback').
- Retains raw unnormalized BM25 scores for downstream score normalization and fusion.
"""

import math
import re
import logging
from typing import List, Dict, Any, Optional

try:
    from rank_bm25 import BM25Plus, BM25Okapi
    HAS_RANK_BM25 = True
except ImportError:
    BM25Plus = None
    BM25Okapi = None
    HAS_RANK_BM25 = False

from .models import HybridChunk

logger = logging.getLogger(__name__)


class SimpleBM25Fallback:
    """
    Standard Okapi BM25 implementation for resilience when rank_bm25 is not installed.
    Matches standard Okapi BM25 formulas:
        IDF(q) = ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1.0)
        Score(D, Q) = sum(IDF(q) * (tf(q, D) * (k1 + 1)) / (tf(q, D) + k1 * (1 - b + b * (|D| / avgdl))))
    """

    def __init__(self, corpus_tokenized: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus_tokenized
        self.corpus_size = len(corpus_tokenized)
        self.avgdl = sum(len(doc) for doc in corpus_tokenized) / max(1, self.corpus_size)
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = {}
        for doc in corpus_tokenized:
            for token in set(doc):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

    def _idf(self, q: str) -> float:
        df = self.doc_freqs.get(q, 0)
        return math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def get_scores(self, tokenized_query: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for idx, doc in enumerate(self.corpus):
            doc_len = len(doc)
            freqs: Dict[str, int] = {}
            for t in doc:
                freqs[t] = freqs.get(t, 0) + 1

            for q in tokenized_query:
                if q not in freqs:
                    continue
                tf = freqs[q]
                idf = self._idf(q)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1e-6, self.avgdl)))
                scores[idx] += idf * (numerator / denominator)
        return scores


class BM25Retriever:
    """
    Standalone BM25 Lexical Retriever with 60/15 chunking, raw score preservation,
    and deterministic ranking.
    """

    def __init__(
        self,
        chunk_size_words: int = 60,
        chunk_overlap_words: int = 15,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.k1 = k1
        self.b = b

        self.chunks: List[HybridChunk] = []
        self.bm25: Optional[Any] = None
        self.corpus_tokenized: List[List[str]] = []

        # Backend identification
        self.backend = "rank_bm25" if (HAS_RANK_BM25 and BM25Okapi is not None) else "fallback"

    @property
    def backend_name(self) -> str:
        """Explicitly reports the active BM25 backend."""
        return self.backend

    def _tokenize(self, text: str) -> List[str]:
        """Lowercases and extracts alphanumeric token sequences."""
        return [w.lower() for w in re.findall(r'\b\w+\b', text)]

    def add_document(self, doc_name: str, text: str) -> List[Dict[str, Any]]:
        """
        Splits text into chunks of chunk_size_words with chunk_overlap_words overlap,
        indexes them, and returns created chunks.
        """
        words = text.split()
        if not words:
            return []

        created_chunks: List[HybridChunk] = []
        chunk_idx = len(self.chunks) + 1
        step = max(1, self.chunk_size_words - self.chunk_overlap_words)

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size_words]
            chunk_text = " ".join(chunk_words)
            chunk_id = f"chunk_{chunk_idx}"
            hc = HybridChunk(
                chunk_id=chunk_id,
                doc_name=doc_name,
                text=chunk_text,
                metadata={"start_word": i, "end_word": i + len(chunk_words)},
            )
            created_chunks.append(hc)
            chunk_idx += 1

        self.chunks.extend(created_chunks)
        self._rebuild_index()
        return [c.to_dict() for c in created_chunks]

    def add_chunks_directly(self, chunks: List[Dict[str, Any]]) -> None:
        """Adds pre-formed chunk dictionaries directly to the retriever."""
        for c in chunks:
            hc = HybridChunk(
                chunk_id=c.get("chunk_id", f"chunk_{len(self.chunks)+1}"),
                doc_name=c.get("doc_name", "direct_doc"),
                text=c.get("text", ""),
                metadata=c.get("metadata", {}),
            )
            self.chunks.append(hc)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Tokenizes corpus and rebuilds the BM25 index."""
        if not self.chunks:
            self.bm25 = None
            self.corpus_tokenized = []
            return

        self.corpus_tokenized = [self._tokenize(c.text) for c in self.chunks]
        if self.backend == "rank_bm25":
            if BM25Plus is not None:
                self.bm25 = BM25Plus(self.corpus_tokenized, k1=self.k1, b=self.b)
            elif BM25Okapi is not None:
                self.bm25 = BM25Okapi(self.corpus_tokenized, k1=self.k1, b=self.b)
            else:
                self.bm25 = SimpleBM25Fallback(self.corpus_tokenized, k1=self.k1, b=self.b)
        else:
            self.bm25 = SimpleBM25Fallback(self.corpus_tokenized, k1=self.k1, b=self.b)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most lexically relevant chunks for the given query.

        Args:
            query: Query text.
            top_k: Maximum number of results to return.
            min_score: Minimum raw score threshold to include.

        Returns:
            List of result dictionaries with raw BM25 scores:
            [
                {
                    "chunk_id": "chunk_1",
                    "doc_name": "...",
                    "text": "...",
                    "score": <raw_bm25_score>,
                    "metadata": {...}
                }
            ]
        """
        if not self.chunks or not self.bm25:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        raw_scores = self.bm25.get_scores(tokenized_query)

        # Build candidate list with (score, chunk_id, index) for deterministic tie-breaking
        scored_candidates = []
        for idx, score in enumerate(raw_scores):
            chunk = self.chunks[idx]
            if min_score is not None and score < min_score:
                continue
            scored_candidates.append((float(score), chunk.chunk_id, idx))

        # Deterministic sorting: score descending, tie-break by chunk_id ascending
        scored_candidates.sort(key=lambda x: (-x[0], x[1]))

        results: List[Dict[str, Any]] = []
        for score, chunk_id, idx in scored_candidates[:top_k]:
            chunk = self.chunks[idx]
            res_dict: Dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "doc_name": chunk.doc_name,
                "text": chunk.text,
                "score": float(score),
            }
            if chunk.metadata:
                res_dict["metadata"] = dict(chunk.metadata)
            results.append(res_dict)

        return results

    def clear(self) -> None:
        """Clears all indexed chunks and the BM25 model."""
        self.chunks.clear()
        self.bm25 = None
        self.corpus_tokenized.clear()

    def stats(self) -> Dict[str, Any]:
        """Returns diagnostic statistics for the retriever."""
        return {
            "indexed_chunks": len(self.chunks),
            "backend": self.backend,
            "chunk_size_words": self.chunk_size_words,
            "chunk_overlap_words": self.chunk_overlap_words,
            "k1": self.k1,
            "b": self.b,
        }
