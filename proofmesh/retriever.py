"""
ProofMesh Hybrid Retriever
--------------------------
Combines BM25 lexical ranking and dense vector representations for precision
evidence chunk extraction across local offline documents.
"""

import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

# Attempt to import rank_bm25; provide a self-contained fallback if not installed
try:
    from rank_bm25 import BM25Plus, BM25Okapi
    HAS_RANK_BM25 = True
except ImportError:
    BM25Plus = None
    BM25Okapi = None
    HAS_RANK_BM25 = False

try:
    from .dense_retrieval import SemanticRetriever, SemanticChunk
except (ImportError, ValueError):
    try:
        from proofmesh.dense_retrieval import SemanticRetriever, SemanticChunk  # type: ignore
    except ImportError:
        from dense_retrieval import SemanticRetriever, SemanticChunk  # type: ignore


class SimpleBM25Fallback:
    """
    Lightweight, self-contained BM25 fallback implementation when rank_bm25 is not installed.
    Ensures offline execution never crashes due to missing third-party packages.
    """

    def __init__(self, corpus_tokenized: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus_tokenized
        self.corpus_size = len(corpus_tokenized)
        self.avgdl = sum(len(doc) for doc in corpus_tokenized) / max(1, self.corpus_size)
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = {}
        for doc in corpus_tokenized:
            seen = set(doc)
            for token in seen:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

    def _idf(self, q: str) -> float:
        df = self.doc_freqs.get(q, 0)
        # Standard BM25 IDF formula
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


@dataclass
class DocumentChunk:
    """
    Standard ProofMesh evidence chunk representation.
    """
    chunk_id: str
    doc_name: str
    text: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "text": self.text,
            "score": self.score
        }
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


class HybridRetriever:
    """
    True Hybrid Retriever combining BM25 lexical ranking and dense vector representations
    for precision evidence chunk extraction across local offline documents.
    """

    def __init__(
        self,
        chunk_size_words: int = 60,
        chunk_overlap_words: int = 15,
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        dense_weight: float = 0.5,
        enable_dense: bool = True,
    ):
        """
        Initialize the HybridRetriever.

        Args:
            chunk_size_words: Words per chunk during document splitting.
            chunk_overlap_words: Overlapping words between adjacent chunks.
            dense_model_name: Sentence-transformers model name or 'mock'.
            device: Device for dense embeddings ('cpu' or 'auto').
            dense_weight: Weight alpha for dense scores vs lexical scores (0.0 = pure BM25, 1.0 = pure Dense).
            enable_dense: Whether to activate dense semantic retrieval alongside BM25.
        """
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.dense_weight = max(0.0, min(1.0, dense_weight))
        self.enable_dense = enable_dense

        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[Any] = None
        self.corpus_tokenized: List[List[str]] = []
        self.dense_retriever: Optional[SemanticRetriever] = None

        if self.enable_dense:
            self.dense_retriever = SemanticRetriever(
                model_name=dense_model_name,
                device=device,
                batch_size=32,
                allow_fallback=True,
            )

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r'\b\w+\b', text)]

    def add_document(self, doc_name: str, text: str) -> List[DocumentChunk]:
        """Splits document into chunks with structured IDs and indexes them."""
        words = text.split()
        if not words:
            return []

        doc_chunks: List[DocumentChunk] = []
        chunk_idx = len(self.chunks) + 1

        step = max(1, self.chunk_size_words - self.chunk_overlap_words)
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size_words]
            chunk_text = " ".join(chunk_words)
            chunk_id = f"chunk_{chunk_idx}"
            dc = DocumentChunk(chunk_id=chunk_id, doc_name=doc_name, text=chunk_text)
            doc_chunks.append(dc)
            chunk_idx += 1

        self.chunks.extend(doc_chunks)
        self._rebuild_index()

        # Update dense semantic index
        if self.dense_retriever:
            self.dense_retriever.add_chunks_directly([c.to_dict() for c in doc_chunks])

        return doc_chunks

    def add_chunks_directly(self, chunks: List[Dict[str, Any]]):
        """Add pre-chunked dictionary items."""
        added_chunks = []
        for c in chunks:
            dc = DocumentChunk(
                chunk_id=c.get("chunk_id", f"chunk_{len(self.chunks)+1}"),
                doc_name=c.get("doc_name", "manual_doc"),
                text=c.get("text", ""),
                metadata=c.get("metadata", {}),
            )
            self.chunks.append(dc)
            added_chunks.append(dc)

        self._rebuild_index()

        # Update dense semantic index
        if self.dense_retriever:
            self.dense_retriever.add_chunks_directly(chunks)

    def _rebuild_index(self):
        if not self.chunks:
            self.bm25 = None
            self.corpus_tokenized = []
            return

        self.corpus_tokenized = [self._tokenize(c.text) for c in self.chunks]
        if HAS_RANK_BM25:
            if BM25Plus is not None:
                self.bm25 = BM25Plus(self.corpus_tokenized)
            elif BM25Okapi is not None:
                self.bm25 = BM25Okapi(self.corpus_tokenized)
            else:
                self.bm25 = SimpleBM25Fallback(self.corpus_tokenized)
        else:
            self.bm25 = SimpleBM25Fallback(self.corpus_tokenized)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        alpha: Optional[float] = None,
        min_score: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most relevant evidence chunks combining BM25 lexical scoring
        and dense semantic vector similarity.

        Args:
            query: User search query.
            top_k: Number of candidate chunks to extract.
            alpha: Weight for dense semantic scoring vs lexical BM25 (0.0 to 1.0).
                   If None, uses self.dense_weight.
            min_score: Minimum relevance threshold to filter irrelevant chunks.

        Returns:
            List of result dictionaries matching ProofMesh output schema.
        """
        if not self.chunks:
            return []

        eff_alpha = self.dense_weight if alpha is None else max(0.0, min(1.0, alpha))

        # 1. Lexical BM25 Scoring
        bm25_scores = [0.0] * len(self.chunks)
        if self.bm25:
            tokenized_query = self._tokenize(query)
            if tokenized_query:
                raw_bm25 = self.bm25.get_scores(tokenized_query)
                max_val = float(max(raw_bm25)) if len(raw_bm25) > 0 else 0.0
                max_bm25 = max_val if max_val > 0 else 0.0
                if max_bm25 > 0:
                    bm25_scores = [float(s) / max_bm25 for s in raw_bm25]
                else:
                    bm25_scores = [0.0] * len(self.chunks)

        # 2. Dense Semantic Scoring
        dense_scores: Dict[str, float] = {}
        if self.enable_dense and self.dense_retriever and eff_alpha > 0.0:
            dense_results = self.dense_retriever.retrieve(query, top_k=len(self.chunks), min_score=None)
            for r in dense_results:
                dense_scores[r["chunk_id"]] = float(r["score"])

        # 3. Hybrid Score Fusion
        # hybrid_score = (1 - alpha) * bm25_norm + alpha * dense_score
        candidate_scores = []
        for idx, chunk in enumerate(self.chunks):
            lex_s = bm25_scores[idx] if idx < len(bm25_scores) else 0.0
            dense_s = dense_scores.get(chunk.chunk_id, 0.0)

            if not self.enable_dense or self.dense_retriever is None or eff_alpha <= 0.0:
                final_score = lex_s
            elif eff_alpha >= 1.0:
                final_score = dense_s
            else:
                final_score = (1.0 - eff_alpha) * lex_s + eff_alpha * dense_s

            candidate_scores.append((final_score, chunk.chunk_id, idx, float(lex_s), float(dense_s)))

        # Deterministic ranking: score descending, chunk_id ascending
        candidate_scores.sort(key=lambda x: (-x[0], x[1]))

        results = []
        for final_score, chunk_id, idx, lex_score, d_score in candidate_scores:
            if final_score > min_score:
                chunk = self.chunks[idx]
                item = {
                    "chunk_id": chunk.chunk_id,
                    "doc_name": chunk.doc_name,
                    "text": chunk.text,
                    "score": float(final_score),
                    "bm25_score": lex_score,
                    "dense_score": d_score,
                }
                if chunk.metadata:
                    item["metadata"] = dict(chunk.metadata)
                results.append(item)

            if len(results) >= top_k:
                break

        return results

    def clear(self):
        """Clear all lexical and dense indices."""
        self.chunks.clear()
        self.bm25 = None
        self.corpus_tokenized.clear()
        if self.dense_retriever:
            self.dense_retriever.clear()

    def stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics of the hybrid retriever."""
        stats_dict = {
            "indexed_chunks": len(self.chunks),
            "dense_enabled": self.enable_dense,
            "dense_weight": self.dense_weight,
            "has_rank_bm25_pkg": HAS_RANK_BM25,
        }
        if self.dense_retriever:
            stats_dict["dense_retriever"] = self.dense_retriever.stats()
        return stats_dict


if __name__ == "__main__":
    print("Testing HybridRetriever standalone...")
    retriever = HybridRetriever(dense_weight=0.5)
    sample_text = (
        "Project Hackingly is an offline evidence-bound RAG system designed for Intel CPU architecture. "
        "It achieves sub-second latency with strict citation verification and zero cloud dependencies. "
        "The hardware warranty is 24 months from the purchase date."
    )
    chunks = retriever.add_document("sample_doc.txt", sample_text)
    print(f"Indexed {len(chunks)} chunks.")
    
    query = "What is the warranty period for hardware?"
    results = retriever.retrieve(query, top_k=2)
    print(f"Query: '{query}' -> Retrieved {len(results)} chunks:")
    for r in results:
        print(f"  [{r['chunk_id']}] (score: {r['score']:.4f}) {r['text']}")
    print("Stats:", retriever.stats())
    print("Success: HybridRetriever executed cleanly.")
