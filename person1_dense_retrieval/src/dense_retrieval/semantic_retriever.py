"""
SemanticRetriever: In-memory dense semantic vector retrieval subsystem.
Operates fully offline, performing local batch embedding, vectorized cosine similarity,
and deterministic ranking matching the ProofMesh retrieval data contracts.
"""

from typing import List, Dict, Any, Optional, Union, Sequence
import numpy as np

try:
    from .models import SemanticChunk, ChunkRecord
    from .embeddings import SemanticEmbedder
except (ImportError, ValueError):
    try:
        from proofmesh.dense_retrieval.models import SemanticChunk, ChunkRecord  # type: ignore
        from proofmesh.dense_retrieval.embeddings import SemanticEmbedder  # type: ignore
    except ImportError:
        from dense_retrieval.models import SemanticChunk, ChunkRecord  # type: ignore
        from dense_retrieval.embeddings import SemanticEmbedder  # type: ignore


class SemanticRetriever:
    """
    Independent dense semantic retriever using vector representations.
    Designed for zero merge conflicts with existing ProofMesh BM25 retrieval.
    """

    def __init__(
        self,
        model_name: str = SemanticEmbedder.DEFAULT_MODEL_NAME,
        device: str = "cpu",
        batch_size: int = 32,
        min_score: Optional[float] = None,
        chunk_size_words: int = 60,
        chunk_overlap_words: int = 15,
        embedder: Optional[SemanticEmbedder] = None,
        allow_fallback: bool = True,
    ):
        """
        Initialize the semantic retriever.

        Args:
            model_name: Sentence-transformers model name or 'mock'.
            device: 'cpu' or 'auto'.
            batch_size: Default batch size for document encoding.
            min_score: Optional default similarity threshold for filtering.
            chunk_size_words: Number of words per chunk for add_document().
            chunk_overlap_words: Word overlap between consecutive chunks.
            embedder: Optional pre-configured SemanticEmbedder instance.
            allow_fallback: Allow fallback to deterministic mock backend if dependencies are missing.
        """
        self.chunk_size_words = max(1, chunk_size_words)
        self.chunk_overlap_words = max(0, chunk_overlap_words)
        self.batch_size = max(1, batch_size)
        self.default_min_score = min_score

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = SemanticEmbedder(
                model_name=model_name,
                device=device,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                allow_fallback=allow_fallback,
            )

        self.chunks: List[SemanticChunk] = []
        self._embeddings: Optional[np.ndarray] = None  # Shape: (N, dimension)

    def add_document(self, doc_name: str, text: str) -> List[SemanticChunk]:
        """
        Splits document text into overlapping word windows matching ProofMesh conventions,
        computes dense embeddings in batch, and stores them in the in-memory index.

        Args:
            doc_name: Document identifier/title.
            text: Raw document text content.

        Returns:
            List of newly created SemanticChunk objects.
        """
        words = text.split()
        if not words:
            return []

        doc_chunks: List[SemanticChunk] = []
        chunk_idx = len(self.chunks) + 1
        step = max(1, self.chunk_size_words - self.chunk_overlap_words)

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size_words]
            chunk_text = " ".join(chunk_words)
            chunk_id = f"chunk_{chunk_idx}"
            chunk = SemanticChunk(
                chunk_id=chunk_id,
                doc_name=doc_name,
                text=chunk_text,
                score=0.0,
            )
            doc_chunks.append(chunk)
            chunk_idx += 1

        self._index_chunks(doc_chunks)
        return doc_chunks

    def add_chunks_directly(
        self,
        chunks: Sequence[Union[Dict[str, Any], SemanticChunk, ChunkRecord]],
    ) -> None:
        """
        Index pre-chunked items directly without re-splitting. Accepts SemanticChunk,
        ChunkRecord, or standard dictionaries.

        Preserves existing chunk IDs, document names, and text verbatim.
        """
        if not chunks:
            return

        normalized_chunks: List[SemanticChunk] = []
        current_count = len(self.chunks)

        for i, c in enumerate(chunks):
            if isinstance(c, dict):
                normalized_chunks.append(
                    SemanticChunk(
                        chunk_id=c.get("chunk_id", f"chunk_{current_count + i + 1}"),
                        doc_name=c.get("doc_name", c.get("source", "manual_doc")),
                        text=c.get("text", ""),
                        score=0.0,
                        metadata=dict(c.get("metadata", {})),
                    )
                )
            elif hasattr(c, "to_semantic_chunk"):
                to_chunk_fn = getattr(c, "to_semantic_chunk")
                normalized_chunks.append(to_chunk_fn())
            elif hasattr(c, "chunk_id") and hasattr(c, "text"):
                chunk_id_val = str(getattr(c, "chunk_id"))
                doc_name_val = str(getattr(c, "doc_name", getattr(c, "source", "manual_doc")))
                text_val = str(getattr(c, "text"))
                meta_val = dict(getattr(c, "metadata", {}))
                normalized_chunks.append(
                    SemanticChunk(
                        chunk_id=chunk_id_val,
                        doc_name=doc_name_val,
                        text=text_val,
                        score=0.0,
                        metadata=meta_val,
                    )
                )
            else:
                raise TypeError(f"Unsupported chunk type: {type(c)}")

        self._index_chunks(normalized_chunks)

    def _index_chunks(self, new_chunks: List[SemanticChunk]) -> None:
        """Batch-encode new chunks and append to in-memory embedding matrix."""
        if not new_chunks:
            return

        texts = [chunk.text for chunk in new_chunks]
        new_embeddings = self.embedder.encode_texts(
            texts,
            batch_size=self.batch_size,
            normalize=True,
        )

        if self._embeddings is None or len(self.chunks) == 0:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        self.chunks.extend(new_chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most semantically relevant chunks for the given query.

        Args:
            query: Query string.
            top_k: Maximum number of results to return.
            min_score: Similarity threshold filter. Results with score < min_score are omitted.

        Returns:
            List of result dictionaries matching ProofMesh output schema:
            [{
                "chunk_id": "chunk_1",
                "doc_name": "...",
                "text": "...",
                "score": 0.85
            }]
        """
        # Validate inputs
        if not isinstance(top_k, int):
            raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}")
        if top_k <= 0:
            return []

        if not query or not query.strip():
            return []

        if not self.chunks or self._embeddings is None or len(self._embeddings) == 0:
            return []

        embeddings = self._embeddings
        # Vectorized query encoding
        q_vec = self.embedder.encode_text(query, normalize=True)

        # Vectorized cosine similarity: document_vectors @ query_vector
        # Shape: (N,)
        raw_scores = np.dot(embeddings, q_vec)

        # Build candidate tuples: (score, chunk_id, index)
        # Primary sort: score descending (-score)
        # Secondary tie-breaker: chunk_id ascending
        candidates = []
        for idx in range(len(self.chunks)):
            sc = float(raw_scores[idx])
            candidates.append((sc, self.chunks[idx].chunk_id, idx))

        candidates.sort(key=lambda item: (-item[0], item[1]))

        # Threshold filtering
        threshold = self.default_min_score if min_score is None else min_score

        results: List[Dict[str, Any]] = []
        for score, chunk_id, idx in candidates:
            if threshold is not None and score < threshold:
                continue

            chunk = self.chunks[idx]
            result_item = {
                "chunk_id": chunk.chunk_id,
                "doc_name": chunk.doc_name,
                "text": chunk.text,
                "score": float(score),
            }
            if chunk.metadata:
                result_item["metadata"] = dict(chunk.metadata)

            results.append(result_item)

            if len(results) >= top_k:
                break

        return results

    # API aliases for integration flexibility
    search = retrieve
    index = add_chunks_directly

    def clear(self) -> None:
        """Reset the in-memory vector index and chunk catalog."""
        self.chunks.clear()
        self._embeddings = None

    def stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics of the retriever and embedding subsystem."""
        embedder_info = self.embedder.stats()
        return {
            "indexed_chunks": len(self.chunks),
            "embedding_matrix_shape": (
                list(self._embeddings.shape) if self._embeddings is not None else [0, embedder_info["dimension"]]
            ),
            "batch_size": self.batch_size,
            "default_min_score": self.default_min_score,
            "embedder": embedder_info,
        }
