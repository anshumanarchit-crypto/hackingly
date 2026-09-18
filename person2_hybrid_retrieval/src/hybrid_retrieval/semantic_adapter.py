"""
Semantic Adapter Subsystem for ProofMesh (Person 2 Workstream).
Defines a clean protocol for consuming dense semantic retrieval results independently
of Person 1's internal architecture, alongside a deterministic Mock Semantic Retriever
for offline unit testing.

Integration Philosophy:
Person 2 depends strictly on the public retrieval contract:
    retrieve(query: str, top_k: int = 3, min_score: Optional[float] = None) -> List[Dict[str, Any]]
Where each result dict contains: {"chunk_id": str, "doc_name": str, "text": str, "score": float}.
"""

from typing import Protocol, runtime_checkable, List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@runtime_checkable
class SemanticRetrieverProtocol(Protocol):
    """
    Public retrieval contract required for any dense semantic retrieval backend.
    Enables zero-coupling between Person 2's fusion engine and Person 1's internal implementation.
    """

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k most semantically relevant chunks."""
        ...

    def add_document(self, doc_name: str, text: str) -> Any:
        """Indexes a raw document."""
        ...

    def add_chunks_directly(self, chunks: List[Dict[str, Any]]) -> Any:
        """Indexes pre-chunked dictionary items."""
        ...

    def clear(self) -> None:
        """Clears all indexed chunks and embeddings."""
        ...


class MockSemanticRetriever:
    """
    MOCK SEMANTIC PROVIDER:
    Deterministic semantic retrieval stub used strictly for unit testing the fusion layer
    when Person 1 is offline or developing in parallel.

    WARNING:
    This mock does NOT represent real neural embedding quality and should NEVER
    be cited as empirical evidence in retrieval quality benchmarks.
    """

    def __init__(self, predefined_results: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        self.predefined_results = predefined_results or {}
        self.chunks: List[Dict[str, Any]] = []

    def set_mock_results(self, query: str, results: List[Dict[str, Any]]) -> None:
        """Set fixed deterministic results for a specific query string."""
        self.predefined_results[query] = results

    def add_document(self, doc_name: str, text: str) -> List[Dict[str, Any]]:
        """Mock document addition."""
        words = text.split()
        created = []
        chunk_idx = len(self.chunks) + 1
        for i in range(0, len(words), 45):
            chunk_text = " ".join(words[i : i + 60])
            item = {
                "chunk_id": f"chunk_{chunk_idx}",
                "doc_name": doc_name,
                "text": chunk_text,
            }
            self.chunks.append(item)
            created.append(item)
            chunk_idx += 1
        return created

    def add_chunks_directly(self, chunks: List[Dict[str, Any]]) -> None:
        """Mock direct chunk addition."""
        self.chunks.extend(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns mock semantic retrieval results.
        If a canned query response exists, returns that.
        Otherwise, returns deterministic scores over indexed chunks based on keyword matching.
        """
        if query in self.predefined_results:
            results = self.predefined_results[query]
            if min_score is not None:
                results = [r for r in results if r.get("score", 0.0) >= min_score]
            return results[:top_k]

        if not self.chunks:
            return []

        # Synthetic deterministic score generation for unmocked queries
        q_words = set(query.lower().split())
        scored = []
        for idx, chunk in enumerate(self.chunks):
            c_text = chunk.get("text", "").lower()
            overlap = sum(1 for w in q_words if w in c_text)
            # Produce mock cosine-like score between 0.10 and 0.95
            score = 0.50 + 0.10 * overlap - 0.02 * idx
            score = max(0.05, min(0.95, score))
            if min_score is not None and score < min_score:
                continue
            scored.append((score, chunk["chunk_id"], chunk))

        scored.sort(key=lambda x: (-x[0], x[1]))
        return [
            {
                "chunk_id": item["chunk_id"],
                "doc_name": item.get("doc_name", "mock_doc"),
                "text": item.get("text", ""),
                "score": round(score, 4),
            }
            for score, _, item in scored[:top_k]
        ]

    def clear(self) -> None:
        self.chunks.clear()
        self.predefined_results.clear()


class Person1SemanticAdapter:
    """
    Adapter wrapping Person 1's real SemanticRetriever instance.
    Guarantees strict adherence to SemanticRetrieverProtocol without modifying
    or exposing Person 1 internals.
    """

    def __init__(self, real_retriever: Optional[Any] = None):
        self.retriever = real_retriever
        self._is_available = real_retriever is not None

    @property
    def is_available(self) -> bool:
        """Indicates whether a real Person 1 retriever is active."""
        return self._is_available

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Forward retrieval to Person 1's retriever and standardize schema."""
        if not self._is_available or self.retriever is None:
            raise RuntimeError(
                "Person 1 SemanticRetriever is not loaded. "
                "Use MockSemanticRetriever for standalone testing."
            )

        raw_results = self.retriever.retrieve(query=query, top_k=top_k, min_score=min_score)
        standardized = []
        for r in raw_results:
            standardized.append({
                "chunk_id": str(r.get("chunk_id", "")),
                "doc_name": str(r.get("doc_name", "")),
                "text": str(r.get("text", "")),
                "score": float(r.get("score", 0.0)),
                "metadata": r.get("metadata", {}),
            })
        return standardized

    def add_document(self, doc_name: str, text: str) -> Any:
        if self.retriever is not None and hasattr(self.retriever, "add_document"):
            return self.retriever.add_document(doc_name, text)

    def add_chunks_directly(self, chunks: List[Dict[str, Any]]) -> Any:
        if self.retriever is not None and hasattr(self.retriever, "add_chunks_directly"):
            return self.retriever.add_chunks_directly(chunks)

    def clear(self) -> None:
        if self.retriever is not None and hasattr(self.retriever, "clear"):
            self.retriever.clear()
