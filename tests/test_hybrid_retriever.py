"""
Unit & Integration Tests for ProofMesh HybridRetriever (BM25 + Dense Semantic Retrieval).
"""

import unittest
from proofmesh.retriever import HybridRetriever, DocumentChunk


class TestHybridRetriever(unittest.TestCase):

    def setUp(self):
        self.retriever = HybridRetriever(
            chunk_size_words=60,
            chunk_overlap_words=15,
            dense_model_name="mock",
            dense_weight=0.5,
            enable_dense=True,
        )

    def test_add_document_and_chunking(self):
        text = "The quick brown fox jumps over the lazy dog. " * 15
        chunks = self.retriever.add_document("doc_fox", text)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].doc_name, "doc_fox")
        self.assertEqual(chunks[0].chunk_id, "chunk_1")
        self.assertEqual(len(self.retriever.chunks), len(chunks))

    def test_add_chunks_directly(self):
        direct = [
            {"chunk_id": "c1", "doc_name": "d1", "text": "Annual software license agreement terms."},
            {"chunk_id": "c2", "doc_name": "d2", "text": "Hardware warranty duration and coverage."},
        ]
        self.retriever.add_chunks_directly(direct)
        self.assertEqual(len(self.retriever.chunks), 2)
        self.assertEqual(self.retriever.chunks[0].chunk_id, "c1")
        self.assertEqual(self.retriever.chunks[1].chunk_id, "c2")

    def test_pure_lexical_retrieval(self):
        """When alpha=0.0, retriever operates in pure BM25 lexical mode."""
        chunks = [
            {"chunk_id": "c1", "doc_name": "d1", "text": "Kubernetes container orchestration on cloud."},
            {"chunk_id": "c2", "doc_name": "d2", "text": "Baking fresh sourdough bread in oven."},
        ]
        self.retriever.add_chunks_directly(chunks)
        results = self.retriever.retrieve("Kubernetes container", top_k=1, alpha=0.0)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "c1")

    def test_pure_dense_retrieval(self):
        """When alpha=1.0, retriever operates in pure dense semantic mode."""
        chunks = [
            {"chunk_id": "c1", "doc_name": "d1", "text": "Kubernetes container orchestration on cloud."},
            {"chunk_id": "c2", "doc_name": "d2", "text": "Baking fresh sourdough bread in oven."},
        ]
        self.retriever.add_chunks_directly(chunks)
        results = self.retriever.retrieve("Kubernetes container", top_k=1, alpha=1.0)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "c1")

    def test_hybrid_retrieval_fusion(self):
        """Default alpha=0.5 balances lexical and semantic scoring."""
        chunks = [
            {"chunk_id": "c_legal", "doc_name": "contract.pdf", "text": "The purchaser shall remit the outstanding invoice within thirty calendar days."},
            {"chunk_id": "c_tech", "doc_name": "tech.pdf", "text": "PostgreSQL database clustering and replication configuration."},
            {"chunk_id": "c_hr", "doc_name": "hr.pdf", "text": "Employees must submit vacation requests via the HR portal."},
        ]
        self.retriever.add_chunks_directly(chunks)
        results = self.retriever.retrieve("invoice payment terms", top_k=2, alpha=0.5)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["chunk_id"], "c_legal")
        self.assertIn("score", results[0])
        self.assertIsInstance(results[0]["score"], float)

    def test_clear_resets_indices(self):
        self.retriever.add_chunks_directly([{"chunk_id": "c1", "doc_name": "d1", "text": "test"}])
        self.assertEqual(len(self.retriever.chunks), 1)
        self.retriever.clear()
        self.assertEqual(len(self.retriever.chunks), 0)
        self.assertEqual(self.retriever.retrieve("test"), [])


if __name__ == "__main__":
    unittest.main()
