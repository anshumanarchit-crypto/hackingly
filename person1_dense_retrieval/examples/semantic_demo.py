"""
Semantic Retrieval Demonstration for ProofMesh (Person 1 Workstream).
Demonstrates semantic retrieval on a controlled legal/operational corpus,
showing retrieval based on conceptual meaning rather than exact word matches.
"""

import sys
import os

# Ensure module can be imported regardless of execution directory
current_dir = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.abspath(os.path.join(current_dir, "..")),
    os.path.abspath(os.path.join(current_dir, "..", "src")),
    os.path.abspath(os.path.join(current_dir, "..", "..")),
    os.path.abspath(os.path.join(current_dir, "..", "person1_dense_retrieval", "src")),
]
for p in possible_paths:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from dense_retrieval.semantic_retriever import SemanticRetriever
except ImportError:
    from proofmesh.dense_retrieval.semantic_retriever import SemanticRetriever  # type: ignore[import-not-found]


def run_demo():
    print("=" * 75)
    print("ProofMesh Dense Semantic Retrieval - Quality Demonstration")
    print("=" * 75)

    retriever = SemanticRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        batch_size=16,
        allow_fallback=True,
    )

    stats = retriever.stats()
    embedder_info = stats["embedder"]
    print(f"Active Backend:   {embedder_info['backend'].upper()}")
    print(f"Production Model: {embedder_info['model_name']}")
    print(f"Device:           {embedder_info['device']}")
    print(f"Vector Dimension: {embedder_info['dimension']}")
    if embedder_info["is_mock"]:
        print("NOTE: Running with deterministic mock backend for offline demonstration.")
        print("      For production neural embeddings, install sentence-transformers.")
    print("-" * 75)

    # Controlled Corpus
    corpus = [
        {
            "chunk_id": "chunk_1",
            "doc_name": "contract_terms.pdf",
            "text": "The purchaser shall remit the outstanding invoice within thirty calendar days.",
        },
        {
            "chunk_id": "chunk_2",
            "doc_name": "warranty_policy.pdf",
            "text": "The hardware warranty remains valid for twenty-four months from the original purchase date.",
        },
        {
            "chunk_id": "chunk_3",
            "doc_name": "employee_handbook.pdf",
            "text": "Employees must submit vacation requests through the HR portal.",
        },
    ]

    print(f"Indexing {len(corpus)} sample chunks into in-memory vector index...")
    retriever.add_chunks_directly(corpus)
    print("Indexing complete.")
    print("-" * 75)

    # Test Queries demonstrating conceptual equivalence
    queries = [
        {
            "query": "When does the customer need to pay?",
            "expected_id": "chunk_1",
            "concept": "Payment timeframe ('purchaser shall remit' vs 'customer need to pay')",
        },
        {
            "query": "How long is the device protected?",
            "expected_id": "chunk_2",
            "concept": "Coverage duration ('warranty remains valid 24 months' vs 'device protected')",
        },
        {
            "query": "Where do I request leave?",
            "expected_id": "chunk_3",
            "concept": "Leave portal ('vacation requests through HR portal' vs 'request leave')",
        },
    ]

    for idx, q_item in enumerate(queries, 1):
        query = q_item["query"]
        expected = q_item["expected_id"]
        concept = q_item["concept"]

        print(f"\n[Test Query {idx}] \"{query}\"")
        print(f"Target Concept: {concept}")

        results = retriever.retrieve(query, top_k=2)

        if not results:
            print("  No results returned.")
            continue

        top_match = results[0]
        match_status = "MATCH" if top_match["chunk_id"] == expected else "PARTIAL/ALTERNATIVE"

        print(f"  Top Match [{match_status}]: {top_match['chunk_id']} (Score: {top_match['score']:.4f})")
        print(f"  Source Doc:  {top_match['doc_name']}")
        print(f"  Chunk Text:  \"{top_match['text']}\"")

        print("  All Ranked Candidates:")
        for rank, res in enumerate(results, 1):
            print(f"    Rank {rank}: {res['chunk_id']} -> Score: {res['score']:.4f} ({res['doc_name']})")

    # Unrelated Query Demo
    unrelated_query = "What is the patient's blood type?"
    print(f"\n[Unrelated Query Test] \"{unrelated_query}\"")
    unrelated_results = retriever.retrieve(unrelated_query, top_k=3, min_score=0.50)
    print(f"  Results with min_score=0.50 filter: {len(unrelated_results)} chunks returned.")
    if len(unrelated_results) == 0:
        print("  [SUCCESS] Unrelated query cleanly rejected by confidence threshold filtering.")

    print("\n" + "=" * 75)
    print("Demonstration finished successfully.")
    print("=" * 75)


if __name__ == "__main__":
    run_demo()
