"""
ProofMesh Hybrid Retrieval Demonstration (Person 2 Workstream).
Demonstrates how BM25 lexical ranking and dense semantic ranking capture
complementary signals, and how hybrid fusion produces more robust top-k evidence.

Corpus:
- chunk_1: "The purchaser shall remit the outstanding invoice within thirty calendar days."
- chunk_2: "The hardware warranty remains valid for twenty-four months from the original purchase date."
- chunk_3: "Employees must submit vacation requests through the HR portal."
- chunk_4: "The invoice reference number must be included in all payment correspondence."

Queries:
1. "When does the customer need to pay?" (Paraphrase/Conceptual query)
2. "How long is the device protected?" (Conceptual coverage query)
3. "What is the invoice reference requirement?" (Exact terminology query)
"""

import sys
import os

# Set up local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Connect to Person 1 real semantic retriever if present, otherwise use mock
person1_src = os.path.abspath(os.path.join(current_dir, "..", "..", "person1_dense_retrieval", "src"))
if os.path.isdir(person1_src) and person1_src not in sys.path:
    sys.path.insert(0, person1_src)

from hybrid_retrieval.bm25_retriever import BM25Retriever
from hybrid_retrieval.hybrid_retriever import HybridRetriever
from hybrid_retrieval.semantic_adapter import Person1SemanticAdapter, MockSemanticRetriever

try:
    from dense_retrieval.semantic_retriever import SemanticRetriever as P1SemanticRetriever
    p1_instance = P1SemanticRetriever(allow_fallback=True)
    semantic_provider = Person1SemanticAdapter(p1_instance)
    sem_source_desc = "Person 1 Dense Semantic Retriever (Integrated via Adapter)"
except Exception:
    semantic_provider = MockSemanticRetriever()
    sem_source_desc = "Deterministic Mock Semantic Provider (Fallback)"


def run_demo():
    print("=" * 80)
    print("ProofMesh Hybrid Retrieval Demonstration (Person 2 Workstream)")
    print("=" * 80)
    print(f"Semantic Backend: {sem_source_desc}")

    bm25 = BM25Retriever()
    print(f"BM25 Backend:     {bm25.backend_name.upper()}")
    print("-" * 80)

    # 1. Index the prescribed 4-chunk corpus
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
        {
            "chunk_id": "chunk_4",
            "doc_name": "billing_guidelines.pdf",
            "text": "The invoice reference number must be included in all payment correspondence.",
        },
    ]

    # Initialize HybridRetriever with candidate pool = 4 and final top_k = 2
    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25,
        semantic_retriever=semantic_provider,
        fusion_strategy="rrf",
        rrf_k=60,
        bm25_top_k=4,
        semantic_top_k=4,
        final_top_k=2,
    )

    print(f"Indexing {len(corpus)} document chunks across both BM25 and Semantic indices...")
    hybrid_retriever.add_chunks_directly(corpus)
    print("Indexing complete.\n")

    # 2. Test queries
    queries = [
        {
            "id": 1,
            "query": "When does the customer need to pay?",
            "type": "Paraphrase / Conceptual Query",
            "note": "Lexical search looks for 'pay'/'customer'. Semantic search identifies 'purchaser shall remit invoice'.",
        },
        {
            "id": 2,
            "query": "How long is the device protected?",
            "type": "Conceptual Coverage Query",
            "note": "No direct keyword match for 'device protected'. Semantic retrieval aligns with 'hardware warranty valid 24 months'.",
        },
        {
            "id": 3,
            "query": "What is the invoice reference requirement?",
            "type": "Exact Terminology Query",
            "note": "Exact keyword overlap heavily rewards chunk_4 ('invoice reference number').",
        },
    ]

    for q_item in queries:
        qid = q_item["id"]
        q_text = q_item["query"]
        q_type = q_item["type"]
        q_note = q_item["note"]

        print("=" * 80)
        print(f"QUERY {qid}: \"{q_text}\"")
        print(f"Query Type: {q_type}")
        print(f"Analysis:   {q_note}")
        print("-" * 80)

        # A. Raw BM25 results
        bm25_res = bm25.retrieve(q_text, top_k=3)
        print("  [BM25 Lexical Ranking]")
        if bm25_res:
            for rank, r in enumerate(bm25_res, 1):
                print(f"    Rank {rank}: {r['chunk_id']} | Raw Score: {r['score']:.4f} | {r['doc_name']}")
        else:
            print("    (No lexical matches)")

        # B. Semantic results
        sem_res = semantic_provider.retrieve(q_text, top_k=3)
        print("\n  [Dense Semantic Ranking]")
        if sem_res:
            for rank, r in enumerate(sem_res, 1):
                print(f"    Rank {rank}: {r['chunk_id']} | Score: {r['score']:.4f} | {r['doc_name']}")
        else:
            print("    (No semantic matches)")

        # C. Final Hybrid results
        hybrid_res = hybrid_retriever.retrieve(q_text, top_k=2)
        print("\n  [Final Hybrid Fused Ranking (RRF k=60)]")
        for rank, r in enumerate(hybrid_res, 1):
            bm25_info = f"BM25 Rank: {r.get('bm25_rank', 'N/A')}"
            sem_info = f"Semantic Rank: {r.get('semantic_rank', 'N/A')}"
            print(f"    Rank {rank}: {r['chunk_id']} | Fused Score: {r['score']:.4f} | {bm25_info}, {sem_info}")
            print(f"            Text: \"{r['text']}\"")

        print()

    print("=" * 80)
    print("Demonstration finished successfully.")
    print("Key Takeaway: Hybrid retrieval combines exact keyword precision with conceptual recall,")
    print("preventing single-retriever failure modes.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
