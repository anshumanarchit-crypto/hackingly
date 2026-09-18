"""
Latency and Throughput Benchmark for ProofMesh Dense Semantic Retrieval.
Measures:
- Embedder initialization & model load latency
- Batch indexing latency across scale (10, 100, 500 chunks)
- Query embedding latency
- Vector cosine similarity & ranking latency
- End-to-end retrieval latency
"""

import sys
import os
import time
from typing import List

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
    from dense_retrieval.embeddings import SemanticEmbedder
    from dense_retrieval.semantic_retriever import SemanticRetriever
except ImportError:
    from proofmesh.dense_retrieval.embeddings import SemanticEmbedder  # type: ignore[import-not-found]
    from proofmesh.dense_retrieval.semantic_retriever import SemanticRetriever  # type: ignore[import-not-found]


def generate_synthetic_chunks(n: int) -> List[dict]:
    """Generates synthetic legal and technical document chunks for benchmarking."""
    topics = [
        "The software licensee shall maintain strict confidentiality regarding source algorithms and architecture.",
        "System uptime SLA is guaranteed at ninety-nine point nine percent excluding scheduled maintenance windows.",
        "Emergency data backups are mirrored across dual off-site geographic data centers daily at midnight.",
        "The vendor warrants hardware against operational defects for twenty-four months post-delivery.",
        "Invoices must be remitted by electronic wire transfer within thirty calendar days of billing.",
    ]
    chunks = []
    for i in range(n):
        base = topics[i % len(topics)]
        text = f"{base} Chunk sequence index {i} with audit metadata record."
        chunks.append({
            "chunk_id": f"bench_chunk_{i:04d}",
            "doc_name": f"doc_{i // 10}.pdf",
            "text": text,
        })
    return chunks


def run_benchmark():
    print("=" * 80)
    print("ProofMesh Dense Semantic Retrieval - Benchmark Suite")
    print("=" * 80)

    # 1. Initialization latency
    t0 = time.perf_counter()
    embedder = SemanticEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        batch_size=32,
        allow_fallback=True,
    )
    init_latency_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Model load latency (trigger first lazy load)
    t0 = time.perf_counter()
    dimension = embedder.get_dimension()
    load_latency_ms = (time.perf_counter() - t0) * 1000.0

    backend = embedder.backend_type.upper()
    is_mock = embedder.is_mock

    print(f"Backend:            {backend}")
    if is_mock:
        print(">> NOTICE: BENCHMARK RUNNING ON MOCK BACKEND <<")
        print(">> Note: Latencies represent mock hashing throughput, not neural inference. <<")
    print(f"Configured Model:   {embedder.model_name}")
    print(f"Device:             {embedder.device}")
    print(f"Vector Dimension:   {dimension}")
    print(f"Init Latency:       {init_latency_ms:.2f} ms")
    print(f"Model Load Latency: {load_latency_ms:.2f} ms")
    print("-" * 80)

    # 3. Indexing benchmarks across corpus sizes
    test_sizes = [10, 100, 500]
    retrievers = {}

    print(f"{'Corpus Size':<15} | {'Indexing Time (ms)':<20} | {'Throughput (chunks/sec)':<25}")
    print("-" * 65)

    for size in test_sizes:
        chunks = generate_synthetic_chunks(size)
        r = SemanticRetriever(embedder=embedder, batch_size=32)  # type: ignore[arg-type]

        t_start = time.perf_counter()
        r.add_chunks_directly(chunks)
        t_total_ms = (time.perf_counter() - t_start) * 1000.0
        throughput = (size / (t_total_ms / 1000.0)) if t_total_ms > 0 else 0.0

        retrievers[size] = r
        print(f"{size:<15} | {t_total_ms:<20.2f} | {throughput:<25.1f}")

    print("-" * 80)

    # 4. Query Latency Breakdown (on 500-chunk index)
    r_500 = retrievers[500]
    query = "What is the guaranteed system uptime SLA and maintenance window?"

    # Query embedding latency (repeated 10 times for stable average)
    num_runs = 10
    q_embed_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        q_vec = embedder.encode_text(query, normalize=True)
        q_embed_times.append((time.perf_counter() - t0) * 1000.0)
    avg_q_embed_ms = sum(q_embed_times) / len(q_embed_times)

    # Search (matrix multiply + ranking) latency on 500 chunks
    search_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        # Dot product and sort
        if r_500._embeddings is not None:
            scores = r_500._embeddings @ q_vec
            candidates = [(float(scores[i]), r_500.chunks[i].chunk_id, i) for i in range(len(r_500.chunks))]
            candidates.sort(key=lambda item: (-item[0], item[1]))
            _ = candidates[:3]
        search_times.append((time.perf_counter() - t0) * 1000.0)
    avg_search_ms = sum(search_times) / len(search_times)

    # End-to-end retrieve() latency
    e2e_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = r_500.retrieve(query, top_k=3)
        e2e_times.append((time.perf_counter() - t0) * 1000.0)
    avg_e2e_ms = sum(e2e_times) / len(e2e_times)

    print("Query Latency Analysis (500 chunks, top_k=3, averaged over 10 runs):")
    print(f"  - Query Embedding:       {avg_q_embed_ms:.3f} ms")
    print(f"  - Dot-Product & Ranking:  {avg_search_ms:.3f} ms")
    print(f"  - Total End-to-End:       {avg_e2e_ms:.3f} ms")
    print("=" * 80)
    print("Benchmark complete.")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
