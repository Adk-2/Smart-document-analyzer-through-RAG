import os
import sys
import time

sys.path.append(os.path.abspath("src"))

from pipeline import build_default_retriever


TEST_DATA = [
    {
        "question": "What are attention mechanisms?",
        "expected_source": "Attention_2000.pdf",
    },
    {
        "question": "What is self-attention?",
        "expected_source": "Attention_2000.pdf",
    },
    {
        "question": "What are transformers?",
        "expected_source": "Attention_2000.pdf",
    },
    {
        "question": "What is generative AI?",
        "expected_source": "GenAI_2000.pdf",
    },
    {
        "question": "What are large language models?",
        "expected_source": "LLMs_2000.pdf",
    },
]


def get_source(doc):
    metadata = doc.get("metadata", {})
    source = metadata.get("source_file", metadata.get("source", ""))
    return os.path.basename(str(source))


def reciprocal_rank(sources, expected):
    for rank, source in enumerate(sources, start=1):
        if source == expected:
            return 1 / rank
    return 0.0


def main():
    retriever = build_default_retriever()

    results = []

    print("\n===== RETRIEVAL BENCHMARK =====\n")

    for item in TEST_DATA:
        question = item["question"]
        expected = item["expected_source"]

        start = time.perf_counter()

        docs = retriever.retrieve(
            question,
            top_k=8,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        sources = [get_source(doc) for doc in docs]

        rr = reciprocal_rank(sources, expected)

        results.append(
            {
                "question": question,
                "expected": expected,
                "sources": sources,
                "hit1": int(len(sources) >= 1 and sources[0] == expected),
                "hit3": int(expected in sources[:3]),
                "hit5": int(expected in sources[:5]),
                "rr": rr,
                "latency_ms": latency_ms,
            }
        )

        print(f"Question: {question}")
        print(f"Expected: {expected}")
        print(f"Retrieved: {sources}")
        print(f"Hit@1: {results[-1]['hit1']}")
        print(f"Hit@3: {results[-1]['hit3']}")
        print(f"Hit@5: {results[-1]['hit5']}")
        print(f"Reciprocal Rank: {rr:.3f}")
        print(f"Latency: {latency_ms:.1f} ms")
        print("-" * 60)

    n = len(results)

    hit1 = sum(r["hit1"] for r in results) / n
    hit3 = sum(r["hit3"] for r in results) / n
    hit5 = sum(r["hit5"] for r in results) / n
    mrr = sum(r["rr"] for r in results) / n
    avg_latency = sum(r["latency_ms"] for r in results) / n

    print("\n===== BASELINE RESULTS =====")
    print(f"Hit@1: {hit1:.2%}")
    print(f"Hit@3: {hit3:.2%}")
    print(f"Hit@5: {hit5:.2%}")
    print(f"MRR: {mrr:.3f}")
    print(f"Average retrieval latency: {avg_latency:.1f} ms")
    print("============================")


if __name__ == "__main__":
    main()