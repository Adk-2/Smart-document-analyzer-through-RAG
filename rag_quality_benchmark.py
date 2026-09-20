import os
import sys
import re
import time
import statistics

sys.path.append(os.path.abspath("src"))

from rag_pipeline import answer_question
from sentence_transformers import SentenceTransformer, util


# ============================================================
# EMBEDDING MODEL
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# EVALUATION QUESTIONS
#
# Import the same 29-question dataset we already created.
# This prevents the benchmark from silently using a different
# evaluation set.
# ============================================================

from evaluation import test_data


# ============================================================
# HELPERS
# ============================================================

def get_source_name(value):
    if not value:
        return ""

    value = str(value)

    # Handle Windows paths
    value = value.replace("\\", "/")

    return value.split("/")[-1]


def split_sentences(text):
    """
    Simple sentence splitter for diagnostic purposes.
    """
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) >= 10
    ]


# ============================================================
# ANSWER -> CONTEXT EVIDENCE SUPPORT
# ============================================================

def evidence_support(answer, context, threshold=0.55):

    if not answer or not context:
        return 0.0

    answer_sentences = split_sentences(answer)

    if not answer_sentences:
        return 0.0

    context_embedding = model.encode(
        context,
        convert_to_tensor=True
    )

    answer_embeddings = model.encode(
        answer_sentences,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        answer_embeddings,
        context_embedding
    ).squeeze(1)

    supported = sum(
        float(score) >= threshold
        for score in similarities
    )

    return supported / len(answer_sentences)


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def retrieval_metrics(documents, expected_source):

    if not expected_source:
        return {
            "hit1": None,
            "hit3": None,
            "hit5": None,
            "rr": None,
        }

    from rag_pipeline import _source_key

    sources = [
        get_source_name(_source_key(doc))
        for doc in (documents or [])
    ]

    expected_source = get_source_name(expected_source)

    rank = None

    for index, source in enumerate(sources, start=1):
        if source == expected_source:
            rank = index
            break

    return {
        "hit1": int(rank == 1),
        "hit3": int(rank is not None and rank <= 3),
        "hit5": int(rank is not None and rank <= 5),
        "rr": 1 / rank if rank else 0.0,
    }
def get_retrieved_sources(documents):
    from rag_pipeline import _source_key

    return [
        get_source_name(_source_key(doc))
        for doc in (documents or [])
    ]
# ============================================================
# BENCHMARK
# ============================================================

def run_benchmark():

    results = []

    print("\n")
    print("=" * 70)
    print("RAG QUALITY BENCHMARK")
    print("=" * 70)

    for index, item in enumerate(test_data, start=1):

        question = item["question"]

        print(
            f"\n[{index}/{len(test_data)}] "
            f"{question}"
        )

        start = time.perf_counter()

        # return_context=True gives us:
        # answer, sources, context, documents, etc.
        result = answer_question(
            question,
            return_context=True
        )

        latency = time.perf_counter() - start

        answer = result.get("answer", "")
        context = result.get("context", "")
        documents = result.get("documents", [])
        raw_documents = result.get("raw_documents", [])
        
        expected_source = item.get(
            "expected_source"
        )

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        raw_retrieval = retrieval_metrics(
            raw_documents,
            expected_source
        )
        final_retrieval = retrieval_metrics(
            documents,
            expected_source
        )

        raw_sources = get_retrieved_sources(raw_documents)
        final_sources = get_retrieved_sources(documents)

        expected_name = get_source_name(expected_source)

        if (
            expected_name
            and raw_sources
            and raw_sources[0] == expected_name
            and (not final_sources or final_sources[0] != expected_name)
        ):
            print("\n" + "=" * 70)
            print("RAW SUCCESS -> FINAL FAILURE")
            print("=" * 70)
            print(f"Question: {question}")
            print(f"Expected: {expected_name}")

            print("\nRAW:")
            for i, source in enumerate(raw_sources, 1):
                print(f"{i}. {source}")

            print("\nFINAL:")
            for i, source in enumerate(final_sources, 1):
                print(f"{i}. {source}")

            print("=" * 70)

            # Stop immediately so we don't waste Groq calls
            return results

        # ----------------------------------------------------
        # Failure analysis
        # ----------------------------------------------------

        raw_sources = get_retrieved_sources(raw_documents)
        final_sources = get_retrieved_sources(documents)

        expected_name = get_source_name(expected_source)

        raw_hit1 = (
            bool(raw_sources)
            and raw_sources[0] == expected_name
        )

        final_hit1 = (
            bool(final_sources)
            and final_sources[0] == expected_name
        )

        if expected_source and raw_hit1 and not final_hit1:

            print("\n*** RAW SUCCESS -> FINAL FAILURE ***")

            print(f"Question: {question}")
            print(f"Expected source: {expected_name}")

            print("\nRaw sources:")
            for rank, source in enumerate(raw_sources, start=1):
                print(f"  {rank}. {source}")

            print("\nFinal sources:")
            for rank, source in enumerate(final_sources, start=1):
                print(f"  {rank}. {source}")

            print("\nExpected source present in raw:",
                expected_name in raw_sources)

            print("Expected source present in final:",
                expected_name in final_sources)

            print("*** END FAILURE ANALYSIS ***")
        # ----------------------------------------------------
        # Evidence support
        # ----------------------------------------------------

        support = evidence_support(
            answer,
            context
        )

        # ----------------------------------------------------
        # Pipeline diagnostics
        # ----------------------------------------------------

        results.append(
            {
                "question": question,
                "latency": latency,
                "evidence_support": support,
                "raw_hit1": raw_retrieval["hit1"],
                "raw_hit3": raw_retrieval["hit3"],
                "raw_hit5": raw_retrieval["hit5"],
                "raw_rr": raw_retrieval["rr"],

                "hit1": final_retrieval["hit1"],
                "hit3": final_retrieval["hit3"],
                "hit5": final_retrieval["hit5"],
                "rr": final_retrieval["rr"],
                "answer": answer,
            }
        )

        print(
            f"Evidence support: {support:.2%}"
        )

        if final_retrieval["hit1"] is not None:

            print(
                f"Raw Hit@1: {raw_retrieval['hit1']}"
            )

            print(
                f"Final Hit@1: {final_retrieval['hit1']}"
            )

            print(
                f"Raw Hit@3: {raw_retrieval['hit3']}"
            )

            print(
                f"Final Hit@3: {final_retrieval['hit3']}"
            )


    return results


# ============================================================
# FINAL RESULTS
# ============================================================

if __name__ == "__main__":

    results = run_benchmark()

    answerable = [
        r for r in results
        if r["hit1"] is not None
    ]

    latencies = [
        r["latency"]
        for r in results
    ]

    supports = [
        r["evidence_support"]
        for r in results
    ]

    hit1_values = [
        r["hit1"]
        for r in answerable
    ]

    raw_hit1_values = [
        r["raw_hit1"]
        for r in answerable
    ]

    raw_hit3_values = [
        r["raw_hit3"]
        for r in answerable
    ]

    hit3_values = [
        r["hit3"]
        for r in answerable
    ]

    hit5_values = [
        r["hit5"]
        for r in answerable
    ]

    rr_values = [
        r["rr"]
        for r in answerable
    ]


    print("\n")
    print("=" * 70)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 70)

    print(
        f"Total questions: {len(results)}"
    )

    print(
        f"Answerable questions: {len(answerable)}"
    )

    print(
        f"Out-of-corpus questions: "
        f"{len(results) - len(answerable)}"
    )

    print(
        f"Mean evidence support: "
        f"{statistics.mean(supports):.2%}"
    )

    print(
        f"Mean Hit@1: "
        f"{statistics.mean(hit1_values):.2%}"
    )

    print(
        f"Mean Hit@3: "
        f"{statistics.mean(hit3_values):.2%}"
    )

    print(
        f"Mean Hit@5: "
        f"{statistics.mean(hit5_values):.2%}"
    )

    print(
        f"Mean MRR: "
        f"{statistics.mean(rr_values):.3f}"
    )

    print(
        f"Mean end-to-end latency: "
        f"{statistics.mean(latencies):.2f}s"
    )

    print(
        f"Median end-to-end latency: "
        f"{statistics.median(latencies):.2f}s"
    )

    print(
        f"Raw retrieval Hit@1: "
        f"{statistics.mean(raw_hit1_values):.2%}"
    )

    print(
        f"Final retrieval Hit@1: "
        f"{statistics.mean(hit1_values):.2%}"
    )

    print(
        f"Raw retrieval Hit@3: "
        f"{statistics.mean(raw_hit3_values):.2%}"
    )

    print(
        f"Final retrieval Hit@3: "
        f"{statistics.mean(hit3_values):.2%}"
    )

    print("=" * 70)