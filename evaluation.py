import os
import sys
import time

sys.path.append(os.path.abspath("src"))

from rag_pipeline import answer_question
from sentence_transformers import SentenceTransformer, util


# ============================================================
# MODELS
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# 25-QUESTION EVALUATION SET
#
# Questions are grounded in the actual project PDFs.
# ============================================================

test_data = [

    # --------------------------------------------------------
    # ATTENTION
    # --------------------------------------------------------

    {
        "question": "What limitation of RNNs and LSTMs does attention address?",
        "expected_source": "Attention_2000.pdf",
        "expected_answer": (
            "Attention addresses the difficulty RNNs and LSTMs have with "
            "long sequences and the loss of important information when "
            "an entire sequence is encoded into a fixed-size vector."
        ),
        "key_facts": [
            "RNNs and LSTMs struggle with long sequences",
            "fixed-size vectors can lose important information",
            "attention dynamically focuses on different parts of the input",
        ],
    },

    {
        "question": "What are the three main components of attention?",
        "expected_source": "Attention_2000.pdf",
        "expected_answer": (
            "The three main components are queries, keys, and values. "
            "Queries are compared with keys to produce similarity scores, "
            "which are used to compute a weighted sum of the values."
        ),
        "key_facts": [
            "queries",
            "keys",
            "values",
            "queries are compared with keys",
        ],
    },

    {
        "question": "How does attention produce a context-aware representation?",
        "expected_source": "Attention_2000.pdf",
        "expected_answer": (
            "Attention compares queries with keys to produce similarity "
            "scores, normalizes the scores using softmax, and uses them "
            "to calculate a weighted sum of the values."
        ),
        "key_facts": [
            "queries are compared with keys",
            "similarity scores are produced",
            "softmax normalizes the scores",
            "weighted sum of values",
        ],
    },

    {
        "question": "What is self-attention?",
        "expected_source": "Attention_2000.pdf",
        "expected_answer": (
            "Self-attention is an attention mechanism in which each element "
            "in a sequence attends to all other elements, allowing the model "
            "to capture long-range dependencies."
        ),
        "key_facts": [
            "each element attends to other elements",
            "operates within a sequence",
            "captures long-range dependencies",
        ],
    },

    {
        "question": "What is multi-head attention?",
        "expected_source": "Attention_2000.pdf",
        "expected_answer": (
            "Multi-head attention runs multiple attention heads in parallel. "
            "Each head can capture different relationships, such as syntactic "
            "and semantic connections, and their outputs are combined."
        ),
        "key_facts": [
            "multiple attention heads operate in parallel",
            "different relationships can be captured",
            "syntactic relationships",
            "semantic relationships",
            "outputs are combined",
        ],
    },

    {
        "question": "What types of relationships can different attention heads capture?",
        "expected_source": "Attention_2000.pdf",
        "expected_answer": (
            "Different attention heads can capture different types of "
            "relationships, including syntactic and semantic connections."
        ),
        "key_facts": [
            "syntactic relationships",
            "semantic relationships",
            "different heads focus on different aspects",
        ],
    },

    {
        "question": "Where are attention mechanisms used?",
        "expected_source": "Attention_2000.pdf",
        "expected_answer": (
            "Attention is used in machine translation, text summarization, "
            "sentiment analysis, question answering, and computer vision."
        ),
        "key_facts": [
            "machine translation",
            "text summarization",
            "sentiment analysis",
            "question answering",
            "computer vision",
        ],
    },

    {
        "question": "How can attention weights improve interpretability?",
        "expected_source": "Attention_2000.pdf",
        "expected_answer": (
            "Researchers can examine attention weights to understand which "
            "parts of the input the model considers important, providing "
            "insight into its decision-making."
        ),
        "key_facts": [
            "attention weights can be examined",
            "identify important parts of the input",
            "provide insight into model decisions",
        ],
    },


    # --------------------------------------------------------
    # GENERATIVE AI
    # --------------------------------------------------------

    {
        "question": "What is generative AI?",
        "expected_source": "GenAI_2000.pdf",
        "expected_answer": (
            "Generative AI refers to AI systems that create new content "
            "such as text, images, audio, and video by learning patterns "
            "from large datasets."
        ),
        "key_facts": [
            "creates new content",
            "text",
            "images",
            "audio",
            "video",
            "learns patterns from large datasets",
        ],
    },

    {
        "question": "What techniques are commonly used in generative AI?",
        "expected_source": "GenAI_2000.pdf",
        "expected_answer": (
            "Common generative AI techniques include transformer models, "
            "GANs, and diffusion models."
        ),
        "key_facts": [
            "transformer models",
            "GANs",
            "diffusion models",
        ],
    },

    {
        "question": "What are some applications of generative AI?",
        "expected_source": "GenAI_2000.pdf",
        "expected_answer": (
            "Generative AI is used for content creation, design, "
            "entertainment, education, text generation, image creation, "
            "music composition, and video generation."
        ),
        "key_facts": [
            "content creation",
            "design",
            "entertainment",
            "education",
            "text generation",
            "image creation",
        ],
    },

    {
        "question": "What is a major advantage of generative AI?",
        "expected_source": "GenAI_2000.pdf",
        "expected_answer": (
            "A major advantage is that generative AI can automate creative "
            "tasks, improving productivity and creating opportunities for innovation."
        ),
        "key_facts": [
            "automates creative tasks",
            "improves productivity",
            "creates opportunities for innovation",
        ],
    },

    {
        "question": "What ethical concerns are associated with generative AI?",
        "expected_source": "GenAI_2000.pdf",
        "expected_answer": (
            "Generative AI raises concerns including misinformation, "
            "copyright issues, and misuse."
        ),
        "key_facts": [
            "misinformation",
            "copyright issues",
            "misuse",
        ],
    },

    {
        "question": "How are researchers improving the safety and reliability of generative models?",
        "expected_source": "GenAI_2000.pdf",
        "expected_answer": (
            "Researchers use techniques such as fine-tuning and "
            "reinforcement learning to align generative models with human values."
        ),
        "key_facts": [
            "fine-tuning",
            "reinforcement learning",
            "align models with human values",
        ],
    },


    # --------------------------------------------------------
    # LARGE LANGUAGE MODELS
    # --------------------------------------------------------

    {
        "question": "What are large language models?",
        "expected_source": "LLMs_2000.pdf",
        "expected_answer": (
            "Large language models are AI systems designed to understand, "
            "generate, and manipulate human language. They are trained on "
            "massive datasets containing text."
        ),
        "key_facts": [
            "AI systems",
            "understand human language",
            "generate human language",
            "trained on massive text datasets",
        ],
    },

    {
        "question": "What architecture are LLMs typically built using?",
        "expected_source": "LLMs_2000.pdf",
        "expected_answer": (
            "LLMs are typically built using transformer architectures, "
            "which rely heavily on self-attention mechanisms."
        ),
        "key_facts": [
            "transformer architectures",
            "self-attention mechanisms",
        ],
    },

    {
        "question": "What does an LLM learn to predict during training?",
        "expected_source": "LLMs_2000.pdf",
        "expected_answer": (
            "During training, an LLM learns to predict the next word in a "
            "sequence given the previous context."
        ),
        "key_facts": [
            "predict the next word",
            "uses previous context",
        ],
    },

    {
        "question": "Why is scale important for large language models?",
        "expected_source": "LLMs_2000.pdf",
        "expected_answer": (
            "Modern language models can contain billions or even trillions "
            "of parameters, allowing them to store vast amounts of information."
        ),
        "key_facts": [
            "billions of parameters",
            "trillions of parameters",
            "store vast amounts of information",
        ],
    },

    {
        "question": "What are zero-shot and few-shot learning capabilities?",
        "expected_source": "LLMs_2000.pdf",
        "expected_answer": (
            "Zero-shot and few-shot learning allow LLMs to perform tasks "
            "without explicit training on those tasks, making them versatile."
        ),
        "key_facts": [
            "zero-shot learning",
            "few-shot learning",
            "perform tasks without explicit task-specific training",
        ],
    },

    {
        "question": "What are some limitations of large language models?",
        "expected_source": "LLMs_2000.pdf",
        "expected_answer": (
            "LLMs can produce incorrect or biased outputs and require "
            "significant computational resources for training and deployment."
        ),
        "key_facts": [
            "incorrect outputs",
            "biased outputs",
            "significant computational resources",
        ],
    },

    {
        "question": "What are some applications of large language models?",
        "expected_source": "LLMs_2000.pdf",
        "expected_answer": (
            "LLMs are used in chatbots, virtual assistants, content "
            "generation tools, and coding assistants."
        ),
        "key_facts": [
            "chatbots",
            "virtual assistants",
            "content generation",
            "coding assistants",
        ],
    },

    {
        "question": "What ethical issues should be considered when using LLMs?",
        "expected_source": "LLMs_2000.pdf",
        "expected_answer": (
            "Important ethical issues include misinformation, bias, "
            "and data privacy."
        ),
        "key_facts": [
            "misinformation",
            "bias",
            "data privacy",
        ],
    },


    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    {
        "question": "What is Retrieval-Augmented Generation?",
        "expected_source": "RAG_2000.pdf",
        "expected_answer": (
            "Retrieval-Augmented Generation combines information retrieval "
            "with generative models to improve the accuracy and reliability "
            "of AI systems by incorporating external knowledge."
        ),
        "key_facts": [
            "combines information retrieval",
            "uses generative models",
            "incorporates external knowledge",
            "improves accuracy and reliability",
        ],
    },

    {
        "question": "What are the two main components of a RAG system?",
        "expected_source": "RAG_2000.pdf",
        "expected_answer": (
            "A RAG system consists of a retriever and a generator. "
            "The retriever finds relevant information and the generator "
            "uses that information to produce the response."
        ),
        "key_facts": [
            "retriever",
            "generator",
            "retriever searches for relevant information",
            "generator produces the response",
        ],
    },

    {
        "question": "How does RAG reduce hallucination?",
        "expected_source": "RAG_2000.pdf",
        "expected_answer": (
            "RAG reduces hallucination by grounding generated responses "
            "in real external data, which improves trust and reliability."
        ),
        "key_facts": [
            "grounds responses in real data",
            "reduces hallucination",
            "improves trust",
            "improves reliability",
        ],
    },

    {
        "question": "Why can RAG keep information up to date without retraining a model?",
        "expected_source": "RAG_2000.pdf",
        "expected_answer": (
            "RAG incorporates external knowledge sources during retrieval, "
            "allowing systems to use updated information without retraining "
            "the underlying language model."
        ),
        "key_facts": [
            "uses external knowledge sources",
            "retrieves information at query time",
            "does not require retraining",
            "can use updated information",
        ],
    },

    # --------------------------------------------------------
    # OUT-OF-CORPUS QUESTIONS
    #
    # These test whether the system invents information.
    # --------------------------------------------------------

    {
        "question": "What is the capital of France?",
        "expected_source": None,
        "expected_answer": None,
        "key_facts": [],
    },

    {
        "question": "Who won the 2026 FIFA World Cup?",
        "expected_source": None,
        "expected_answer": None,
        "key_facts": [],
    },

    {
        "question": "What is the population of India in 2026?",
        "expected_source": None,
        "expected_answer": None,
        "key_facts": [],
    },
]


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def semantic_score(answer, expected):

    if not expected:
        return None

    emb1 = model.encode(
        answer,
        convert_to_tensor=True
    )

    emb2 = model.encode(
        expected,
        convert_to_tensor=True
    )

    return util.cos_sim(emb1, emb2).item()


# ============================================================
# KEY-FACT COVERAGE
#
# Instead of exact string matching, compare each fact with
# the generated answer semantically.
# ============================================================

def fact_coverage(answer, key_facts):

    if not key_facts:
        return None

    answer_embedding = model.encode(
        answer,
        convert_to_tensor=True
    )

    fact_embeddings = model.encode(
        key_facts,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        fact_embeddings,
        answer_embedding
    ).squeeze(1)

    # 0.55 is used as a conservative semantic-match threshold.
    matched = sum(
        float(score) >= 0.55
        for score in similarities
    )

    return matched / len(key_facts)


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    results = []

    for index, item in enumerate(test_data, start=1):

        question = item["question"]

        print(
            f"\n[{index}/{len(test_data)}] "
            f"Evaluating: {question}"
        )

        start_time = time.perf_counter()

        result = answer_question(question)

        latency = time.perf_counter() - start_time

        answer = result["answer"]

        expected = item["expected_answer"]

        llm_failed = (
            not answer
            or "LLM service temporarily unavailable" in answer
        )

        retrieval_failed = (
            "I could not find this information in the indexed sources" in answer
        )

        if llm_failed or retrieval_failed:
            semantic = None
            coverage = None
        else:
            semantic = semantic_score(
                answer,
                expected
            )

            coverage = fact_coverage(
                answer,
                item["key_facts"]
            )

        results.append(
            {
                "question": question,
                "semantic_score": semantic,
                "fact_coverage": coverage,
                "latency": latency,
                "answer": answer,
                "expected_source": item["expected_source"],
                "llm_failed": llm_failed,
                "retrieval_failed": retrieval_failed,
            }
        )

    return results


# ============================================================
# RESULTS
# ============================================================

if __name__ == "__main__":

    results = evaluate()

    failed_generations = [
        r for r in results
        if r["llm_failed"] or r["retrieval_failed"]
    ]

    print("\n")
    print("=" * 70)
    print("GENERATION STATUS")
    print("=" * 70)

    print(
        f"Successful answer generations: "
        f"{len(results) - len(failed_generations)}"
    )

    print(
        f"Failed answer generations: "
        f"{len(failed_generations)}"
    )

    print("\n")
    print("=" * 70)
    print("LOW QUALITY ANSWERS")
    print("=" * 70)

    for r in results:
        if (
            r["expected_source"] is not None
            and (
                (r["semantic_score"] is not None and r["semantic_score"] < 0.65)
                or
                (r["fact_coverage"] is not None and r["fact_coverage"] < 0.60)
            )
        ):
            print("\nQuestion:")
            print(r["question"])

            print(f"Semantic similarity: {r['semantic_score']:.2%}")
            print(f"Key-fact coverage: {r['fact_coverage']:.2%}")

            print("\nAnswer:")
            print(r["answer"])

            print("-" * 70)

    answerable = [
        r for r in results
        if r["expected_source"] is not None
        and not r["llm_failed"]
        and not r["retrieval_failed"]
    ]

    print("\n")
    print("=" * 70)
    print("RAG ANSWER QUALITY BENCHMARK")
    print("=" * 70)

    for r in results:

        print("\n" + "-" * 70)

        print(
            f"Question: {r['question']}"
        )

        if r["semantic_score"] is not None:

            print(
                f"Semantic similarity: "
                f"{r['semantic_score']:.2%}"
            )

            print(
                f"Key-fact coverage: "
                f"{r['fact_coverage']:.2%}"
            )

        else:

            print(
                "Type: OUT-OF-CORPUS"
            )

        print(
            f"End-to-end latency: "
            f"{r['latency']:.2f} seconds"
        )

        print(
            f"Expected source: "
            f"{r['expected_source']}"
        )

        print(
            f"Answer: "
            f"{r['answer'][:300]}..."
        )


    # --------------------------------------------------------
    # Overall answerable-question metrics
    # --------------------------------------------------------

    mean_semantic = (
        sum(r["semantic_score"] for r in answerable)
        / len(answerable)
    )

    mean_coverage = (
        sum(r["fact_coverage"] for r in answerable)
        / len(answerable)
    )

    mean_latency = (
        sum(r["latency"] for r in answerable)
        / len(answerable)
    )

    latencies = sorted(
        r["latency"] for r in answerable
    )

    median_latency = latencies[
        len(latencies) // 2
    ]



    print("\n")
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(
        f"Answerable questions: "
        f"{len(answerable)}"
    )

    print(
        f"Out-of-corpus questions: "
        f"{len(results) - len(answerable)}"
    )

    print(
        f"Mean semantic similarity: "
        f"{mean_semantic:.2%}"
    )

    print(
        f"Mean key-fact coverage: "
        f"{mean_coverage:.2%}"
    )

    print(
        f"Mean end-to-end latency: "
        f"{mean_latency:.2f} seconds"
    )

    print(
        f"Median end-to-end latency: "
        f"{median_latency:.2f} seconds"
    )

    print("=" * 70)