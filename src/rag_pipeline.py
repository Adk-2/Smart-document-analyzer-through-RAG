import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from langchain_groq import ChatGroq

try:
    from .config import (
        ANSWER_TOP_K,
        GROQ_MAX_TOKENS,
        GROQ_MODEL,
        GROQ_REQUEST_TIMEOUT_SECONDS,
        GROQ_TEMPERATURE,
        RETRIEVAL_TIMEOUT_SECONDS,
    )
    from .document_index import DOCUMENT_INDEX
    from .pipeline import _build_context, _build_prompt, _invoke_llm, build_default_retriever
    from .utils import (
        add_grounded_reinforcements,
        deduplicate_sources,
        filter_relevant_docs,
        get_relevance_filter_diagnostics,
        get_document_metadata,
        get_document_text,
        safe_console_text,
    )
    from .workspace_manager import WorkspaceManager
except ImportError:
    from config import (
        ANSWER_TOP_K,
        GROQ_MAX_TOKENS,
        GROQ_MODEL,
        GROQ_REQUEST_TIMEOUT_SECONDS,
        GROQ_TEMPERATURE,
        RETRIEVAL_TIMEOUT_SECONDS,
    )
    from document_index import DOCUMENT_INDEX
    from pipeline import _build_context, _build_prompt, _invoke_llm, build_default_retriever
    from utils import (
        add_grounded_reinforcements,
        deduplicate_sources,
        filter_relevant_docs,
        get_relevance_filter_diagnostics,
        get_document_metadata,
        get_document_text,
        safe_console_text,
    )
    from workspace_manager import WorkspaceManager


_llm = None
_retrievers = {}
_retrieval_executor = ThreadPoolExecutor(max_workers=4)
logger = logging.getLogger(__name__)

UNSUPPORTED_ANSWER = "I could not find this information in the indexed sources."
MIN_RETRIEVAL_CONFIDENCE = 0.2
ANSWER_RETRIEVAL_TOP_K = 8
ANSWER_CONTEXT_TOP_K = 6
MIN_TOPIC_RELEVANCE = 0.45
MIN_VECTOR_SIMILARITY = 0.22
SOURCE_DOMINANCE_RATIO = 0.55

_ANSWER_ESCAPE_PATTERNS = (
    "however",
    "general knowledge",
    "outside the context",
    "outside of the context",
    "not mentioned in the context, but",
    "not in the context, but",
    "the context does not mention",
    "the provided context does not mention",
)

_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "whether",
    "who",
    "why",
    "with",
}

_QUERY_INTENT_TERMS = {
    "all",
    "application",
    "applications",
    "benefit",
    "benefits",
    "compare",
    "difference",
    "differences",
    "example",
    "examples",
    "explain",
    "feature",
    "features",
    "key",
    "list",
    "main",
    "overview",
    "summarize",
    "summary",
    "type",
    "types",
    "use",
    "uses",
}

_GENERIC_TOPIC_TERMS = {
    "ai",
    "artificial",
    "intelligence",
    "model",
    "models",
    "system",
    "systems",
    "technology",
}

_MULTI_SOURCE_QUERY_MARKERS = (
    "across documents",
    "across sources",
    "all documents",
    "all files",
    "all pdfs",
    "compare documents",
    "compare sources",
    "each document",
    "each file",
    "multiple documents",
    "multiple sources",
)

_GROUNDING_ALIASES = {
    "osho": {"osho", "rajneesh", "bhagwan", "acharya"},
    "rajneesh": {"osho", "rajneesh", "bhagwan", "acharya"},
    "bhagwan": {"osho", "rajneesh", "bhagwan", "acharya"},
    "acharya": {"osho", "rajneesh", "bhagwan", "acharya"},
    "llm": {"llm", "large language model", "large language models"},
    "large": {"llm", "large language model", "large language models"},
    "language": {"llm", "large language model", "large language models"},
    "model": {"llm", "large language model", "large language models"},
    "ai": {"ai", "artificial intelligence"},
    "artificial": {"ai", "artificial intelligence"},
    "intelligence": {"ai", "artificial intelligence"},
}

_SHORT_QUERY_TERMS = {"ai"}

_ANSWER_ALLOWED_UNSUPPORTED_TERMS = {
    "according",
    "answer",
    "based",
    "could",
    "find",
    "found",
    "indexed",
    "information",
    "mention",
    "mentioned",
    "no",
    "provided",
    "source",
    "sources",
    "state",
    "stated",
    "states",
    "yes",
}

_FOLLOW_UP_REFERENCE_PATTERN = re.compile(
    r"\b(he|she|his|her|him|it|its|they|them|their|this|that|these|those|former|latter)\b",
    re.IGNORECASE,
)


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
        _llm = ChatGroq(
            api_key=groq_api_key,
            model_name=GROQ_MODEL,
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS,
            timeout=GROQ_REQUEST_TIMEOUT_SECONDS,
        )
    return _llm


def _get_retriever(workspace_name: str | None = None):
    workspace_name = workspace_name or WorkspaceManager.get_workspace()
    if workspace_name not in _retrievers:
        _retrievers[workspace_name] = build_default_retriever(workspace_name=workspace_name)
    else:
        print(f"Using collection: {workspace_name}")
    return _retrievers[workspace_name]


def _format_conversation_history(conversation_history: list[dict[str, str]] | None, max_turns: int = 2) -> str:
    if not conversation_history:
        return ""

    sanitized_messages = []
    for message in conversation_history:
        role = (message.get("role") or "").strip().lower()
        content = (message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        label = "User" if role == "user" else "Assistant"
        sanitized_messages.append(f"{label}: {content[:600]}")

    return "\n".join(sanitized_messages[-max_turns * 2:])


def _looks_like_follow_up(query: str) -> bool:
    query_text = (query or "").strip().lower()
    if not query_text:
        return False
    follow_up_starts = (
        "what about",
        "how about",
        "and ",
        "what is his",
        "what is her",
        "what is its",
        "what did he",
        "what did she",
        "what did it",
        "where did he",
        "where did she",
        "where did it",
        "when did he",
        "when did she",
        "when did it",
        "why did he",
        "why did she",
        "why did it",
    )
    return query_text.startswith(follow_up_starts) or bool(_FOLLOW_UP_REFERENCE_PATTERN.search(query_text))


def rewrite_follow_up_query(
    query: str,
    conversation_history: list[dict[str, str]] | None = None,
    request_id: str | None = None,
) -> str:
    history_text = _format_conversation_history(conversation_history)
    original_query = (query or "").strip()
    if not original_query or not history_text or not _looks_like_follow_up(original_query):
        return original_query

    prompt = f"""Rewrite the follow-up question into a standalone search query.

Rules:
- Resolve pronouns and references using only the recent conversation shown below.
- Keep the user's intent unchanged.
- Do not answer the question.
- Do not add facts that are not needed to make the query standalone.
- Ignore older unrelated topics; use prior turns only for coreference.
- Return only the standalone query.

Conversation:
{history_text}

Follow-up question:
{original_query}

Standalone search query:"""

    try:
        rewritten_query = _invoke_with_retry(_get_llm(), prompt, retries=1).strip().strip('"')
    except Exception as exc:
        logger.warning(
            "Query rewrite failed; using original query",
            extra={
                "event": "rag_query_rewrite_failed",
                "context": {
                    "request_id": request_id,
                    "query": original_query,
                    "error_type": type(exc).__name__,
                },
            },
        )
        return original_query

    if (
        not rewritten_query
        or len(rewritten_query) > 500
        or "llm service temporarily unavailable" in rewritten_query.lower()
    ):
        return original_query

    logger.info(
        "Query rewrite completed",
        extra={
            "event": "rag_query_rewrite",
            "context": {
                "request_id": request_id,
                "original_query": original_query,
                "rewritten_query": rewritten_query,
            },
        },
    )
    return rewritten_query


def refresh_retriever(workspace_name: str | None = None):
    workspace_name = workspace_name or WorkspaceManager.get_workspace()
    _retrievers[workspace_name] = build_default_retriever(workspace_name=workspace_name)
    return _retrievers[workspace_name]


def _invoke_with_retry(
    llm,
    prompt,
    retries=2
):

    for attempt in range(retries + 1):

        try:

            return _invoke_llm(
                llm,
                prompt
            )

        except Exception as e:

            text = str(e).lower()

            if (
                "rate_limit" in text
                and attempt < retries
            ):

                print(
                    "Groq rate limit hit. Retrying..."
                )

                time.sleep(3)

                continue

            if attempt == retries:

                return (
                    "⚠️ LLM service temporarily unavailable."
                )

            raise


def _normalize_terms(text: str) -> list[str]:
    terms = re.findall(r"[a-z0-9]+", (text or "").lower())
    normalized = []
    for term in terms:
        if term in _QUERY_STOPWORDS or (len(term) < 3 and term not in _SHORT_QUERY_TERMS):
            continue
        if term.endswith("s") and len(term) > 4:
            term = term[:-1]
        normalized.append(term)
    return normalized


def _normalized_phrase_in_text(phrase: str, text: str) -> bool:
    phrase_terms = _normalize_terms(phrase)
    text_terms = _normalize_terms(text)
    if not phrase_terms:
        return False
    if len(phrase_terms) == 1:
        return phrase_terms[0] in set(text_terms)
    phrase_text = " ".join(phrase_terms)
    normalized_text = " ".join(text_terms)
    return phrase_text in normalized_text


def _term_evidence(term: str, context: str) -> dict:
    aliases = sorted(_GROUNDING_ALIASES.get(term, {term}))
    matched_aliases = [
        alias
        for alias in aliases
        if _normalized_phrase_in_text(alias, context)
    ]
    return {
        "term": term,
        "aliases": aliases,
        "matched_aliases": matched_aliases,
        "matched": bool(matched_aliases),
    }


def _chunk_has_query_evidence(query: str, doc) -> bool:
    text = get_document_text(doc)
    return any(
        _term_evidence(term, text)["matched"]
        for term in dict.fromkeys(_normalize_terms(query))
    )


def _source_key(doc) -> str:
    metadata = get_document_metadata(doc)
    return str(metadata.get("source_file", metadata.get("source", "Unknown")) or "Unknown")


def _query_focus_terms(query: str) -> list[str]:
    terms = []
    for term in dict.fromkeys(_normalize_terms(query)):
        if term in _QUERY_INTENT_TERMS or term in _GENERIC_TOPIC_TERMS:
            continue
        terms.append(term)
    if terms:
        return terms
    return [
        term
        for term in dict.fromkeys(_normalize_terms(query))
        if term not in _QUERY_INTENT_TERMS
    ]


def _is_multi_source_query(query: str) -> bool:
    query_text = (query or "").lower()
    return any(marker in query_text for marker in _MULTI_SOURCE_QUERY_MARKERS)


def _doc_topic_relevance(query: str, doc) -> float:
    text = get_document_text(doc)
    text_lower = text.lower()
    focus_terms = _query_focus_terms(query)
    if not focus_terms:
        return 1.0 if text.strip() else 0.0

    matched_terms = [
        term
        for term in focus_terms
        if _term_evidence(term, text)["matched"]
    ]
    coverage = len(matched_terms) / len(focus_terms)
    phrase_boost = 0.0
    query_lower = (query or "").lower()
    for phrase in ("generative ai", "large language model", "large language models", "self-attention"):
        if phrase in query_lower and phrase in text_lower:
            phrase_boost = max(phrase_boost, 0.25)
    return min(1.0, coverage + phrase_boost)


def _passes_similarity_floor(doc) -> bool:
    if not isinstance(doc, dict) or doc.get("similarity_score") is None:
        return True
    return float(doc.get("similarity_score") or 0.0) >= MIN_VECTOR_SIMILARITY


def _isolate_retrieval_context(query: str, docs) -> tuple[list, dict]:
    if not docs:
        return [], {
            "input_count": 0,
            "output_count": 0,
            "removed_count": 0,
            "dominant_source": None,
            "source_isolation_applied": False,
        }

    scored_docs = []
    for rank, doc in enumerate(docs, start=1):
        topic_relevance = _doc_topic_relevance(query, doc)
        metadata = dict(get_document_metadata(doc))
        metadata["topic_relevance"] = topic_relevance
        if isinstance(doc, dict):
            doc = doc.copy()
            doc["metadata"] = metadata
        scored_docs.append((doc, topic_relevance, rank))

    topic_filtered = [
        (doc, score, rank)
        for doc, score, rank in scored_docs
        if _passes_similarity_floor(doc)
    ]
    if not topic_filtered:
        topic_filtered = [
            (doc, score, rank)
            for doc, score, rank in scored_docs
            if score > 0 and _passes_similarity_floor(doc)
        ]
    if not topic_filtered:
        topic_filtered = scored_docs[: min(len(scored_docs), ANSWER_CONTEXT_TOP_K)]

    source_weights = {}
    for doc, score, _rank in topic_filtered:
        source_weights[_source_key(doc)] = source_weights.get(_source_key(doc), 0.0) + max(score, 0.1)

    dominant_source = None
    source_isolation_applied = False
    if source_weights and not _is_multi_source_query(query):
        dominant_source, dominant_weight = max(source_weights.items(), key=lambda item: item[1])
        total_weight = sum(source_weights.values())
        if total_weight and dominant_weight / total_weight >= SOURCE_DOMINANCE_RATIO:
            source_isolation_applied = True
            topic_filtered = [
                item
                for item in topic_filtered
                if _source_key(item[0]) == dominant_source or item[1] >= 0.85
            ]

    isolated = [
        doc
        for doc, _score, _rank in sorted(
            topic_filtered,
            key=lambda item: item[2],
        )
    ][:ANSWER_CONTEXT_TOP_K]

    return isolated, {
        "input_count": len(docs),
        "output_count": len(isolated),
        "removed_count": max(0, len(docs) - len(isolated)),
        "dominant_source": dominant_source,
        "source_isolation_applied": source_isolation_applied,
        "focus_terms": _query_focus_terms(query),
        "min_topic_relevance": MIN_TOPIC_RELEVANCE,
        "min_vector_similarity": MIN_VECTOR_SIMILARITY,
        "source_weights": source_weights,
    }


def _missing_query_terms(query: str, context: str) -> list[str]:
    query_terms = list(dict.fromkeys(_normalize_terms(query)))
    return [
        evidence["term"]
        for evidence in (_term_evidence(term, context) for term in query_terms)
        if not evidence["matched"]
    ]


def _context_grounding_evaluation(query: str, context: str, docs) -> dict:
    query_terms = list(dict.fromkeys(_normalize_terms(query)))
    term_evidence = [_term_evidence(term, context) for term in query_terms]
    matched_terms = [item["term"] for item in term_evidence if item["matched"]]
    missing_terms = [item["term"] for item in term_evidence if not item["matched"]]
    confidence = _retrieval_confidence(docs)
    has_docs = bool(docs)
    has_context = bool((context or "").strip())
    has_scored_docs = any(
        isinstance(doc, dict) and doc.get("similarity_score") is not None
        for doc in docs
    )
    retrieval_supported = (
        confidence >= MIN_RETRIEVAL_CONFIDENCE
        if has_scored_docs
        else has_docs
    )
    evidence_ratio = len(matched_terms) / len(query_terms) if query_terms else 1.0
    passed = has_context and retrieval_supported and (not query_terms or evidence_ratio >= 0.5)
    if passed:
        reason = "context_contains_alias_or_term_evidence"
    elif not has_context:
        reason = "no_context"
    elif not retrieval_supported:
        reason = "retrieval_confidence_too_low"
    else:
        reason = "insufficient_context_evidence"
    return {
        "passed": passed,
        "reason": reason,
        "query_terms": query_terms,
        "matched_terms": matched_terms,
        "missing_terms": missing_terms,
        "term_evidence": term_evidence,
        "retrieval_confidence": confidence,
        "retrieval_supported": retrieval_supported,
        "grounding_confidence": evidence_ratio,
    }


def _is_fallback_answer(answer: str) -> bool:
    return (answer or "").strip().lower() == UNSUPPORTED_ANSWER.lower()


def _extract_grounded_answer_from_context(query: str, context: str) -> str | None:
    query_terms = list(dict.fromkeys(_normalize_terms(query)))
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", context or "")
        if sentence.strip()
    ]
    if not sentences:
        return None

    def sentence_score(sentence: str) -> tuple[int, int]:
        matched_terms = sum(
            1
            for term in query_terms
            if _term_evidence(term, sentence)["matched"]
        )
        alias_hits = sum(
            len(_term_evidence(term, sentence)["matched_aliases"])
            for term in query_terms
        )
        return matched_terms, alias_hits

    best_sentence = max(sentences, key=sentence_score)
    if sentence_score(best_sentence) == (0, 0):
        return None
    return best_sentence


def _retrieval_confidence(docs) -> float:
    scores = [
        doc.get("similarity_score")
        for doc in docs
        if isinstance(doc, dict) and doc.get("similarity_score") is not None
    ]
    return max(scores) if scores else 0.0


def _has_sufficient_confidence(docs) -> bool:
    scored_docs = [
        doc
        for doc in docs
        if isinstance(doc, dict) and doc.get("similarity_score") is not None
    ]
    if not scored_docs:
        return bool(docs)
    return _retrieval_confidence(scored_docs) >= MIN_RETRIEVAL_CONFIDENCE


def _log_retrieval_state(
    query: str,
    workspace: str,
    docs,
    context: str = "",
    request_id: str | None = None,
) -> None:
    chunk_logs = []
    for index, doc in enumerate(docs, start=1):
        metadata = get_document_metadata(doc)
        chunk_logs.append(
            {
                "rank": index,
                "score": doc.get("similarity_score") if isinstance(doc, dict) else None,
                "distance": doc.get("distance") if isinstance(doc, dict) else None,
                "source": metadata.get("source_file", metadata.get("source", "Unknown")),
                "page": metadata.get("page", "?"),
                "preview": get_document_text(doc)[:300],
            }
        )

    logger.info(
        "Retrieved RAG chunks",
        extra={
            "event": "rag_retrieval",
            "context": {
                "request_id": request_id,
                "workspace_name": workspace,
                "query": query,
                "chunk_count": len(docs),
                "top_sources": chunk_logs[:3],
                "chunks": chunk_logs,
            },
        },
    )

    if context:
        logger.info(
            "Final RAG context passed to LLM",
            extra={
                "event": "rag_context",
                "context": {
                    "request_id": request_id,
                    "workspace_name": workspace,
                    "query": query,
                    "context_length": len(context),
                    "context": context,
                },
            },
        )


def _is_answer_grounded(answer: str, context: str) -> bool:
    answer_text = (answer or "").strip()
    if not answer_text:
        return False

    answer_lower = answer_text.lower()
    if any(pattern in answer_lower for pattern in _ANSWER_ESCAPE_PATTERNS):
        return False

    context_terms = set(_normalize_terms(context))
    answer_terms = set(_normalize_terms(answer_text))
    if not answer_terms:
        return False

    unsupported_terms = set()
    for term in answer_terms:
        if term in context_terms or term in _ANSWER_ALLOWED_UNSUPPORTED_TERMS:
            continue
        if _term_evidence(term, context)["matched"]:
            continue
        unsupported_terms.add(term)
    return not unsupported_terms


def _answer_grounding_evaluation(answer: str, context: str) -> dict:
    answer_text = (answer or "").strip()
    if not answer_text:
        return {
            "passed": False,
            "reason": "empty_answer",
            "grounding_confidence": 0.0,
            "unsupported_terms": [],
        }

    answer_lower = answer_text.lower()
    escape_patterns = [
        pattern
        for pattern in _ANSWER_ESCAPE_PATTERNS
        if pattern in answer_lower
    ]
    if escape_patterns:
        return {
            "passed": False,
            "reason": "escape_pattern",
            "grounding_confidence": 0.0,
            "unsupported_terms": [],
            "escape_patterns": escape_patterns,
        }

    answer_terms = set(_normalize_terms(answer_text))
    if not answer_terms:
        return {
            "passed": False,
            "reason": "no_answer_terms",
            "grounding_confidence": 0.0,
            "unsupported_terms": [],
        }

    context_terms = set(_normalize_terms(context))
    supported_terms = set()
    unsupported_terms = set()
    for term in answer_terms:
        if term in context_terms or term in _ANSWER_ALLOWED_UNSUPPORTED_TERMS:
            supported_terms.add(term)
            continue
        if _term_evidence(term, context)["matched"]:
            supported_terms.add(term)
            continue
        unsupported_terms.add(term)

    grounding_confidence = len(supported_terms) / len(answer_terms)
    passed = not unsupported_terms
    return {
        "passed": passed,
        "reason": "answer_terms_supported" if passed else "unsupported_answer_terms",
        "grounding_confidence": grounding_confidence,
        "supported_terms": sorted(supported_terms),
        "unsupported_terms": sorted(unsupported_terms),
        "answer_term_count": len(answer_terms),
    }


def _remove_unsupported_list_items(answer: str, context: str) -> tuple[str, dict]:
    context_terms = set(_normalize_terms(context))
    removed_items = []
    kept_lines = []
    list_item_pattern = re.compile(r"^(\s*)(?:[-*]\s+|\d+[.)]\s+)(.+)")

    for line in (answer or "").splitlines():
        match = list_item_pattern.match(line)
        if not match:
            kept_lines.append(line)
            continue

        item_text = match.group(2)
        item_terms = [
            term
            for term in dict.fromkeys(_normalize_terms(item_text))
            if term not in _ANSWER_ALLOWED_UNSUPPORTED_TERMS
            and term not in _QUERY_INTENT_TERMS
            and term not in _GENERIC_TOPIC_TERMS
        ]
        if len(item_terms) < 2:
            kept_lines.append(line)
            continue

        supported_terms = [
            term
            for term in item_terms
            if term in context_terms or _term_evidence(term, context)["matched"]
        ]
        support_ratio = len(supported_terms) / len(item_terms)
        unsupported_terms = sorted(set(item_terms) - set(supported_terms))

        if support_ratio < 0.4 and len(unsupported_terms) >= 2:
            removed_items.append(
                {
                    "text": item_text[:200],
                    "support_ratio": support_ratio,
                    "unsupported_terms": unsupported_terms[:10],
                }
            )
            continue

        kept_lines.append(line)

    cleaned_answer = "\n".join(kept_lines).strip()
    return cleaned_answer or answer, {
        "removed_count": len(removed_items),
        "removed_items": removed_items,
    }


def _unsupported_result(sources, return_context: bool, context: str = "", docs=None) -> dict:
    result = {
        "answer": UNSUPPORTED_ANSWER,
        "sources": sources,
    }
    if return_context:
        result["context"] = context
        result["documents"] = docs or []
    return result


def _retrieve_with_timeout(
    retriever,
    query: str,
    top_k: int,
    request_id: str | None,
    score_threshold: float | None = None,
):
    future = _retrieval_executor.submit(
        retriever.retrieve,
        query,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    try:
        return future.result(timeout=RETRIEVAL_TIMEOUT_SECONDS)
    except FutureTimeoutError as exc:
        future.cancel()
        logger.error(
            "RAG retrieval timed out",
            extra={
                "event": "rag_retrieval_timeout",
                "context": {
                    "request_id": request_id,
                    "top_k": top_k,
                    "timeout_seconds": RETRIEVAL_TIMEOUT_SECONDS,
                },
            },
        )
        raise TimeoutError(
            f"Retrieval timed out after {RETRIEVAL_TIMEOUT_SECONDS} seconds"
        ) from exc


def _is_global_metadata_query(query: str) -> bool:
    query_text = (query or "").lower()
    metadata_terms = (
        "how many literature",
        "all titles",
        "list papers",
        "reviewed papers",
        "literature survey",
    )
    return any(term in query_text for term in metadata_terms)


def _format_global_metadata_answer(workspace: str) -> str:
    workspace_index = DOCUMENT_INDEX.get(workspace, {})

    if not workspace_index:
        return "No document metadata available."

    def clean_entry_text(entry_text: str) -> str:
        return re.sub(
            r"^\s*(\[\d+\]|\d+[\.\)])\s+",
            "",
            entry_text,
        ).strip()

    blocks = []
    for document, metadata in workspace_index.items():
        literature_entries = metadata.get("literature_entries", [])
        entry_texts = [
            clean_entry_text(entry.get("raw", ""))
            for entry in literature_entries
            if clean_entry_text(entry.get("raw", ""))
        ]
        total_papers = len(entry_texts)

        lines = [
            f"Document: {document}",
            f"Total Literature Papers: {total_papers}",
            "",
        ]

        if entry_texts:
            lines.extend(
                f"{index}. {entry_text}"
                for index, entry_text in enumerate(entry_texts, start=1)
            )
        else:
            lines.append("1. No literature entries found.")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def expand_query(query):
    expansions = {
        "osho": "Rajneesh Bhagwan Shree Rajneesh",
        "real name": "birth name original name",
    }

    q = query.lower()

    for key, value in expansions.items():
        if key in q:
            query += " " + value

    return query


def _debug_retrieved_docs(docs):
    print("\n===== RETRIEVED DOCS =====")

    for i, doc in enumerate(docs):
        print(f"\nDOC {i + 1}")
        print(safe_console_text(get_document_metadata(doc)))
        print(safe_console_text(get_document_text(doc)[:1000]))

    print("\n==========================")


def _build_chunk_diagnostics(docs):
    diagnostics = []
    for index, doc in enumerate(docs, start=1):
        metadata = get_document_metadata(doc)
        diagnostics.append(
            {
                "rank": index,
                "source": metadata.get("source_file", metadata.get("source", "Unknown")),
                "page": metadata.get("page", "?"),
                "score": doc.get("similarity_score") if isinstance(doc, dict) else None,
                "distance": doc.get("distance") if isinstance(doc, dict) else None,
                "metadata": metadata,
                "preview": get_document_text(doc)[:300],
            }
        )
    return diagnostics


def _print_retrieval_diagnostics(
    workspace: str,
    query: str,
    request_id: str | None,
    docs,
    collection_count: int | None,
    stage: str,
) -> None:
    chunk_diagnostics = _build_chunk_diagnostics(docs)
    print("===== RAG RETRIEVAL DIAGNOSTICS =====")
    print(f"Request ID: {request_id}")
    print(f"Workspace: {workspace}")
    print(f"Stage: {stage}")
    print(f"Query: {query}")
    print(f"Vector collection count: {collection_count}")
    print(f"Retrieved chunk count: {len(docs)}")
    for chunk in chunk_diagnostics[:10]:
        print(f"Chunk {chunk['rank']}")
        print(f"Source: {chunk['source']}")
        print(f"Page: {chunk['page']}")
        print(f"Score: {chunk['score']}")
        print(f"Distance: {chunk['distance']}")
        print(f"Metadata: {safe_console_text(chunk['metadata'])}")
        print(f"Preview: {safe_console_text(chunk['preview'])}")
    print("=====================================")

    logger.info(
        "RAG retrieval diagnostics",
        extra={
            "event": "rag_retrieval_diagnostics",
            "context": {
                "request_id": request_id,
                "workspace_name": workspace,
                "query": query,
                "stage": stage,
                "collection_count": collection_count,
                "chunk_count": len(docs),
                "chunks": chunk_diagnostics,
            },
        },
    )


def answer_question(
    query,
    return_context=False,
    workspace_name: str | None = None,
    request_id: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
):
    request_started_at = time.perf_counter()
    original_query = (query or "").strip()
    rewritten_query = rewrite_follow_up_query(
        original_query,
        conversation_history=conversation_history,
        request_id=request_id,
    )
    query_for_retrieval = rewritten_query or original_query
    query_was_rewritten = query_for_retrieval != original_query

    if workspace_name is not None:
        WorkspaceManager.set_workspace(workspace_name)

    workspace = WorkspaceManager.get_workspace()
    print(f"Retrieving from workspace: {workspace}")

    if _is_global_metadata_query(query_for_retrieval):
        answer = _format_global_metadata_answer(workspace)
        result = {
            "answer": answer,
            "sources": [],
            "rewritten_query": query_for_retrieval,
            "query_was_rewritten": query_was_rewritten,
        }
        if return_context:
            result["context"] = ""
            result["documents"] = []
        logger.info(
            "RAG answer completed",
            extra={
                "event": "rag_answer_complete",
                "context": {
                    "request_id": request_id,
                    "workspace_name": workspace,
                    "query": original_query,
                    "rewritten_query": query_for_retrieval,
                    "duration_ms": round((time.perf_counter() - request_started_at) * 1000),
                    "source_count": 0,
                    "used_global_metadata": True,
                },
            },
        )
        return result

    retriever = _get_retriever(workspace_name=workspace)
    collection_count = retriever.vector_store.collection.count()
    if collection_count == 0:
        logger.warning(
            "RAG retrieval workspace has empty vector collection",
            extra={
                "event": "rag_empty_collection",
                "context": {
                    "request_id": request_id,
                    "workspace_name": workspace,
                    "query": original_query,
                },
            },
        )
        print("===== RAG EMPTY COLLECTION =====")
        print(f"Request ID: {request_id}")
        print(f"Workspace: {workspace}")
        print(f"Query: {original_query}")
        print("Vector collection is empty.")
        print("================================")
    expanded_query = expand_query(query_for_retrieval)
    top_k = ANSWER_RETRIEVAL_TOP_K
    retrieval_started_at = time.perf_counter()
    logger.info(
        "RAG retrieval started",
        extra={
            "event": "rag_retrieval_start",
            "context": {
                "request_id": request_id,
                "workspace_name": workspace,
                "query": original_query,
                "rewritten_query": query_for_retrieval,
                "top_k": top_k,
                "timeout_seconds": RETRIEVAL_TIMEOUT_SECONDS,
            },
        },
    )
    docs = _retrieve_with_timeout(
        retriever,
        expanded_query,
        top_k=top_k,
        request_id=request_id,
        score_threshold=MIN_VECTOR_SIMILARITY,
    )
    _debug_retrieved_docs(docs)
    if len(docs) == 0:
        print("Warning: No docs found. Retrying with broader retrieval...")
        docs = _retrieve_with_timeout(
            retriever,
            expanded_query,
            top_k=top_k * 2,
            request_id=request_id,
            score_threshold=MIN_VECTOR_SIMILARITY,
        )
        _debug_retrieved_docs(docs)
    raw_docs = docs
    raw_retrieval_docs = list(docs)
    retrieval_chunks = _build_chunk_diagnostics(docs)
    _print_retrieval_diagnostics(
        workspace,
        query_for_retrieval,
        request_id,
        docs,
        collection_count,
        "raw_retrieval",
    )
    logger.info(
        "RAG retrieval completed",
        extra={
            "event": "rag_retrieval_end",
            "context": {
                "request_id": request_id,
                "workspace_name": workspace,
                "query": original_query,
                "rewritten_query": query_for_retrieval,
                "chunk_count": len(docs),
                "top_sources": retrieval_chunks[:3],
                "chunks": retrieval_chunks,
                "duration_ms": round((time.perf_counter() - retrieval_started_at) * 1000),
            },
        },
    )

    filter_diagnostics = get_relevance_filter_diagnostics(docs, query_for_retrieval)
    logger.info(
        "RAG relevance filter diagnostics",
        extra={
            "event": "rag_filter_diagnostics",
            "context": {
                "request_id": request_id,
                "workspace_name": workspace,
                "query": original_query,
                "rewritten_query": query_for_retrieval,
                **filter_diagnostics,
            },
        },
    )
    print("===== RAG FILTER DIAGNOSTICS =====")
    print(f"Request ID: {request_id}")
    print(f"Workspace: {workspace}")
    print(f"Query keywords: {safe_console_text(filter_diagnostics['query_keywords'])}")
    print(f"Threshold: {filter_diagnostics['threshold']}")
    print(f"Input chunks: {filter_diagnostics['input_count']}")
    print(f"Passed chunks: {filter_diagnostics['passed_count']}")
    print(f"Failed chunks: {filter_diagnostics['failed_count']}")
    for chunk in filter_diagnostics["chunks"][:10]:
        print(f"Chunk {chunk['rank']} passed={chunk['passed']} overlap={chunk['overlap_ratio']}")
        print(f"Matched keywords: {safe_console_text(chunk['matched_keywords'])}")
        print(f"Source: {chunk['source']}")
        print(f"Preview: {safe_console_text(chunk['preview'])}")
    print("==================================")

    docs = filter_relevant_docs(docs, query_for_retrieval)
    # docs = list(docs)
    restored_docs = []
    filtered_ids = {
        doc.get("id")
        for doc in docs
        if isinstance(doc, dict) and doc.get("id")
    }
    filtered_texts = {get_document_text(doc) for doc in docs}
    for doc in raw_docs:
        doc_id = doc.get("id") if isinstance(doc, dict) else None
        already_present = (
            doc_id in filtered_ids
            if doc_id
            else get_document_text(doc) in filtered_texts
        )
        if (
            already_present
            or not _chunk_has_query_evidence(query_for_retrieval, doc)
            or _doc_topic_relevance(query_for_retrieval, doc) < MIN_TOPIC_RELEVANCE
        ):
            continue
        restored_docs.append(doc)

    if restored_docs:
        docs = restored_docs + docs
        logger.info(
            "RAG restored semantically grounded chunks after keyword filtering",
            extra={
                "event": "rag_filter_restored_evidence",
                "context": {
                    "request_id": request_id,
                    "workspace_name": workspace,
                    "query": original_query,
                    "rewritten_query": query_for_retrieval,
                    "restored_count": len(restored_docs),
                    "restored_chunks": _build_chunk_diagnostics(restored_docs),
                },
            },
        )
    docs, isolation_diagnostics = _isolate_retrieval_context(query_for_retrieval, docs)
    logger.info(
        "RAG retrieval source isolation completed",
        extra={
            "event": "rag_source_isolation",
            "context": {
                "request_id": request_id,
                "workspace_name": workspace,
                "query": original_query,
                "rewritten_query": query_for_retrieval,
                **isolation_diagnostics,
                "chunks": _build_chunk_diagnostics(docs),
            },
        },
    )
    print("===== RAG SOURCE ISOLATION =====")
    print(f"Request ID: {request_id}")
    print(f"Focus terms: {safe_console_text(isolation_diagnostics['focus_terms'])}")
    print(f"Input chunks: {isolation_diagnostics['input_count']}")
    print(f"Output chunks: {isolation_diagnostics['output_count']}")
    print(f"Dominant source: {safe_console_text(isolation_diagnostics['dominant_source'])}")
    print(f"Source isolation applied: {isolation_diagnostics['source_isolation_applied']}")
    print("================================")

    _print_retrieval_diagnostics(
        workspace,
        query_for_retrieval,
        request_id,
        docs,
        collection_count,
        "after_source_isolation",
    )
    if len(docs) == 0:
        logger.warning(
            "RAG keyword filtering removed all retrieved chunks",
            extra={
                "event": "rag_filter_empty",
                "context": {
                    "request_id": request_id,
                    "workspace_name": workspace,
                    "query": original_query,
                    "collection_count": collection_count,
                },
            },
        )
    _log_retrieval_state(query_for_retrieval, workspace, docs, request_id=request_id)

    sources = []
    for doc in docs:
        metadata = get_document_metadata(doc)
        source = metadata.get("source_file", metadata.get("source", "Unknown"))
        source_type = metadata.get("type", "pdf")
        page = metadata.get("page", "?")
        preview = get_document_text(doc)[:250]

        sources.append({
            "source": source,
            "page": page,
            "preview": preview,
            "metadata": metadata,
            "type": source_type,
        })
    sources = deduplicate_sources(sources)

    context, _ = _build_context(docs)
    if not context:
        logger.info(
            "RAG answer completed",
            extra={
                "event": "rag_answer_complete",
                "context": {
                    "request_id": request_id,
                    "workspace_name": workspace,
                    "query": original_query,
                    "duration_ms": round((time.perf_counter() - request_started_at) * 1000),
                    "source_count": len(sources),
                    "answer_type": "unsupported_no_context",
                },
            },
        )
        result = _unsupported_result(sources, return_context, docs=docs)
        result["rewritten_query"] = query_for_retrieval
        result["query_was_rewritten"] = query_was_rewritten
        return result

    _log_retrieval_state(query_for_retrieval, workspace, docs, context, request_id=request_id)
    print(f"Context length: {len(context)} characters")

    context_grounding = _context_grounding_evaluation(query_for_retrieval, context, docs)
    logger.info(
        "RAG semantic grounding diagnostics",
        extra={
            "event": "rag_semantic_grounding",
            "context": {
                "request_id": request_id,
                "workspace_name": workspace,
                "query": original_query,
                "rewritten_query": query_for_retrieval,
                **context_grounding,
            },
        },
    )
    print("===== RAG SEMANTIC GROUNDING =====")
    print(f"Request ID: {request_id}")
    print(f"Workspace: {workspace}")
    print(f"Passed: {context_grounding['passed']}")
    print(f"Reason: {context_grounding['reason']}")
    print(f"Retrieval confidence: {context_grounding['retrieval_confidence']}")
    print(f"Grounding confidence: {context_grounding['grounding_confidence']}")
    print(f"Matched terms: {safe_console_text(context_grounding['matched_terms'])}")
    print(f"Missing terms: {safe_console_text(context_grounding['missing_terms'])}")
    print(f"Term evidence: {safe_console_text(context_grounding['term_evidence'])}")
    print("==================================")
    if not context_grounding["passed"]:
        logger.info(
            "RAG semantic grounding check failed; continuing to LLM generation",
            extra={
                "event": "rag_semantic_grounding_warning",
                "context": {
                    "request_id": request_id,
                    "workspace_name": workspace,
                    "query": original_query,
                    "rewritten_query": query_for_retrieval,
                    "reason": context_grounding["reason"],
                    "missing_terms": context_grounding["missing_terms"],
                    "matched_terms": context_grounding["matched_terms"],
                    "retrieval_confidence": context_grounding["retrieval_confidence"],
                    "grounding_confidence": context_grounding["grounding_confidence"],
                    "retrieved_chunk_count": len(docs),
                    "collection_count": collection_count,
                },
            },
        )

    confidence = _retrieval_confidence(docs)
    if not _has_sufficient_confidence(docs):
        logger.info(
            "RAG retrieval confidence is low; continuing to LLM generation",
            extra={
                "event": "rag_retrieval_confidence_warning",
                "context": {
                    "request_id": request_id,
                    "workspace_name": workspace,
                    "query": original_query,
                    "rewritten_query": query_for_retrieval,
                    "confidence": confidence,
                    "min_confidence": MIN_RETRIEVAL_CONFIDENCE,
                    "retrieved_chunk_count": len(docs),
                    "collection_count": collection_count,
                },
            },
        )

    llm_started_at = time.perf_counter()
    logger.info(
        "RAG LLM generation started",
        extra={
            "event": "rag_llm_start",
            "context": {
                "request_id": request_id,
                "workspace_name": workspace,
                "query": original_query,
                "rewritten_query": query_for_retrieval,
                "context_length": len(context),
                "timeout_seconds": GROQ_REQUEST_TIMEOUT_SECONDS,
            },
        },
    )
    llm = _get_llm()
    answer = _invoke_with_retry(llm, _build_prompt(query_for_retrieval, context))
    original_llm_answer = answer
    logger.info(
        "RAG LLM generation completed",
        extra={
            "event": "rag_llm_end",
            "context": {
                "request_id": request_id,
                "workspace_name": workspace,
                "query": original_query,
                "rewritten_query": query_for_retrieval,
                "answer_length": len(answer or ""),
                "original_answer": original_llm_answer,
                "duration_ms": round((time.perf_counter() - llm_started_at) * 1000),
            },
        },
    )
    fallback_override = {
        "triggered": False,
        "reason": None,
    }
    if _is_fallback_answer(answer) and context_grounding["passed"]:
        extractive_answer = _extract_grounded_answer_from_context(query_for_retrieval, context)
        if extractive_answer:
            answer = extractive_answer
            fallback_override = {
                "triggered": True,
                "reason": "llm_returned_fallback_despite_grounded_context",
            }
        else:
            fallback_override = {
                "triggered": False,
                "reason": "no_extractive_context_sentence_found",
            }
    logger.info(
        "RAG fallback override diagnostics",
        extra={
            "event": "rag_fallback_override",
            "context": {
                "request_id": request_id,
                "workspace_name": workspace,
                "query": original_query,
                "rewritten_query": query_for_retrieval,
                "original_llm_answer": original_llm_answer,
                "final_candidate_answer": answer,
                "semantic_grounding_passed": context_grounding["passed"],
                "retrieval_confidence": confidence,
                "grounding_confidence": context_grounding["grounding_confidence"],
                **fallback_override,
            },
        },
    )
    answer = add_grounded_reinforcements(answer, context, query_for_retrieval)
    answer, answer_safety_diagnostics = _remove_unsupported_list_items(answer, context)
    if answer_safety_diagnostics["removed_count"]:
        logger.info(
            "RAG unsupported list items removed from answer",
            extra={
                "event": "rag_answer_safety_filter",
                "context": {
                    "request_id": request_id,
                    "workspace_name": workspace,
                    "query": original_query,
                    "rewritten_query": query_for_retrieval,
                    **answer_safety_diagnostics,
                },
            },
        )
    answer_grounding = _answer_grounding_evaluation(answer, context)
    logger.info(
        "RAG answer grounding diagnostics",
        extra={
            "event": "rag_answer_grounding",
            "context": {
                "request_id": request_id,
                "workspace_name": workspace,
                "query": original_query,
                "rewritten_query": query_for_retrieval,
                "retrieval_confidence": confidence,
                **answer_grounding,
            },
        },
    )
    if not answer_grounding["passed"]:
        logger.info(
            "RAG answer grounding check failed; returning answer with diagnostics",
            extra={
                "event": "rag_grounding_check_failed",
                "context": {
                    "request_id": request_id,
                    "workspace_name": workspace,
                    "query": original_query,
                    "rewritten_query": query_for_retrieval,
                    "answer": answer,
                    "reason": answer_grounding["reason"],
                    "unsupported_terms": answer_grounding.get("unsupported_terms", []),
                    "retrieval_confidence": confidence,
                    "grounding_confidence": answer_grounding["grounding_confidence"],
                },
            },
        )

    logger.info(
        "RAG answer completed",
        extra={
            "event": "rag_answer_complete",
            "context": {
                "request_id": request_id,
                "workspace_name": workspace,
                "query": original_query,
                "rewritten_query": query_for_retrieval,
                "duration_ms": round((time.perf_counter() - request_started_at) * 1000),
                "source_count": len(sources),
                "answer_type": (
                    "grounded_answer"
                    if answer_grounding["passed"]
                    else "answer_returned_with_grounding_warnings"
                ),
                "fallback_override_triggered": fallback_override["triggered"],
                "fallback_override_reason": fallback_override["reason"],
                "answer_safety_removed_items": answer_safety_diagnostics["removed_count"],
                "context_grounding_passed": context_grounding["passed"],
                "context_grounding_reason": context_grounding["reason"],
                "context_grounding_confidence": context_grounding["grounding_confidence"],
                "retrieval_confidence": confidence,
                "retrieval_confidence_passed": _has_sufficient_confidence(docs),
                "answer_grounding_passed": answer_grounding["passed"],
                "answer_grounding_reason": answer_grounding["reason"],
                "answer_grounding_confidence": answer_grounding["grounding_confidence"],
                "unsupported_terms": answer_grounding.get("unsupported_terms", []),
            },
        },
    )

    if return_context:
        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "documents": docs,
            "rewritten_query": query_for_retrieval,
            "query_was_rewritten": query_was_rewritten,
            "raw_documents": raw_retrieval_docs,
        }
    return {
        "answer": answer,
        "sources": sources,
        "rewritten_query": query_for_retrieval,
        "query_was_rewritten": query_was_rewritten,
    }
