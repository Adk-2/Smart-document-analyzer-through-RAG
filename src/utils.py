import re
from typing import Any, Dict, List

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


def get_document_text(doc: Any) -> str:
    if isinstance(doc, dict):
        return doc.get("document") or doc.get("text") or doc.get("content") or ""
    return getattr(doc, "page_content", str(doc))


def safe_console_text(value: Any) -> str:
    return str(value).encode("ascii", errors="replace").decode("ascii")


def set_document_text(doc: Any, text: str) -> Any:
    if isinstance(doc, dict):
        doc["document"] = text
    else:
        doc.page_content = text
    return doc


def get_document_metadata(doc: Any) -> Dict[str, Any]:
    if isinstance(doc, dict):
        return doc.get("metadata") or {}
    return getattr(doc, "metadata", {}) or {}


def webpage_to_document(webpage):
    return Document(
        page_content=webpage["text"],
        metadata={
            "source": webpage["url"],
            "title": webpage["title"],
            "type": "webpage"
        }
    )


def truncate_text(text, max_chars=4000):
    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n...[truncated]"


def get_page_number(doc: Any) -> int:
    page = get_document_metadata(doc).get("page", 0)
    try:
        return int(page)
    except (TypeError, ValueError):
        return 0


def deduplicate_docs_advanced(docs: List[Any]) -> List[Any]:
    seen_text = set()
    seen_meta = set()
    unique_docs = []

    for doc in docs:
        text = " ".join(get_document_text(doc).split())
        metadata = get_document_metadata(doc)
        source = metadata.get("source") or metadata.get("source_file")
        page = metadata.get("page")
        meta_key = (source, page) if page is not None else None

        if text in seen_text:
            continue

        if meta_key is not None and meta_key in seen_meta:
            continue

        if text:
            seen_text.add(text)
            if meta_key is not None:
                seen_meta.add(meta_key)
            unique_docs.append(set_document_text(doc, text))

    return unique_docs


def deduplicate_sources(sources):
    seen = set()
    unique = []

    for src in sources:
        key = (
            src["source"],
            src["preview"]
        )

        if key not in seen:
            seen.add(key)
            unique.append(src)

    return unique


def clean_docs(docs: List[Any]) -> List[Any]:
    return deduplicate_docs_advanced(docs)


def compress_context(docs: List[Any], max_docs: int = 8) -> List[str]:
    docs = deduplicate_docs_advanced(docs)
    docs = docs[:max_docs]

    seen = set()
    final_docs = []

    for doc in docs:
        text = get_document_text(doc).strip()
        if text not in seen:
            seen.add(text)
            final_docs.append(text)

    return final_docs


RELEVANCE_STOPWORDS = {
    "a",
    "an",
    "are",
    "does",
    "how",
    "in",
    "is",
    "of",
    "the",
    "what",
    "who",
    "why",
}


def normalize_keyword_tokens(text: str, remove_stopwords: bool = False) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    normalized = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if remove_stopwords and token in RELEVANCE_STOPWORDS:
            continue
        normalized.append(token)
    return normalized


def extract_query_keywords(query: str) -> List[str]:
    return list(dict.fromkeys(normalize_keyword_tokens(query, remove_stopwords=True)))


def filter_relevant_docs(docs, query: str, threshold: float = 0.1):
    """
    Filter docs by keyword overlap with query.
    Falls back to all docs if nothing matches.
    """

    query_keywords = extract_query_keywords(query)

    if not query_keywords:
        return docs

    relevant_docs = []

    for doc in docs:
        chunk_tokens = set(normalize_keyword_tokens(get_document_text(doc)))
        overlap = sum(1 for keyword in query_keywords if keyword in chunk_tokens)

        if overlap / len(query_keywords) >= threshold:
            relevant_docs.append(doc)

    return relevant_docs if relevant_docs else docs


def get_relevance_filter_diagnostics(docs, query: str, threshold: float = 0.1) -> Dict[str, Any]:
    normalized_query_tokens = normalize_keyword_tokens(query)
    query_keywords = extract_query_keywords(query)
    diagnostics = {
        "threshold": threshold,
        "normalized_query_tokens": normalized_query_tokens,
        "query_keywords": query_keywords,
        "input_count": len(docs),
        "passed_count": 0,
        "failed_count": 0,
        "chunks": [],
    }

    if not query_keywords:
        diagnostics["passed_count"] = len(docs)
        return diagnostics

    for index, doc in enumerate(docs, start=1):
        normalized_chunk_tokens = normalize_keyword_tokens(get_document_text(doc))
        chunk_token_set = set(normalized_chunk_tokens)
        matched_keywords = [
            keyword
            for keyword in query_keywords
            if keyword in chunk_token_set
        ]
        overlap_ratio = len(matched_keywords) / len(query_keywords)
        passed = overlap_ratio >= threshold
        diagnostics["chunks"].append(
            {
                "rank": index,
                "passed": passed,
                "overlap_ratio": overlap_ratio,
                "matched_keywords": matched_keywords,
                "normalized_chunk_tokens": normalized_chunk_tokens[:80],
                "source": get_document_metadata(doc).get("source_file", get_document_metadata(doc).get("source", "")),
                "preview": get_document_text(doc)[:200],
            }
        )
        if passed:
            diagnostics["passed_count"] += 1
        else:
            diagnostics["failed_count"] += 1

    return diagnostics


def get_source_names(docs: List[Any]) -> List[str]:
    sources = []
    for doc in docs:
        metadata = get_document_metadata(doc)
        source = metadata.get("source_file", metadata.get("source", ""))
        if source and source not in sources:
            sources.append(source)
    return sources


def preprocess_query(query: str) -> str:
    expansions = {
        "llm": "large language models",
        "large language model": "language text generation model",
        "self-attention": "self sequence relationship",
        "transformer": "transformer architecture attention model multi-head",
        "generative ai": "content generation models AI generate content",
    }

    query_lower = query.lower()

    for key, expansion in expansions.items():
        if key in query_lower:
            query += " " + expansion

    return query


def get_reinforcement_terms(query: str) -> List[str]:
    term_map = {
        "self-attention": ["self", "sequence", "relationship"],
        "transformer": ["attention", "multi-head"],
        "generative ai": ["generate", "content"],
        "large language model": ["language", "text"],
        "llm": ["language", "text"],
    }

    query_lower = query.lower()
    terms = []
    for key, values in term_map.items():
        if key in query_lower:
            terms.extend(values)
    return terms


def polish_answer_text(answer: str) -> str:
    if not answer:
        return answer

    text = answer.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"^[ \t]*based on the (?:provided |retrieved )?context[:,]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^[ \t]*(?:while|although) the (?:provided |retrieved )?context "
        r"(?:does not explicitly discuss|does not mention|doesn't mention)[^,.]*[,.]\s*"
        r"(?:it (?:does )?(?:state|states|show|shows|indicate|indicates) that\s*)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = normalize_markdown_answer(text)

    lines = []
    previous = None
    for line in text.split("\n"):
        normalized = line.strip().lower()
        if normalized and normalized == previous:
            continue
        lines.append(line.rstrip())
        previous = normalized

    text = "\n".join(lines).strip()
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]
    return text


def _normalize_heading_line(line: str) -> List[str]:
    stripped = line.strip()
    heading_match = re.match(r"^(#{1,6})\s*(.+)$", stripped)
    if not heading_match:
        return [line.rstrip()]

    level = heading_match.group(1)
    heading_text = heading_match.group(2).strip()
    headings = []
    max_iterations = 12

    for _ in range(max_iterations):
        split_match = re.match(r"^(.+?)\s+#{1,6}\s+(.+)$", heading_text)
        if not split_match:
            break
        current = split_match.group(1).strip()
        if current:
            headings.append(f"{level} {current}")
        heading_text = split_match.group(2).strip()

    heading_text = re.sub(r"\s+#{1,6}\s*$", "", heading_text).strip()
    if heading_text:
        headings.append(f"{level} {heading_text}")

    return headings or [line.rstrip()]


def normalize_markdown_answer(answer: str) -> str:
    if not answer:
        return answer

    text = answer.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"(?<!\n)\s+(#{1,6}\s+)", r"\n\n\1", text)

    normalized_lines = []
    in_code_block = False

    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            normalized_lines.append(stripped)
            continue

        if in_code_block:
            normalized_lines.append(raw_line.rstrip())
            continue

        if not stripped:
            normalized_lines.append("")
            continue

        stripped = re.sub(r"^[•‣◦]\s*", "- ", stripped)
        stripped = re.sub(r"^([*-])\s*", r"\1 ", stripped)
        stripped = re.sub(r"^(\d+)[.)]\s*", r"\1. ", stripped)

        for heading_line in _normalize_heading_line(stripped):
            normalized_lines.append(heading_line)

    compact_lines = []
    for line in normalized_lines:
        stripped = line.strip()
        if not stripped:
            if compact_lines and compact_lines[-1] != "":
                compact_lines.append("")
            continue

        is_heading = bool(re.match(r"^#{1,6}\s+\S", stripped))
        is_list_item = bool(re.match(r"^(\s*)([-*]\s+|\d+\.\s+)", line))
        previous = compact_lines[-1] if compact_lines else None
        previous_is_list = bool(previous and re.match(r"^(\s*)([-*]\s+|\d+\.\s+)", previous))

        if is_heading and compact_lines and compact_lines[-1] != "":
            compact_lines.append("")
        elif is_list_item and compact_lines and compact_lines[-1] != "" and not previous_is_list:
            compact_lines.append("")

        compact_lines.append(stripped if is_heading or is_list_item else line.strip())

        if is_heading:
            compact_lines.append("")

    text = "\n".join(compact_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def add_grounded_reinforcements(answer: str, context: str, query: str) -> str:
    return polish_answer_text(answer)


expand_query = preprocess_query
