from typing import Any, Dict

try:
    from langchain.document_loaders import PyPDFLoader
except ImportError:
    from langchain_community.document_loaders import PyPDFLoader

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .config import ANSWER_TOP_K, CHUNK_OVERLAP, CHUNK_SIZE, RAG_CONTEXT_MAX_CHARS, RAG_CONTEXT_MAX_DOCS
    from .embeddings import EmbeddingManager
    from .retriever import RAGRetriever
    from .utils import (
        add_grounded_reinforcements,
        compress_context,
        filter_relevant_docs,
        get_document_metadata,
        get_document_text,
        safe_console_text,
        get_source_names,
        truncate_text,
        webpage_to_document,
    )
    from .vector_store import VectorStore, VectorStoreLoader
    from .web_loader import WebLoader
    from .workspace_manager import WorkspaceManager
except ImportError:
    from config import ANSWER_TOP_K, CHUNK_OVERLAP, CHUNK_SIZE, RAG_CONTEXT_MAX_CHARS, RAG_CONTEXT_MAX_DOCS
    from embeddings import EmbeddingManager
    from retriever import RAGRetriever
    from utils import add_grounded_reinforcements, compress_context, filter_relevant_docs, get_document_metadata, get_document_text, get_source_names, safe_console_text, truncate_text, webpage_to_document
    from vector_store import VectorStore, VectorStoreLoader
    from web_loader import WebLoader
    from workspace_manager import WorkspaceManager


UNSUPPORTED_ANSWER = "I could not find this information in the indexed sources."


def _build_context(docs):
    context_list = compress_context(docs, max_docs=RAG_CONTEXT_MAX_DOCS)
    context = "\n\n---\n\n".join(context_list)
    if len(context_list) > 1:
        context += "\n\nUse information from ALL sections above."
    context = truncate_text(
        context,
        max_chars=RAG_CONTEXT_MAX_CHARS
    )
    return context, context_list


def _build_prompt(query: str, context: str) -> str:
    return f"""
You are an educational RAG tutor and synthesis assistant.

Answer the user question using ONLY the retrieved context below.

Grounding rules:
- Every claim in the answer must be directly supported by the retrieved context.
- Do not use general knowledge, memory, assumptions, or outside facts.
- Do not continue with "however", "general knowledge", or unsupported additions.
- Treat the retrieved context as the only source of truth for this answer, even if other topics appeared earlier in the conversation.
- Do not introduce examples, applications, comparisons, or project details unless they appear in the retrieved context and are relevant to the current question.
- Ignore unrelated context sections that do not help answer the current question.
- If the retrieved context does not contain the answer, return exactly:
{UNSUPPORTED_ANSWER}
- If the context only partially supports an answer, answer the supported part naturally and stop there.
- Do not repeatedly say that the context does or does not mention something.
- Do not start with defensive phrases such as "While the provided context..." when the context supports a direct answer.
- Do not dump raw chunks or copy long passages. Synthesize the relevant ideas in your own words while preserving meaning.

Answer quality rules:
- Write like a clear educational AI assistant: natural, polished, and easy to read.
- Start with a direct answer in 1-2 sentences when helpful.
- Then organize the details using clean Markdown.
- Use section headings sparingly and only when they improve scanning.
- Put each heading on its own line with one blank line before and after it.
- Put one bullet or numbered item per line.
- Use bullet points for applications, advantages, features, types, examples, or lists.
- For each bullet, include a short explanation, not just a keyword.
- Use numbered steps when the question asks for a process, workflow, or sequence.
- For comparisons, prefer clean bullet-style sections instead of tables:
  X:
  - ...
  Y:
  - ...
  Key Differences:
  1. ...
  2. ...
- Use a Markdown table only when the comparison is large enough that a table is clearly more readable.
- Synthesize across all relevant retrieved sections instead of repeating the same point.
- Avoid duplicated phrases, retrieval-style wording, clutter, filler, and unsupported speculation.
- Do not include citations or source lists unless explicitly asked; the application displays sources separately.

Retrieved context:
{context}

User question:
{query}

Structured grounded answer:
"""


def _invoke_llm(llm, prompt: str) -> str:
    response = llm.invoke(prompt)
    if hasattr(response, "content"):
        return response.content.strip()
    return response.strip()


def rag_simple(query, retriever, llm, top_k=ANSWER_TOP_K):
    docs = retriever.retrieve(query, top_k=top_k)
    if len(docs) == 0:
        print("Warning: No docs found. Retrying with broader retrieval...")
        docs = retriever.retrieve(query, top_k=top_k * 2)

    docs = filter_relevant_docs(docs, query)
    context, _ = _build_context(docs)
    if not context:
        return UNSUPPORTED_ANSWER

    answer = _invoke_llm(llm, _build_prompt(query, context))
    sources_used = get_source_names(docs)
    answer = f"""
{answer.strip()}

Sources used:
- {', '.join(sources_used)}
""".strip() if sources_used else answer.strip()
    return add_grounded_reinforcements(answer, context, query)


def rag_advanced(query, retriever, llm, top_k=ANSWER_TOP_K, min_score=None, return_context=False):
    docs = retriever.retrieve(query, top_k=top_k)

    if len(docs) == 0:
        print("Warning: No docs found. Retrying with broader retrieval...")
        docs = retriever.retrieve(query, top_k=top_k * 2)
    if not docs:
        return {
            "answer": UNSUPPORTED_ANSWER,
            "sources": [],
            "confidence_score": 0.0,
            "context": ""
        }

    docs = filter_relevant_docs(docs, query)
    context, _ = _build_context(docs)
    sources = [{
        "source": doc["metadata"].get("source_file", doc["metadata"].get("source", "unknown")),
        "page": doc["metadata"].get("page", "unknown"),
        "score": doc.get("similarity_score"),
        "preview": get_document_text(doc)[:300] + "..."
    } for doc in docs]
    scored_docs = [doc["similarity_score"] for doc in docs if doc.get("similarity_score") is not None]
    confidence = max(scored_docs) if scored_docs else 0.0

    response = _invoke_llm(llm, _build_prompt(query, context))
    sources_used = get_source_names(docs)
    answer = f"""
{response.strip()}

Sources used:
- {', '.join(sources_used)}
""".strip() if sources_used else response.strip()

    output = {
        "answer": answer,
        "sources": sources,
        "confidence_score": confidence,
    }
    if return_context:
        output["context"] = context
    return output


class AdvancedRAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []

    def query(
        self,
        question: str,
        top_k: int = 6,
        min_score=None,
        stream: bool = False,
        summarize: bool = False,
    ) -> Dict[str, Any]:
        docs = self.retriever.retrieve(question, top_k=top_k)
        if len(docs) == 0:
            print("Warning: No docs found. Retrying with broader retrieval...")
            docs = self.retriever.retrieve(question, top_k=top_k * 2)
        if not docs:
            answer = UNSUPPORTED_ANSWER
            sources = []
            summary = None
            self.history.append({
                "question": question,
                "answer": answer,
                "sources": sources,
                "summary": summary,
            })
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "summary": summary,
                "history": self.history,
            }

        docs = filter_relevant_docs(docs, question)
        context, _ = _build_context(docs)
        sources = [{
            "source": doc["metadata"].get("source_file", doc["metadata"].get("source", "unknown")),
            "page": doc["metadata"].get("page", "unknown"),
            "score": doc.get("similarity_score"),
            "preview": get_document_text(doc)[:300] + "...",
        } for doc in docs]

        if stream:
            print("Generating answer...")

        answer = _invoke_llm(self.llm, _build_prompt(question, context))
        sources_used = get_source_names(docs)
        answer_with_citation = f"""
{answer}

Sources used:
- {', '.join(sources_used)}
""".strip() if sources_used else answer

        summary = None
        if summarize:
            summary_prompt = f"""Summarize the following answer concisely:\n\n{answer_with_citation}\n\nSummary:"""
            summary = _invoke_llm(self.llm, summary_prompt)

        self.history.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "summary": summary,
        })

        return {
            "question": question,
            "answer": answer_with_citation,
            "sources": sources,
            "summary": summary,
            "history": self.history,
        }


def build_default_retriever(workspace_name: str | None = None) -> RAGRetriever:
    workspace = workspace_name or WorkspaceManager.get_workspace()
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore(collection_name=workspace)
    return RAGRetriever(vector_store, embedding_manager)


def ingest_pdfs(pdf_paths, source_names=None):
    try:
        from .document_index import DOCUMENT_INDEX, clear_workspace_index
        from .pdf_structure import extract_literature_entries, extract_paper_titles
    except ImportError:
        from document_index import DOCUMENT_INDEX, clear_workspace_index
        from pdf_structure import extract_literature_entries, extract_paper_titles

    workspace = WorkspaceManager.get_workspace()
    clear_workspace_index(workspace)

    print(f"Ingesting PDFs into workspace: {workspace}")

    vector_store = VectorStore(
        collection_name=workspace
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    all_docs = []
    workspace_index = DOCUMENT_INDEX.setdefault(workspace, {})
    source_names = source_names or {}

    for path in pdf_paths:
        source_name = source_names.get(path, path)
        loader = PyPDFLoader(path)

        docs = loader.load()

        full_text = "\n\n".join(doc.page_content for doc in docs)
        literature_entries = extract_literature_entries(full_text)
        workspace_index[source_name] = {
            "literature_entries": literature_entries,
            "total_papers": len(literature_entries),
        }
        print(f"Indexed {len(literature_entries)} literature entries from {source_name}")
        print(DOCUMENT_INDEX)

        for doc in docs:
            doc.metadata["source"] = source_name
            doc.metadata["page"] = (
                doc.metadata.get("page", 0)
            )

            text = doc.page_content.lower()

            if "literature survey" in text:
                doc.metadata["section"] = "literature_survey"
            elif "introduction" in text:
                doc.metadata["section"] = "introduction"
            elif "proposed system" in text:
                doc.metadata["section"] = "proposed_system"
            elif "requirements" in text:
                doc.metadata["section"] = "requirements"
            else:
                doc.metadata["section"] = "general"

        split_docs = splitter.split_documents(docs)

        all_docs.extend(split_docs)

    print(f"Total chunks created: {len(all_docs)}")

    vector_store.add_documents(all_docs)

    print(f"Added chunks to workspace: {workspace}")

    return len(all_docs)


def ingest_url(url, vector_store=None, workspace_name: str | None = None):
    workspace = workspace_name or WorkspaceManager.get_workspace()
    print(f"Ingesting into workspace: {workspace}")

    webpage = WebLoader.load_url(url)

    doc = webpage_to_document(webpage)

    if vector_store is None:
        vector_store = VectorStore(
            collection_name=workspace
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = splitter.split_documents([doc])

    for chunk in chunks:

        chunk.metadata.update({
            "source": url,
            "type": "webpage"
        })

    chunk_diagnostics = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_diagnostics.append(
            {
                "rank": index,
                "metadata": dict(get_document_metadata(chunk)),
                "preview": get_document_text(chunk)[:300],
            }
        )

    print("===== URL INGESTION DIAGNOSTICS =====")
    print(f"Workspace: {workspace}")
    print(f"URL: {url}")
    print(f"Total chunks created: {len(chunks)}")
    for chunk in chunk_diagnostics[:10]:
        print(f"Chunk {chunk['rank']}")
        print(f"Metadata: {safe_console_text(chunk['metadata'])}")
        print(f"Preview: {safe_console_text(chunk['preview'])}")
    print("=====================================")

    insertion = vector_store.add_documents(chunks)
    vector_state = vector_store.inspect(limit=5)

    print(f"Added {len(chunks)} webpage chunks")

    return {
        "title": webpage["title"],
        "url": webpage["url"],
        "workspace": workspace,
        "chunks": len(chunks),
        "chunk_diagnostics": chunk_diagnostics,
        "vector_insertion": insertion,
        "vector_state": vector_state,
    }
