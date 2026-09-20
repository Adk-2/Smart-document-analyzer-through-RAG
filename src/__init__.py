__all__ = [
    "AdvancedRAGPipeline",
    "EmbeddingManager",
    "RAGRetriever",
    "VectorStore",
    "VectorStoreLoader",
    "answer_question",
    "rag_advanced",
    "rag_simple",
]


def __getattr__(name):
    if name == "EmbeddingManager":
        from .embeddings import EmbeddingManager

        return EmbeddingManager
    if name in {"VectorStore", "VectorStoreLoader"}:
        from .vector_store import VectorStore, VectorStoreLoader

        return {"VectorStore": VectorStore, "VectorStoreLoader": VectorStoreLoader}[name]
    if name == "RAGRetriever":
        from .retriever import RAGRetriever

        return RAGRetriever
    if name in {"AdvancedRAGPipeline", "rag_advanced", "rag_simple"}:
        from .pipeline import AdvancedRAGPipeline, rag_advanced, rag_simple

        return {
            "AdvancedRAGPipeline": AdvancedRAGPipeline,
            "rag_advanced": rag_advanced,
            "rag_simple": rag_simple,
        }[name]
    if name == "answer_question":
        from .rag_pipeline import answer_question

        return answer_question
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
