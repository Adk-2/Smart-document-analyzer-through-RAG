from typing import Any, Dict, List
import logging
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

try:
    from .config import BM25_TOP_K, FETCH_K, RERANKER_MODEL
    from .embeddings import EmbeddingManager
    from .utils import clean_docs, deduplicate_docs_advanced, get_document_metadata, get_document_text, preprocess_query
    from .vector_store import VectorStoreLoader
except ImportError:
    from config import BM25_TOP_K, FETCH_K, RERANKER_MODEL
    from embeddings import EmbeddingManager
    from utils import clean_docs, deduplicate_docs_advanced, get_document_metadata, get_document_text, preprocess_query
    from vector_store import VectorStoreLoader


logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def detect_query_section(query):
    query = query.lower()

    if "literature survey" in query:
        return "literature_survey"

    if "introduction" in query:
        return "introduction"

    if "proposed system" in query:
        return "proposed_system"

    if "requirements" in query:
        return "requirements"

    return None


def _doc_stable_key(doc: Any) -> tuple[str, str, str, str]:
    metadata = get_document_metadata(doc)
    return (
        str(metadata.get("source_file", metadata.get("source", ""))),
        str(metadata.get("page", "")),
        str(doc.get("id", "") if isinstance(doc, dict) else ""),
        get_document_text(doc)[:120],
    )


class RAGRetriever:
    def __init__(self, vector_store: VectorStoreLoader, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.keyword_docs = self._load_keyword_docs()
        self.bm25 = self._build_bm25(self.keyword_docs)
        self.reranker = None

    def _load_keyword_docs(self) -> List[Dict[str, Any]]:
        stored = self.vector_store.collection.get(include=["documents", "metadatas"])
        docs = []
        for rank, (doc_id, document, metadata) in enumerate(
            zip(stored.get("ids", []), stored.get("documents", []), stored.get("metadatas", [])),
            start=1,
        ):
            docs.append(
                {
                    "id": doc_id,
                    "document": document,
                    "metadata": dict(metadata or {}),
                    "similarity_score": None,
                    "distance": None,
                    "rank": rank,
                }
            )
        return clean_docs(sorted(docs, key=_doc_stable_key))

    def _build_bm25(self, docs: List[Any]) -> BM25Okapi | None:
        corpus = [get_document_text(doc) for doc in docs]
        tokenized_corpus = [_tokenize(doc) for doc in corpus]
        return BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def bm25_search(self, query: str, docs: List[Any] | None = None, k: int = BM25_TOP_K) -> List[Any]:
        docs = docs or self.keyword_docs
        if not docs or self.bm25 is None:
            return []

        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(
            zip(scores, docs),
            key=lambda item: (-float(item[0]), _doc_stable_key(item[1])),
        )
        return [doc.copy() if isinstance(doc, dict) else doc for _, doc in ranked[:k]]

    def _get_reranker(self) -> CrossEncoder:
        if self.reranker is None:
            self.reranker = CrossEncoder(RERANKER_MODEL)
        return self.reranker

    def _lexical_fallback_rank(self, query: str, docs: List[Any], k: int = 10) -> List[Any]:
        query_terms = set(_tokenize(query))
        identity_query = any(
            phrase in query.lower()
            for phrase in (
                "who is",
                "what is",
                "real name",
                "birth name",
                "original name",
            )
        )

        identity_markers = (
            "born",
            "also known as",
            "commonly known as",
            "was an",
            "was a",
            "is an",
            "is a",
            "birth name",
            "original name",
        )

        def score(doc: Any) -> float:
            text = get_document_text(doc).lower()
            text_terms = _tokenize(text)
            text_term_set = set(text_terms)
            overlap = len(query_terms & text_term_set)
            repeated_matches = sum(text_terms.count(term) for term in query_terms)
            marker_boost = 0

            if identity_query:
                marker_boost = sum(4 for marker in identity_markers if marker in text)

            return overlap * 3 + repeated_matches + marker_boost

        return sorted(docs, key=lambda doc: (-score(doc), _doc_stable_key(doc)))[:k]

    def _rerank(self, query: str, docs: List[Any], k: int = 10) -> List[Any]:
        if not docs:
            return []

        if len(docs) == 1:
            return docs[:k]

        try:
            pairs = [(query, get_document_text(doc)) for doc in docs]
            scores = self._get_reranker().predict(pairs)
            ranked_docs = []
            for rank, (score, doc) in enumerate(
                sorted(
                    zip(scores, docs),
                    key=lambda item: (-float(item[0]), _doc_stable_key(item[1])),
                ),
                start=1,
            ):
                if isinstance(doc, dict):
                    doc = doc.copy()
                    metadata = dict(doc.get("metadata") or {})
                    metadata["rerank_score"] = float(score)
                    doc["metadata"] = metadata
                    doc["rerank_score"] = float(score)
                    doc["rank"] = rank
                ranked_docs.append(doc)
            return ranked_docs[:k]
        except Exception as exc:
            logger.warning(
                "CrossEncoder reranking failed; falling back to lexical ranking",
                extra={
                    "event": "reranker_fallback",
                    "context": {
                        "error_type": type(exc).__name__,
                        "candidate_count": len(docs),
                        "top_k": k,
                    },
                },
            )
            return self._lexical_fallback_rank(query, docs, k=k)

    def _mmr_select(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[List[float]],
        k: int,
        lambda_mult: float = 0.5,
    ) -> List[int]:
        if candidate_embeddings is None or len(candidate_embeddings) == 0:
            return []

        candidates = np.asarray(candidate_embeddings, dtype=np.float32)
        query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(-1)

        candidate_norms = np.linalg.norm(candidates, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query_vector)
        candidate_norms[candidate_norms == 0] = 1.0
        if query_norm == 0:
            query_norm = 1.0

        normalized_candidates = candidates / candidate_norms
        normalized_query = query_vector / query_norm
        query_scores = normalized_candidates @ normalized_query

        selected = []
        remaining = list(range(len(candidates)))
        while remaining and len(selected) < k:
            if not selected:
                next_idx = sorted(
                    remaining,
                    key=lambda idx: (-float(query_scores[idx]), idx),
                )[0]
            else:
                selected_vectors = normalized_candidates[selected]
                next_idx = sorted(
                    remaining,
                    key=lambda idx: (
                        -float(
                            lambda_mult * query_scores[idx]
                            - (1 - lambda_mult) * np.max(selected_vectors @ normalized_candidates[idx])
                        ),
                        idx,
                    ),
                )[0]
            selected.append(next_idx)
            remaining.remove(next_idx)
        return selected

    def _vector_search(self, query: str, top_k: int = 8, fetch_k: int = FETCH_K) -> List[Dict[str, Any]]:
        if self.vector_store.collection.count() == 0:
            return []

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=max(fetch_k, top_k),
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]
        candidate_embeddings = results.get("embeddings", [[]])[0]
        selected_indexes = self._mmr_select(query_embedding, candidate_embeddings, top_k, lambda_mult=0.75) or list(
            range(min(top_k, len(documents)))
        )

        vector_docs = []
        for rank, index in enumerate(selected_indexes, start=1):
            distance = distances[index]
            similarity_score = 1 / (1 + distance)
            metadata = dict(metadatas[index] or {})
            metadata["score"] = similarity_score
            vector_docs.append(
                {
                    "id": ids[index],
                    "document": documents[index],
                    "metadata": metadata,
                    "similarity_score": similarity_score,
                    "distance": distance,
                    "rank": rank,
                }
            )
        return vector_docs

    def retrieve(self, query: str, top_k: int = 10, score_threshold: float | None = None) -> List[Dict[str, Any]]:
        target_section = detect_query_section(query)
        query = preprocess_query(query)
        candidate_k = max(top_k, FETCH_K)
        vector_docs = self._vector_search(query, top_k=candidate_k, fetch_k=FETCH_K)
        keyword_docs = self.bm25_search(query, self.keyword_docs, k=BM25_TOP_K)
        if score_threshold is not None:
            vector_docs = [
                doc
                for doc in vector_docs
                if doc.get("similarity_score") is None or doc.get("similarity_score", 0.0) >= score_threshold
            ]
        combined_docs = clean_docs(vector_docs + keyword_docs)

        if len(combined_docs) == 0:
            combined_docs = clean_docs(self._vector_search(query, top_k=candidate_k, fetch_k=FETCH_K))

        docs = self._rerank(query, combined_docs, k=top_k)
        docs = deduplicate_docs_advanced(docs)

        if target_section:
            boosted_docs = []
            other_docs = []

            for doc in docs:
                section = get_document_metadata(doc).get(
                    "section",
                    "general"
                )

                if section == target_section:
                    boosted_docs.append(doc)
                else:
                    other_docs.append(doc)

            docs = boosted_docs + other_docs

        print([
            get_document_metadata(doc).get("section")
            for doc in docs
        ])

        return docs[:top_k]
