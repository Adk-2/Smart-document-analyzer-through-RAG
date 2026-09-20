from typing import Any, List
import os
import uuid
from pathlib import Path

import chromadb
import numpy as np

try:
    from .utils import get_document_metadata, get_document_text, safe_console_text
except ImportError:
    from utils import get_document_metadata, get_document_text, safe_console_text


def get_vector_store_path() -> str:
    path = Path("data/vector_store")
    if path.exists():
        return str(path)
    raise FileNotFoundError("Could not find data/vector_store. Run the vector store setup first.")


class VectorStore:
    """Simple ChromaDB vector store used by the notebook ingestion workflow."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str | None = None,
    ):
        if not collection_name:
            raise ValueError("collection_name must be provided.")

        self.collection_name = collection_name
        self.persist_directory = persist_directory or "data/vector_store"
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
        )
        print(f"Using collection: {collection_name}")

    def inspect(self, limit: int = 5) -> dict:
        total_documents = self.collection.count()
        sample = self.collection.get(limit=limit, include=["documents", "metadatas"])
        diagnostics = {
            "workspace_name": self.collection_name,
            "total_documents": total_documents,
            "sample_chunks": [
                {
                    "id": doc_id,
                    "metadata": metadata or {},
                    "preview": (document or "")[:300],
                }
                for doc_id, document, metadata in zip(
                    sample.get("ids", []),
                    sample.get("documents", []),
                    sample.get("metadatas", []),
                )
            ],
        }
        print("===== VECTOR STORE INSPECTION =====")
        print(f"Workspace: {diagnostics['workspace_name']}")
        print(f"Total documents: {diagnostics['total_documents']}")
        for index, chunk in enumerate(diagnostics["sample_chunks"], start=1):
            print(f"Stored chunk {index}")
            print(f"Metadata: {safe_console_text(chunk['metadata'])}")
            print(f"Preview: {safe_console_text(chunk['preview'])}")
        print("===================================")
        return diagnostics

    def add_documents(self, documents: List[Any], embeddings: np.ndarray | None = None):
        before_count = self.collection.count()
        if embeddings is None:
            try:
                from .embeddings import EmbeddingManager
            except ImportError:
                from embeddings import EmbeddingManager

            texts = [get_document_text(doc) for doc in documents]
            embeddings = EmbeddingManager().generate_embeddings(texts)

        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match.")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            metadatas.append(dict(get_document_metadata(doc)))
            documents_text.append(get_document_text(doc))
            embeddings_list.append(embedding.tolist() if hasattr(embedding, "tolist") else embedding)

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text,
        )
        after_count = self.collection.count()
        inserted_count = after_count - before_count
        print(f"Added {len(documents)} docs to collection")
        print(f"Vector insertion count: {inserted_count}")
        print(f"Collection now contains {after_count} documents.")
        return {
            "requested_count": len(documents),
            "before_count": before_count,
            "after_count": after_count,
            "inserted_count": inserted_count,
        }


def inspect_vector_store(collection_name: str, limit: int = 5) -> dict:
    return VectorStore(collection_name=collection_name).inspect(limit=limit)


class VectorStoreLoader(VectorStore):
    pass
