from typing import List
import hashlib
import json
import os
import warnings

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from .config import EMBEDDING_MODEL
except ImportError:
    from config import EMBEDDING_MODEL


class EmbeddingManager:
    FALLBACK_DIMENSIONS = 384

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        cache_dir: str = "data/embed_cache",
    ):
        self.model_name = model_name
        self.model = None
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        key = self._cache_key(texts)

        cache_path = os.path.join(
            self.cache_dir,
            f"{key}.pkl",
        )

        if os.path.exists(cache_path):
            print("Loading embeddings from cache...")
            return joblib.load(cache_path)

        print(f"Generating embeddings for {len(texts)} texts...")

        try:
            embeddings = self._get_model().encode(
                texts,
                show_progress_bar=True,
                batch_size=8,
            )
        except Exception as exc:
            warnings.warn(
                f"Embedding model unavailable; using deterministic fallback embeddings. Reason: {exc}",
                RuntimeWarning,
            )
            embeddings = self._generate_fallback_embeddings(texts)

        joblib.dump(embeddings, cache_path)

        return embeddings

    def _cache_key(self, texts: List[str]) -> str:
        payload = {
            "model_name": self.model_name,
            "texts": [text if isinstance(text, str) else str(text) for text in texts],
        }
        content = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def _generate_fallback_embeddings(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vector = np.zeros(self.FALLBACK_DIMENSIONS, dtype=np.float32)
            tokens = (text or "").lower().split()

            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                for offset, byte in enumerate(digest):
                    index = (byte + offset) % self.FALLBACK_DIMENSIONS
                    vector[index] += 1.0

            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            vectors.append(vector)

        return np.vstack(vectors) if vectors else np.empty((0, self.FALLBACK_DIMENSIONS))
