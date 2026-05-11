from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None


@dataclass
class Embedder:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        self._model = None
        embeddings_enabled = os.getenv("EMBEDDINGS_ENABLED", "0").strip() in {"1", "true", "True"}
        if SentenceTransformer is not None and embeddings_enabled:
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception:
                self._model = None

    def create_index(self, embeddings: np.ndarray):
        if faiss is None:
            return None
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product for Cosine Similarity
        index.add(embeddings.astype("float32"))
        return index

    @property
    def enabled(self) -> bool:
        return self._model is not None

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)
        if self._model is None:
            # Fallback: return deterministic pseudo-vectors so pipeline still runs.
            vectors = []
            for text in texts:
                base = np.zeros(384, dtype=np.float32)
                for idx, ch in enumerate(text.lower()[:1500]):
                    base[idx % 384] += (ord(ch) % 31) / 31.0
                norm = np.linalg.norm(base)
                vectors.append(base if norm == 0 else base / norm)
            return np.vstack(vectors)
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.array(vectors, dtype=np.float32)
