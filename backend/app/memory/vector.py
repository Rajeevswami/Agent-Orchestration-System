"""Semantic memory: ChromaDB-backed vector store with pluggable embeddings.

Two embedding strategies are supported:

* ``hashing`` — deterministic, fully offline feature-hashing embedder. Great for
  development, CI and demos (no model downloads, no API keys).
* ``openai``  — OpenAI embeddings for production-grade retrieval quality.

The store talks to Chroma in three modes: ``persistent`` (local directory),
``ephemeral`` (in-process) or ``http`` (a Chroma server, e.g. in Docker).
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.config import Settings, get_settings

try:  # Chroma is a hard dependency in requirements.txt, but degrade gracefully.
    import chromadb
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

    CHROMA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without chromadb installed
    CHROMA_AVAILABLE = False
    Documents = List[str]  # type: ignore[misc, assignment]
    Embeddings = List[List[float]]  # type: ignore[misc, assignment]

    class EmbeddingFunction:  # type: ignore[no-redef]
        """Minimal fallback protocol when chromadb is not importable."""


TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder(EmbeddingFunction[Documents]):
    """Feature-hashing bag-of-words embedder — offline and deterministic.

    Not as sharp as a neural embedding model, but plenty for demo-scale
    retrieval and it removes every network dependency from the memory path.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    @staticmethod
    def name() -> str:
        return "hashing-embedder"

    def get_config(self) -> Dict[str, Any]:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "HashingEmbedder":
        return HashingEmbedder(dim=int(config.get("dim", 512)))

    def default_space(self) -> str:
        return "cosine"

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - chroma protocol name
        out: Embeddings = []
        for text in input:
            vec = [0.0] * self.dim
            for token in TOKEN_RE.findall(text.lower()):
                digest = int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "big")
                vec[digest % self.dim] += 1.0
                vec[(digest >> 16) % self.dim] += 0.5
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class OpenAIEmbedder(EmbeddingFunction[Documents]):
    """OpenAI embeddings wrapper implementing the Chroma embedding protocol."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = ""):
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key or None)

    @staticmethod
    def name() -> str:
        return "openai-embedder"

    def get_config(self) -> Dict[str, Any]:
        return {"model": self.model}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "OpenAIEmbedder":
        return OpenAIEmbedder(model=config.get("model", "text-embedding-3-small"))

    def default_space(self) -> str:
        return "cosine"

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        response = self._client.embeddings.create(model=self.model, input=list(input))
        return [item.embedding for item in response.data]


@dataclass
class MemoryHit:
    id: str
    text: str
    metadata: dict
    score: float

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "score": round(self.score, 4),
        }


def build_embedder(settings: Settings):
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(
            model=settings.embedding_model, api_key=settings.openai_api_key
        )
    return HashingEmbedder()


class VectorMemory:
    """Semantic memory over a ChromaDB collection."""

    COLLECTION = "agent-semantic-memory"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if not CHROMA_AVAILABLE:  # pragma: no cover
            raise RuntimeError(
                "chromadb is required for semantic memory. Install requirements.txt."
            )
        self.embedder = build_embedder(self.settings)
        self._client = self._make_client()
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )

    def _make_client(self):
        mode = self.settings.chroma_mode
        if mode == "http":
            return chromadb.HttpClient(
                host=self.settings.chroma_host, port=self.settings.chroma_port
            )
        if mode == "ephemeral":
            return chromadb.EphemeralClient()
        return chromadb.PersistentClient(path=self.settings.chroma_path)

    # ── Core operations ──────────────────────────────────────────────────────
    def add(
        self,
        text: str,
        metadata: dict | None = None,
        memory_id: str | None = None,
    ) -> str:
        memory_id = memory_id or uuid.uuid4().hex
        meta = {k: v for k, v in (metadata or {}).items() if v is not None}
        meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        self._collection.add(ids=[memory_id], documents=[text], metadatas=[meta])
        return memory_id

    def search(
        self,
        query: str,
        k: int | None = None,
        where: dict | None = None,
    ) -> list[MemoryHit]:
        k = k or self.settings.memory_top_k
        if self._collection.count() == 0:
            return []
        kwargs: dict[str, Any] = {"query_texts": [query], "n_results": min(k, self._collection.count())}
        if where:
            kwargs["where"] = where
        result = self._collection.query(**kwargs)
        hits: list[MemoryHit] = []
        for i, doc_id in enumerate(result["ids"][0]):
            distance = result["distances"][0][i] if result.get("distances") else 0.0
            hits.append(
                MemoryHit(
                    id=doc_id,
                    text=result["documents"][0][i] or "",
                    metadata=(result["metadatas"][0][i] or {}) if result.get("metadatas") else {},
                    score=1.0 - float(distance),  # cosine distance -> similarity
                )
            )
        return hits

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        self._client.delete_collection(self.COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )
