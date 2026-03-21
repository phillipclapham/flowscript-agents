"""
Embedding providers for FlowScript vector search.

Three implementations, all lazy-importing their dependencies:
- OpenAIEmbeddings: cloud, best quality, requires openai package
- SentenceTransformerEmbeddings: local, free, requires sentence-transformers
- OllamaEmbeddings: local, free, zero additional deps (raw HTTP)

Custom providers: implement the EmbeddingProvider protocol (embed + dimensions).
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers. Implement embed() and dimensions."""

    @property
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        ...


# =============================================================================
# OpenAI
# =============================================================================

# Known dimensions for OpenAI models (avoids a probe call)
_OPENAI_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddings:
    """OpenAI embeddings via the openai Python package.

    Requires: pip install openai
    Default model: text-embedding-3-small ($0.02/MTok, 1536 dims)
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI embeddings require the openai package. "
                "Install with: pip install openai"
            )
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._dims = _OPENAI_DIMS.get(model)

    @property
    def dimensions(self) -> int:
        if self._dims is None:
            # Probe with a test embedding to discover dimensions
            result = self.embed(["test"])
            self._dims = len(result[0])
        return self._dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(input=texts, model=self._model)
        # Sort by index to ensure order matches input
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    def __repr__(self) -> str:
        return f"OpenAIEmbeddings(model={self._model!r}, dims={self._dims})"


# =============================================================================
# Sentence Transformers (local, free)
# =============================================================================


class SentenceTransformerEmbeddings:
    """Local embeddings via sentence-transformers.

    Requires: pip install sentence-transformers
    Default model: all-MiniLM-L6-v2 (384 dims, 22M params, very fast)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Local embeddings require sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(model_name)
        self._dims: int = self._model.get_sentence_embedding_dimension()

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # normalize_embeddings=True makes dot product = cosine similarity
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def __repr__(self) -> str:
        return f"SentenceTransformerEmbeddings(dims={self._dims})"


# =============================================================================
# Ollama (local, zero additional Python deps — raw HTTP)
# =============================================================================


class OllamaEmbeddings:
    """Local embeddings via Ollama REST API.

    Requires: Ollama running locally (ollama.com)
    Default model: nomic-embed-text (768 dims, 137M params)
    Zero Python package dependencies — uses stdlib urllib.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dims: int | None = None

    @property
    def dimensions(self) -> int:
        if self._dims is None:
            result = self.embed(["test"])
            self._dims = len(result[0])
        return self._dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = json.dumps({"model": self._model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self._base_url}. "
                f"Is Ollama running? Error: {e}"
            )
        embeddings = result.get("embeddings", [])
        if not embeddings:
            raise ValueError(
                f"Ollama returned no embeddings. Model '{self._model}' may not "
                "support embeddings. Try: ollama pull nomic-embed-text"
            )
        if self._dims is None:
            self._dims = len(embeddings[0])
        return embeddings

    def __repr__(self) -> str:
        return f"OllamaEmbeddings(model={self._model!r}, dims={self._dims})"
