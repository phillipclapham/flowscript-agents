"""
VectorIndex — Pure-Python vector similarity search over Memory nodes.

Stores embeddings in a sidecar file ({memory_path}.embeddings.json) to avoid
bloating the main MemoryJSON with large float arrays. Pre-normalizes vectors
at storage time so dot product = cosine similarity (fast search).

Design:
- Composes on any Memory instance (no Memory changes needed)
- Sidecar persistence: embeddings saved/loaded independently
- Pure Python math (no numpy required) — fast enough for <10K nodes
- Batch embedding via provider for efficiency
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..memory import Memory, Node

from .providers import EmbeddingProvider


# =============================================================================
# Result types
# =============================================================================


@dataclass
class VectorSearchResult:
    """A single vector search result."""

    node_id: str
    score: float  # cosine similarity (0-1 for normalized vectors)
    content: str
    node_type: str
    tier: str | None = None
    frequency: int | None = None


# =============================================================================
# Pure-Python vector math
# =============================================================================


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def _magnitude(v: list[float]) -> float:
    """Euclidean magnitude of a vector."""
    return math.sqrt(sum(x * x for x in v))


def _normalize(v: list[float]) -> list[float]:
    """Normalize vector to unit length. Returns zero vector if magnitude is 0."""
    mag = _magnitude(v)
    if mag == 0:
        return v
    return [x / mag for x in v]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two pre-normalized vectors = dot product."""
    return _dot(a, b)


# =============================================================================
# VectorIndex
# =============================================================================


class VectorIndex:
    """Vector similarity search over Memory nodes.

    Stores pre-normalized embeddings in memory for fast search.
    Persists to sidecar file alongside the Memory JSON.

    Usage:
        mem = Memory.load_or_create("./agent.json")
        index = VectorIndex(mem, OpenAIEmbeddings())
        index.index_all()          # embed all unindexed nodes
        results = index.search("database decisions", top_k=5)
        index.save()               # persist to ./agent.json.embeddings.json
    """

    def __init__(self, memory: Memory, provider: EmbeddingProvider) -> None:
        self._memory = memory
        self._provider = provider
        # node_id → pre-normalized embedding vector
        self._vectors: dict[str, list[float]] = {}
        # Track which provider generated each embedding (for reindex detection)
        self._provider_id: str = repr(provider)

    @property
    def memory(self) -> Memory:
        return self._memory

    @property
    def indexed_count(self) -> int:
        """Number of nodes with embeddings."""
        return len(self._vectors)

    @property
    def sidecar_path(self) -> str | None:
        """Path to the sidecar embeddings file."""
        if not self._memory.file_path:
            return None
        return self._memory.file_path + ".embeddings.json"

    # -- Indexing --

    def index_node(self, node_id: str) -> bool:
        """Embed a single node and add to index. Returns True if newly indexed."""
        node = self._memory.get_node(node_id)
        if node is None:
            return False
        if node_id in self._vectors:
            return False  # already indexed

        embeddings = self._provider.embed([node.content])
        if not embeddings or not embeddings[0]:
            return False

        self._vectors[node_id] = _normalize(embeddings[0])
        return True

    def index_all(self) -> int:
        """Embed all unindexed nodes in batch. Returns count of newly indexed."""
        to_embed: list[str] = []  # contents
        node_ids: list[str] = []

        for node_ref in self._memory.nodes:
            if node_ref.id not in self._vectors:
                to_embed.append(node_ref.content)
                node_ids.append(node_ref.id)

        if not to_embed:
            return 0

        # Batch embed for efficiency
        embeddings = self._provider.embed(to_embed)

        indexed = 0
        for nid, emb in zip(node_ids, embeddings):
            if emb:
                self._vectors[nid] = _normalize(emb)
                indexed += 1

        return indexed

    def remove_node(self, node_id: str) -> bool:
        """Remove a node's embedding from the index."""
        if node_id in self._vectors:
            del self._vectors[node_id]
            return True
        return False

    def reindex_all(self) -> int:
        """Re-embed ALL nodes (e.g., after changing providers). Returns count."""
        self._vectors.clear()
        return self.index_all()

    # -- Search --

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search by semantic similarity. Returns results sorted by score descending."""
        if not self._vectors:
            return []

        # Embed the query
        query_embeddings = self._provider.embed([query])
        if not query_embeddings or not query_embeddings[0]:
            return []
        query_vec = _normalize(query_embeddings[0])

        # Score all indexed nodes
        scored: list[tuple[str, float]] = []
        for node_id, vec in self._vectors.items():
            score = _cosine_similarity(query_vec, vec)
            if score >= threshold:
                scored.append((node_id, score))

        # Sort by score descending, take top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        # Build results with temporal enrichment
        results: list[VectorSearchResult] = []
        for node_id, score in scored:
            node = self._memory.get_node(node_id)
            if node is None:
                continue  # node was removed since indexing
            temporal = self._memory.get_temporal(node_id)
            results.append(
                VectorSearchResult(
                    node_id=node_id,
                    score=score,
                    content=node.content,
                    node_type=node.type.value,
                    tier=temporal.tier if temporal else None,
                    frequency=temporal.frequency if temporal else None,
                )
            )

        return results

    def find_similar(
        self,
        node_id: str,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Find nodes semantically similar to a given node.
        Useful for dedup detection."""
        if node_id not in self._vectors:
            return []

        query_vec = self._vectors[node_id]

        scored: list[tuple[str, float]] = []
        for nid, vec in self._vectors.items():
            if nid == node_id:
                continue  # skip self
            score = _cosine_similarity(query_vec, vec)
            if score >= threshold:
                scored.append((nid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        results: list[VectorSearchResult] = []
        for nid, score in scored:
            node = self._memory.get_node(nid)
            if node is None:
                continue
            temporal = self._memory.get_temporal(nid)
            results.append(
                VectorSearchResult(
                    node_id=nid,
                    score=score,
                    content=node.content,
                    node_type=node.type.value,
                    tier=temporal.tier if temporal else None,
                    frequency=temporal.frequency if temporal else None,
                )
            )
        return results

    # -- Sidecar persistence --

    def save(self, file_path: str | None = None) -> None:
        """Save embeddings to sidecar file. Atomic write (temp + rename)."""
        target = file_path or self.sidecar_path
        if target is None:
            raise ValueError(
                "No sidecar path available. Set memory.file_path first or provide path."
            )
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "flowscript_embeddings": "1.0.0",
            "provider": self._provider_id,
            "dimensions": self._provider.dimensions,
            "vectors": self._vectors,  # already normalized lists
        }

        # Atomic write
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".embeddings-"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load(self, file_path: str | None = None) -> int:
        """Load embeddings from sidecar file. Returns count loaded.

        Silently returns 0 if file doesn't exist or is corrupt (embeddings
        are regenerable). Validates dimensions match current provider and
        node IDs still exist in memory.
        """
        import sys

        target = file_path or self.sidecar_path
        if target is None or not Path(target).exists():
            return 0

        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(
                f"VectorIndex: corrupt sidecar file {target}, will re-index: {e}",
                file=sys.stderr,
            )
            return 0

        # Validate dimensions match current provider
        saved_dims = data.get("dimensions")
        current_dims = self._provider.dimensions
        if saved_dims is not None and saved_dims != current_dims:
            print(
                f"VectorIndex: dimension mismatch (saved={saved_dims}, "
                f"current={current_dims}). Discarding stale embeddings — will re-index.",
                file=sys.stderr,
            )
            return 0

        vectors = data.get("vectors", {})
        count = 0
        for node_id, vec in vectors.items():
            # Only load if node still exists in memory
            if self._memory.get_node(node_id) is None:
                continue
            # Validate vector length matches expected dimensions
            if len(vec) != current_dims:
                continue
            self._vectors[node_id] = _normalize(vec)  # re-normalize defensively
            count += 1

        return count

    def __repr__(self) -> str:
        return (
            f"VectorIndex(indexed={len(self._vectors)}, "
            f"memory_nodes={self._memory.size}, "
            f"provider={self._provider_id})"
        )
