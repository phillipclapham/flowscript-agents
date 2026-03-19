"""
FlowScript CrewAI Integration.

Implements CrewAI's StorageBackend protocol, making FlowScript memory
available as a drop-in storage backend for CrewAI agents.

Usage:
    from flowscript_agents.crewai import FlowScriptStorage

    storage = FlowScriptStorage("./agent-memory.json")
    # Use with CrewAI Memory:
    # memory = Memory(storage=storage, llm="...", embedder={...})
    # crew = Crew(agents=[...], tasks=[...], memory=memory)

Note: CrewAI requires Python <3.14. This module is importable on 3.14+
but CrewAI itself won't install.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from .memory import Memory, NodeRef


class FlowScriptStorage:
    """CrewAI StorageBackend backed by FlowScript reasoning memory.

    Implements the CrewAI StorageBackend protocol (duck-typed, no inheritance
    needed). Each MemoryRecord maps to a FlowScript node.

    Access FlowScript's semantic queries via the .memory property:
        storage.memory.query.tensions()
        storage.memory.query.blocked()

    For semantic queries, use resolve() to get a NodeRef::

        ref = storage.resolve("record_123")
        ref.block(reason="Waiting on API keys")
        storage.memory.query.blocked()
    """

    def __init__(self, file_path: str | None = None) -> None:
        if file_path:
            self._memory = Memory.load_or_create(file_path)
        else:
            self._memory = Memory()
        # Index: record_id → {node_id, record_data}
        self._records: dict[str, _RecordEntry] = {}
        self._rebuild_index()
        # Start temporal session
        self._memory.session_start()

    @property
    def memory(self) -> Memory:
        return self._memory

    def resolve(self, record_id: str) -> NodeRef | None:
        """Resolve a stored record to a FlowScript NodeRef for semantic operations.

        Returns None if the record doesn't exist. Use the returned NodeRef to
        build relationships that power FlowScript's semantic queries::

            ref1 = storage.resolve("decision_123")
            ref2 = storage.resolve("concern_456")
            if ref1 and ref2:
                ref1.tension_with(ref2, axis="speed vs safety")
                ref1.decide(rationale="Accepted the tradeoff")

            storage.memory.query.tensions()
        """
        entry = self._records.get(record_id)
        if entry is None:
            return None
        try:
            return self._memory.ref(entry.node_id)
        except KeyError:
            return None

    def _rebuild_index(self) -> None:
        for ref in self._memory.nodes:
            node = ref.node
            if node.ext and "crewai_record_id" in node.ext:
                rec_id = node.ext["crewai_record_id"]
                self._records[rec_id] = _RecordEntry(
                    node_id=node.id,
                    record_id=rec_id,
                    content=node.content,
                    scope=node.ext.get("scope", "/"),
                    categories=node.ext.get("categories", []),
                    metadata=node.ext.get("metadata", {}),
                    importance=node.ext.get("importance", 0.5),
                    created_at=node.ext.get("created_at", node.provenance.timestamp),
                    last_accessed=node.ext.get("last_accessed", node.provenance.timestamp),
                    embedding=node.ext.get("embedding"),
                    source=node.ext.get("source"),
                    private=node.ext.get("private", False),
                )

    # -- Required StorageBackend methods --

    def save(self, records: list[Any]) -> None:
        """Save MemoryRecord objects."""
        for record in records:
            rec_id = getattr(record, "id", str(uuid.uuid4()))
            content = getattr(record, "content", str(record))
            scope = getattr(record, "scope", "/")
            categories = getattr(record, "categories", [])
            metadata = getattr(record, "metadata", {})
            importance = getattr(record, "importance", 0.5)
            created_at = getattr(record, "created_at", datetime.now(timezone.utc))
            last_accessed = getattr(record, "last_accessed", datetime.now(timezone.utc))
            embedding = getattr(record, "embedding", None)
            source = getattr(record, "source", None)
            private = getattr(record, "private", False)

            ref = self._memory.thought(content)
            node = ref.node
            node.ext = node.ext or {}
            node.ext.update({
                "crewai_record_id": rec_id,
                "scope": scope,
                "categories": categories,
                "metadata": metadata,
                "importance": importance,
                "created_at": _dt_to_str(created_at),
                "last_accessed": _dt_to_str(last_accessed),
                "embedding": embedding,
                "source": source,
                "private": private,
            })

            self._records[rec_id] = _RecordEntry(
                node_id=ref.id,
                record_id=rec_id,
                content=content,
                scope=scope,
                categories=categories,
                metadata=metadata,
                importance=importance,
                created_at=_dt_to_str(created_at),
                last_accessed=_dt_to_str(last_accessed),
                embedding=embedding,
                source=source,
                private=private,
            )

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[Any, float]]:
        """Search for records. Returns (MemoryRecord-like, score) tuples.

        Uses cosine similarity when embeddings are available,
        falls back to content matching otherwise.
        """
        results: list[tuple[_RecordEntry, float]] = []

        for entry in self._records.values():
            # Scope filter
            if scope_prefix and not entry.scope.startswith(scope_prefix):
                continue

            # Category filter
            if categories:
                if not any(c in entry.categories for c in categories):
                    continue

            # Metadata filter
            if metadata_filter:
                if not all(
                    entry.metadata.get(k) == v for k, v in metadata_filter.items()
                ):
                    continue

            # Score: cosine similarity if embeddings exist
            score = 0.5  # default
            if query_embedding and entry.embedding:
                score = _cosine_similarity(query_embedding, entry.embedding)

            if score >= min_score:
                results.append((entry, score))

        results.sort(key=lambda x: -x[1])
        final = results[:limit]
        # Touch found nodes — search engagement drives temporal graduation
        touched_ids = [e.node_id for e, _ in final]
        if touched_ids:
            self._memory.touch_nodes_session_scoped(touched_ids)
        return [(e.to_dict(), s) for e, s in final]

    def delete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: Any | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        """Delete matching records. Returns count deleted.

        When record_ids is provided, only those specific records are deleted
        (other filters are ignored — ID match is exact). When record_ids is
        None, records are filtered by scope, categories, and metadata.
        """
        to_delete: list[str] = []

        if record_ids is not None:
            # Exact ID match — delete specified records regardless of other filters
            for rec_id in record_ids:
                if rec_id in self._records:
                    to_delete.append(rec_id)
        else:
            # Filter-based deletion
            has_filters = scope_prefix or categories or metadata_filter or older_than
            for rec_id, entry in self._records.items():
                if not has_filters:
                    # No filters = match all
                    to_delete.append(rec_id)
                    continue

                match = True
                if scope_prefix and not entry.scope.startswith(scope_prefix):
                    match = False
                if categories and not any(c in entry.categories for c in categories):
                    match = False
                if metadata_filter and not all(
                    entry.metadata.get(k) == v for k, v in metadata_filter.items()
                ):
                    match = False
                if match:
                    to_delete.append(rec_id)

        for rec_id in to_delete:
            entry = self._records.pop(rec_id)
            self._memory.remove_node(entry.node_id)

        return len(to_delete)

    def update(self, record: Any) -> None:
        """Update an existing record. Syncs changes to the FlowScript node."""
        rec_id = getattr(record, "id", None)
        if rec_id and rec_id in self._records:
            entry = self._records[rec_id]
            new_content = getattr(record, "content", entry.content)

            # If content changed, replace the node
            if new_content != entry.content:
                self._memory.remove_node(entry.node_id)
                ref = self._memory.thought(new_content)
                node = ref.node
                node.ext = node.ext or {}
                entry.node_id = ref.id
                entry.content = new_content
            else:
                node = self._memory.get_node(entry.node_id)
                if node:
                    node.ext = node.ext or {}

            entry.scope = getattr(record, "scope", entry.scope)
            entry.categories = getattr(record, "categories", entry.categories)
            entry.metadata = getattr(record, "metadata", entry.metadata)
            entry.importance = getattr(record, "importance", entry.importance)
            entry.embedding = getattr(record, "embedding", entry.embedding)
            entry.last_accessed = _dt_to_str(datetime.now(timezone.utc))

            # Sync ext data to node
            if node and node.ext is not None:
                node.ext.update({
                    "crewai_record_id": rec_id,
                    "scope": entry.scope,
                    "categories": entry.categories,
                    "metadata": entry.metadata,
                    "importance": entry.importance,
                    "last_accessed": entry.last_accessed,
                    "embedding": entry.embedding,
                })

    def get_record(self, record_id: str) -> Any | None:
        """Get a record by ID."""
        entry = self._records.get(record_id)
        if entry:
            self._memory.touch_nodes_session_scoped([entry.node_id])
            return entry.to_dict()
        return None

    def list_records(
        self,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Any]:
        """List records, optionally filtered by scope."""
        results = []
        for entry in self._records.values():
            if scope_prefix and not entry.scope.startswith(scope_prefix):
                continue
            results.append(entry.to_dict())
        return results[offset : offset + limit]

    def get_scope_info(self, scope: str) -> dict[str, Any]:
        """Get info about a scope."""
        records = [e for e in self._records.values() if e.scope.startswith(scope)]
        cats: set[str] = set()
        for r in records:
            cats.update(r.categories)

        return {
            "path": scope,
            "record_count": len(records),
            "categories": sorted(cats),
            "oldest_record": min((r.created_at for r in records), default=None),
            "newest_record": max((r.created_at for r in records), default=None),
            "child_scopes": sorted(set(
                r.scope for r in records if r.scope != scope and r.scope.startswith(scope)
            )),
        }

    def list_scopes(self, parent: str = "/") -> list[str]:
        """List child scopes under parent."""
        scopes: set[str] = set()
        for entry in self._records.values():
            if entry.scope.startswith(parent) and entry.scope != parent:
                scopes.add(entry.scope)
        return sorted(scopes)

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        """List categories with counts."""
        counts: dict[str, int] = {}
        for entry in self._records.values():
            if scope_prefix and not entry.scope.startswith(scope_prefix):
                continue
            for cat in entry.categories:
                counts[cat] = counts.get(cat, 0) + 1
        return counts

    def count(self, scope_prefix: str | None = None) -> int:
        """Count records."""
        if scope_prefix is None:
            return len(self._records)
        return sum(
            1 for e in self._records.values() if e.scope.startswith(scope_prefix)
        )

    def reset(self, scope_prefix: str | None = None) -> None:
        """Delete all records (or scoped subset). Removes nodes from graph."""
        if scope_prefix is None:
            for entry in self._records.values():
                self._memory.remove_node(entry.node_id)
            self._records.clear()
        else:
            to_delete = [
                k for k, v in self._records.items()
                if v.scope.startswith(scope_prefix)
            ]
            for k in to_delete:
                entry = self._records.pop(k)
                self._memory.remove_node(entry.node_id)

    # -- Async variants (delegate to sync) --

    async def asave(self, records: list[Any]) -> None:
        self.save(records)

    async def asearch(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[Any, float]]:
        return self.search(
            query_embedding, scope_prefix, categories, metadata_filter, limit, min_score
        )

    async def adelete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: Any | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        return self.delete(scope_prefix, categories, record_ids, older_than, metadata_filter)

    def save_to_disk(self) -> None:
        """Persist to disk."""
        self._memory.save()

    def close(self):
        """End the session: prune dormant nodes, save. Returns SessionWrapResult."""
        return self._memory.session_wrap()


class _RecordEntry:
    """Internal record storage."""

    __slots__ = (
        "node_id", "record_id", "content", "scope", "categories",
        "metadata", "importance", "created_at", "last_accessed",
        "embedding", "source", "private",
    )

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.record_id,
            "content": self.content,
            "scope": self.scope,
            "categories": self.categories,
            "metadata": self.metadata,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "embedding": self.embedding,
            "source": self.source,
            "private": self.private,
        }


def _dt_to_str(dt: Any) -> str:
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
