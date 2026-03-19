"""
FlowScript LangGraph Integration.

Implements LangGraph's BaseStore interface, making FlowScript memory
available as a drop-in store for LangGraph agents and LangMem.

Usage:
    from flowscript_agents.langgraph import FlowScriptStore

    store = FlowScriptStore("./agent-memory.json")
    # Use as LangGraph store — nodes stored as items, queries available via .memory
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from .memory import Memory, NodeRef


class FlowScriptStore(BaseStore):
    """LangGraph BaseStore backed by FlowScript reasoning memory.

    Each item stored maps to a FlowScript node. The store provides standard
    LangGraph get/put/search/delete operations, plus access to FlowScript's
    semantic queries (why, tensions, blocked, alternatives, whatIf) via
    the .memory property.

    Namespaces map to FlowScript node metadata:
    - Items are stored as nodes with namespace encoded in provenance
    - Search uses FlowScript's content matching
    - Full FlowScript query engine available via .memory.query

    For semantic queries (tensions, blocked, why, alternatives, whatIf),
    use resolve() to get a NodeRef and build relationships::

        db = store.resolve(("arch", "decisions"), "db_choice")
        cache = store.resolve(("arch", "decisions"), "cache_choice")
        cache.causes(db)
        db.tension_with(cache, axis="simplicity vs resilience")
        db.decide(rationale="Redis for both", on="2026-03-18")

        # Now semantic queries find these relationships
        tensions = store.memory.query.tensions()
    """

    def __init__(self, file_path: str | None = None) -> None:
        super().__init__()
        if file_path:
            self._memory = Memory.load_or_create(file_path)
        else:
            self._memory = Memory()
        # In-memory index: namespace+key → node_id + value
        self._items: dict[tuple[tuple[str, ...], str], _StoredItem] = {}
        # Rebuild index from loaded memory
        self._rebuild_index()
        # Start temporal session (resets touch dedup)
        self._memory.session_start()

    @property
    def memory(self) -> Memory:
        """Access the underlying FlowScript Memory for semantic queries.

        Example:
            tensions = store.memory.query.tensions()
            blocked = store.memory.query.blocked()
        """
        return self._memory

    def resolve(self, namespace: tuple[str, ...], key: str) -> NodeRef | None:
        """Resolve a stored item to a FlowScript NodeRef for semantic operations.

        Returns None if the item doesn't exist. Use the returned NodeRef to
        build relationships that power FlowScript's semantic queries::

            db = store.resolve(("arch", "decisions"), "db_choice")
            cache = store.resolve(("arch", "decisions"), "cache_choice")
            if db and cache:
                cache.causes(db)
                db.tension_with(cache, axis="simplicity vs resilience")
                db.decide(rationale="Redis for both")

            # Semantic queries now find these
            store.memory.query.tensions()
            store.memory.query.blocked()
        """
        stored = self._items.get((namespace, key))
        if stored is None:
            return None
        try:
            return self._memory.ref(stored.node_id)
        except KeyError:
            return None

    def _rebuild_index(self) -> None:
        """Rebuild the namespace/key index from loaded memory nodes."""
        for ref in self._memory.nodes:
            node = ref.node
            if node.ext and "langgraph_ns" in node.ext:
                ns = tuple(node.ext["langgraph_ns"])
                key = node.ext.get("langgraph_key", node.id)
                value = node.ext.get("langgraph_value", {"content": node.content})
                created = node.provenance.timestamp
                self._items[(ns, key)] = _StoredItem(
                    node_id=node.id,
                    namespace=ns,
                    key=key,
                    value=value,
                    created_at=_parse_dt(created),
                    updated_at=_parse_dt(created),
                )

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._handle_get(op))
            elif isinstance(op, PutOp):
                self._handle_put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(self._handle_search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._handle_list_namespaces(op))
            else:
                raise ValueError(f"Unknown op type: {type(op)}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        # Synchronous implementation is sufficient for file-based store
        return self.batch(ops)

    def _handle_get(self, op: GetOp) -> Item | None:
        stored = self._items.get((op.namespace, op.key))
        if stored is None:
            return None
        # Touch: retrieving an item is engagement
        self._memory._touch_nodes_session_scoped([stored.node_id])
        return Item(
            namespace=stored.namespace,
            key=stored.key,
            value=stored.value,
            created_at=stored.created_at,
            updated_at=stored.updated_at,
        )

    def _handle_put(self, op: PutOp) -> None:
        ns = op.namespace
        key = op.key

        if op.value is None:
            # Delete: remove from index AND from FlowScript graph
            stored = self._items.pop((ns, key), None)
            if stored:
                self._memory.remove_node(stored.node_id)
            return

        # Upsert: create or update
        existing = self._items.get((ns, key))
        now = datetime.now(timezone.utc)

        # Determine content for the FlowScript node
        content = op.value.get("content") or op.value.get("memory") or json.dumps(op.value)

        if existing:
            # Remove old node, create new one with updated content
            self._memory.remove_node(existing.node_id)
            ref = self._memory.thought(content)
            node = ref.node
            node.ext = node.ext or {}
            node.ext["langgraph_ns"] = list(ns)
            node.ext["langgraph_key"] = key
            node.ext["langgraph_value"] = op.value
            existing.node_id = ref.id
            existing.value = op.value
            existing.updated_at = now
        else:
            # Create new node
            ref = self._memory.thought(content)
            node = ref.node
            node.ext = node.ext or {}
            node.ext["langgraph_ns"] = list(ns)
            node.ext["langgraph_key"] = key
            node.ext["langgraph_value"] = op.value

            self._items[(ns, key)] = _StoredItem(
                node_id=ref.id,
                namespace=ns,
                key=key,
                value=op.value,
                created_at=now,
                updated_at=now,
            )

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        results: list[SearchItem] = []
        prefix = op.namespace_prefix

        for (ns, key), stored in self._items.items():
            # Check namespace prefix match
            if len(ns) < len(prefix) or ns[: len(prefix)] != prefix:
                continue

            # Apply filter
            if op.filter:
                if not _matches_filter(stored.value, op.filter):
                    continue

            # Score by query match (simple content matching)
            score = 0.0
            if op.query:
                content_str = json.dumps(stored.value).lower()
                query_lower = op.query.lower()
                if query_lower in content_str:
                    # Simple relevance: ratio of query length to content length
                    score = len(query_lower) / max(len(content_str), 1)
                else:
                    # No match — skip if query was specified
                    continue

            results.append(
                SearchItem(
                    namespace=stored.namespace,
                    key=stored.key,
                    value=stored.value,
                    created_at=stored.created_at,
                    updated_at=stored.updated_at,
                    score=score,
                )
            )

        # Sort by score descending, then by recency descending
        results.sort(key=lambda x: (-(x.score or 0), -x.updated_at.timestamp()))

        final = results[op.offset : op.offset + op.limit]

        # Touch found nodes — search engagement drives temporal graduation
        touched_ids = []
        for item in final:
            stored = self._items.get((item.namespace, item.key))
            if stored:
                touched_ids.append(stored.node_id)
        if touched_ids:
            self._memory._touch_nodes_session_scoped(touched_ids)

        return final

    def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        namespaces: set[tuple[str, ...]] = set()

        for (ns, _key) in self._items:
            # Apply max_depth truncation
            effective_ns = ns
            if op.max_depth is not None:
                effective_ns = ns[: op.max_depth]

            # Apply match conditions
            matches = True
            for cond in op.match_conditions:
                if cond.match_type == "prefix":
                    if len(effective_ns) < len(cond.path) or effective_ns[: len(cond.path)] != tuple(cond.path):
                        matches = False
                        break
                elif cond.match_type == "suffix":
                    if len(effective_ns) < len(cond.path) or effective_ns[-len(cond.path) :] != tuple(cond.path):
                        matches = False
                        break

            if matches:
                namespaces.add(effective_ns)

        sorted_ns = sorted(namespaces)
        return sorted_ns[op.offset : op.offset + op.limit]

    def save(self) -> None:
        """Persist the store to disk."""
        self._memory.save()

    def close(self) -> None:
        """End the session: prune dormant nodes, save, capture lifecycle stats."""
        return self._memory.session_wrap()


class _StoredItem:
    """Internal storage for items with their metadata."""

    __slots__ = ("node_id", "namespace", "key", "value", "created_at", "updated_at")

    def __init__(
        self,
        node_id: str,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        created_at: datetime,
        updated_at: datetime,
    ) -> None:
        self.node_id = node_id
        self.namespace = namespace
        self.key = key
        self.value = value
        self.created_at = created_at
        self.updated_at = updated_at


def _matches_filter(value: dict[str, Any], filter_dict: dict[str, Any]) -> bool:
    """Check if a value dict matches all filter criteria."""
    for k, v in filter_dict.items():
        if k not in value or value[k] != v:
            return False
    return True


def _parse_dt(s: str) -> datetime:
    """Parse ISO-8601 datetime string."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc)
