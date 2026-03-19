"""
FlowScript Haystack Integration.

Implements Haystack's MemoryStore protocol, making FlowScript memory
available as a semantic memory backend for Haystack agents and pipelines.

The only existing MemoryStore implementation is Mem0. FlowScript is the
second — and the first to provide queryable reasoning memory (tensions,
blocked, why, alternatives, whatIf) rather than vector retrieval.

Usage:
    from flowscript_agents.haystack import FlowScriptMemoryStore

    store = FlowScriptMemoryStore("./agent-memory.json")
    # Use with Haystack experimental Agent:
    # agent = Agent(
    #     chat_generator=generator,
    #     memory_store=store,
    # )

Note: Requires haystack-ai + haystack-experimental:
    pip install flowscript-agents[haystack]
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from .memory import Memory, NodeRef


class FlowScriptMemoryStore:
    """Haystack MemoryStore backed by FlowScript reasoning memory.

    Implements the MemoryStore protocol (duck-typed). Each memory maps
    to a FlowScript node with temporal intelligence.

    Access FlowScript queries via .memory property::

        store.memory.query.tensions()
        store.memory.query.blocked()

    For semantic queries, use resolve()::

        ref = store.resolve("memory_id_123")
        if ref:
            ref.decide(rationale="Based on load testing results")
    """

    def __init__(self, file_path: str | None = None) -> None:
        if file_path:
            self._memory = Memory.load_or_create(file_path)
        else:
            self._memory = Memory()
        self._file_path = file_path
        # Index: memory_id → node_id
        self._id_map: dict[str, str] = {}
        self._rebuild_index()
        self._memory.session_start()

    @property
    def memory(self) -> Memory:
        return self._memory

    def resolve(self, memory_id: str) -> NodeRef | None:
        """Resolve a memory ID to a FlowScript NodeRef for semantic operations.

        Use the returned NodeRef to build relationships::

            ref = store.resolve("mem_123")
            if ref:
                ref.block(reason="Waiting on stakeholder input")
                ref.tension_with(other_ref, axis="speed vs safety")

            store.memory.query.tensions()
        """
        node_id = self._id_map.get(memory_id)
        if node_id is None:
            return None
        try:
            return self._memory.ref(node_id)
        except KeyError:
            return None

    def _rebuild_index(self) -> None:
        """Rebuild memory_id → node_id index from loaded memory."""
        for ref in self._memory.nodes:
            node = ref.node
            if node.ext and "haystack_memory_id" in node.ext:
                self._id_map[node.ext["haystack_memory_id"]] = node.id

    # -- MemoryStore protocol methods --

    def add_memories(
        self,
        *,
        messages: list[Any],
        user_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Extract and store memories from chat messages.

        Each message becomes a FlowScript node. Messages are chained
        sequentially for causal tracking.

        Args:
            messages: List of ChatMessage objects or dicts.
            user_id: Optional user identifier for scoping.
            **kwargs: Additional args (agent_id, run_id, etc.)
        """
        prev_ref = None
        for msg in messages:
            content = _extract_content(msg)
            if not content:
                continue

            memory_id = str(uuid.uuid4())
            role = _extract_role(msg)

            ref = self._memory.thought(content)
            node = ref.node
            node.ext = node.ext or {}
            node.ext.update({
                "haystack_memory_id": memory_id,
                "haystack_role": role,
                "haystack_user_id": user_id,
            })
            # Store any extra kwargs
            for k, v in kwargs.items():
                node.ext[f"haystack_{k}"] = v

            self._id_map[memory_id] = ref.id

            if prev_ref:
                prev_ref.then(ref)
            prev_ref = ref

        if self._file_path:
            self._memory.save()

    def search_memories(
        self,
        *,
        query: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Search FlowScript memory for relevant content.

        Uses word-level matching for text queries. Filters by user_id
        when provided. Returns ChatMessage-compatible dicts.

        Args:
            query: Text to search for.
            filters: Additional filters (metadata matching).
            top_k: Maximum results.
            user_id: Filter to specific user's memories.
            **kwargs: Additional args.

        Returns:
            List of ChatMessage-compatible dicts with memory content.
        """
        # Word-level search
        scored: list[tuple[NodeRef, float]] = []
        if query:
            query_words = [w.lower() for w in query.split() if len(w) > 2]
            if query_words:
                for node in self._memory._nodes.values():
                    # User filter
                    ext = node.ext or {}
                    if user_id and ext.get("haystack_user_id") != user_id:
                        continue

                    content_lower = node.content.lower()
                    hits = sum(1 for w in query_words if w in content_lower)
                    if hits > 0:
                        score = hits / len(query_words)
                        scored.append((NodeRef(self._memory, node), score))
                scored.sort(key=lambda x: -x[1])
        else:
            # No query — return all (filtered by user)
            for node in self._memory._nodes.values():
                ext = node.ext or {}
                if user_id and ext.get("haystack_user_id") != user_id:
                    continue
                scored.append((NodeRef(self._memory, node), 0.5))

        matches = scored[:top_k]

        # Touch found nodes
        if matches:
            self._memory.touch_nodes_session_scoped(
                [ref.id for ref, _ in matches]
            )

        # Return as ChatMessage-compatible dicts
        results = []
        for ref, score in matches:
            node = ref.node
            ext = node.ext or {}
            meta = self._memory.temporal_map.get(ref.id)

            results.append({
                "content": node.content,
                "role": ext.get("haystack_role", "assistant"),
                "meta": {
                    "memory_id": ext.get("haystack_memory_id", ref.id),
                    "user_id": ext.get("haystack_user_id"),
                    "tier": meta.tier if meta else "current",
                    "frequency": meta.frequency if meta else 1,
                    "score": score,
                    "source": "flowscript",
                },
            })

        return results

    def delete_all_memories(
        self,
        *,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Delete all memories, optionally scoped to a user.

        Args:
            user_id: If provided, only delete this user's memories.
            **kwargs: Additional args.
        """
        if user_id is None:
            # Delete all
            to_delete = list(self._id_map.keys())
        else:
            to_delete = []
            for mem_id, node_id in self._id_map.items():
                try:
                    ref = self._memory.ref(node_id)
                    ext = ref.node.ext or {}
                    if ext.get("haystack_user_id") == user_id:
                        to_delete.append(mem_id)
                except KeyError:
                    to_delete.append(mem_id)  # orphaned entry

        for mem_id in to_delete:
            node_id = self._id_map.pop(mem_id, None)
            if node_id:
                self._memory.remove_node(node_id)

        if self._file_path:
            self._memory.save()

    def delete_memory(self, memory_id: str, **kwargs: Any) -> None:
        """Delete a single memory by ID.

        Args:
            memory_id: The memory to delete.
            **kwargs: Additional args.
        """
        node_id = self._id_map.pop(memory_id, None)
        if node_id:
            self._memory.remove_node(node_id)
            if self._file_path:
                self._memory.save()

    # -- Haystack serialization protocol --

    def to_dict(self) -> dict[str, Any]:
        """Serialize for Haystack pipeline serialization."""
        return {
            "type": "flowscript_agents.haystack.FlowScriptMemoryStore",
            "init_parameters": {
                "file_path": self._file_path,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowScriptMemoryStore":
        """Deserialize from Haystack pipeline serialization."""
        params = data.get("init_parameters", {})
        return cls(file_path=params.get("file_path"))

    # -- Additional convenience methods --

    def save(self) -> None:
        """Persist to disk."""
        self._memory.save()

    def close(self):
        """End session: prune dormant nodes, save. Returns SessionWrapResult."""
        return self._memory.session_wrap()


def _extract_content(msg: Any) -> str | None:
    """Extract text content from a ChatMessage or dict."""
    if isinstance(msg, dict):
        content = msg.get("content", msg.get("text", ""))
        return content if content else None

    # ChatMessage object
    content = getattr(msg, "content", None) or getattr(msg, "text", None)
    if isinstance(content, str) and content:
        return content
    return None


def _extract_role(msg: Any) -> str:
    """Extract role from a ChatMessage or dict."""
    if isinstance(msg, dict):
        return str(msg.get("role", "assistant"))
    role = getattr(msg, "role", "assistant")
    if hasattr(role, "value"):
        return role.value
    return str(role)
