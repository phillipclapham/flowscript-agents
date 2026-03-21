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
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from haystack.dataclasses import ChatMessage as HaystackChatMessage

from .memory import Memory, MemoryOptions, NodeRef

if TYPE_CHECKING:
    from .embeddings.providers import EmbeddingProvider
    from .embeddings.extract import ExtractFn
    from .embeddings.consolidate import ConsolidationProvider
    from .unified import UnifiedMemory


class FlowScriptMemoryStore:
    """Haystack MemoryStore backed by FlowScript reasoning memory.

    Implements the MemoryStore protocol (duck-typed). Each memory maps
    to a FlowScript node with temporal intelligence.

    For auto-extraction and vector search::

        from flowscript_agents.embeddings import OpenAIEmbeddings

        store = FlowScriptMemoryStore(
            "./agent-memory.json",
            embedder=OpenAIEmbeddings(),
            llm=my_llm_fn,
        )

    Access FlowScript queries via .memory property::

        store.memory.query.tensions()
        store.memory.query.blocked()

    For semantic queries, use resolve()::

        ref = store.resolve("memory_id_123")
        if ref:
            ref.decide(rationale="Based on load testing results")
    """

    def __init__(
        self,
        file_path: str | None = None,
        *,
        embedder: EmbeddingProvider | None = None,
        llm: ExtractFn | None = None,
        consolidation_provider: ConsolidationProvider | None = None,
        options: MemoryOptions | None = None,
        **unified_kwargs: Any,
    ) -> None:
        self._unified: UnifiedMemory | None = None
        if embedder or llm or consolidation_provider:
            from .unified import UnifiedMemory as _UM

            self._unified = _UM(
                file_path=file_path,
                embedder=embedder,
                llm=llm,
                consolidation_provider=consolidation_provider,
                options=options,
                **unified_kwargs,
            )
            self._memory = self._unified.memory
        else:
            if file_path:
                self._memory = Memory.load_or_create(file_path, options=options)
            else:
                self._memory = Memory(options=options)
        self._file_path = file_path
        # Index: memory_id → node_id
        self._id_map: dict[str, str] = {}
        self._rebuild_index()
        self._memory.session_start()

    @property
    def memory(self) -> Memory:
        return self._memory

    @property
    def unified(self) -> UnifiedMemory | None:
        """Access UnifiedMemory for vector search + extraction. None if not configured."""
        return self._unified

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

        When UnifiedMemory is configured with an LLM, uses auto-extraction
        to create typed reasoning nodes. Otherwise, stores each message
        as a thought node.

        Args:
            messages: List of ChatMessage objects or dicts.
            user_id: Optional user identifier for scoping.
            **kwargs: Additional args (agent_id, run_id, etc.)
        """
        haystack_meta = {"haystack_user_id": user_id}
        for k, v in kwargs.items():
            haystack_meta[f"haystack_{k}"] = v

        # Use auto-extraction when available
        if self._unified and self._unified.extractor:
            texts = []
            roles = []
            for msg in messages:
                content = _extract_content(msg)
                if content:
                    role = _extract_role(msg)
                    texts.append(f"[{role}] {content}")
                    roles.append(role)
            if texts:
                combined = "\n".join(texts)
                result = self._unified.add(combined, metadata=haystack_meta)
                # Tag extracted nodes and build id_map
                unique_roles = list(set(roles))
                for node_id in result.node_ids:
                    memory_id = str(uuid.uuid4())
                    node = self._memory.get_node(node_id)
                    if node:
                        node.ext = node.ext or {}
                        node.ext["haystack_memory_id"] = memory_id
                        node.ext["haystack_roles"] = unique_roles
                        node.ext.update(haystack_meta)
                    self._id_map[memory_id] = node_id
        else:
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
                })
                node.ext.update(haystack_meta)

                self._id_map[memory_id] = ref.id

                # Index for vector search if available
                if self._unified and self._unified.vector_index:
                    self._unified.vector_index.index_node(ref.id)

                if prev_ref:
                    prev_ref.then(ref)
                prev_ref = ref

        if self._file_path:
            if self._unified:
                self._unified.save()
            else:
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
        # Use unified search when available (vector + keyword + temporal)
        scored: list[tuple[NodeRef, float]] = []
        if query and self._unified:
            unified_results = self._unified.search(query, top_k=top_k * 2)
            for r in unified_results:
                try:
                    ref = self._memory.ref(r.node_id)
                    ext = ref.node.ext or {}
                    if user_id and ext.get("haystack_user_id") != user_id:
                        continue
                    scored.append((ref, r.combined_score))
                except KeyError:
                    continue
        elif query:
            # Fallback: word-level search
            query_words = [w.lower() for w in query.split() if len(w) > 2]
            if query_words:
                for node in self._memory._nodes.values():
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

        # Return as Haystack ChatMessage objects (Agent calls .text on results)
        results: list[HaystackChatMessage] = []
        for ref, score in matches:
            node = ref.node
            ext = node.ext or {}
            meta = self._memory.temporal_map.get(ref.id)

            role = ext.get("haystack_role", "assistant")
            memory_meta = {
                "memory_id": ext.get("haystack_memory_id", ref.id),
                "user_id": ext.get("haystack_user_id"),
                "tier": meta.tier if meta else "current",
                "frequency": meta.frequency if meta else 1,
                "score": score,
                "source": "flowscript",
            }

            if role == "user":
                msg = HaystackChatMessage.from_user(node.content)
            else:
                msg = HaystackChatMessage.from_assistant(node.content, meta=memory_meta)
            # Attach meta to user messages too
            if role == "user":
                msg.meta.update(memory_meta)

            results.append(msg)

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
        """Persist to disk. No-op if no file_path was provided (in-memory mode)."""
        if self._unified:
            self._unified.save()
        elif self._memory.file_path:
            self._memory.save()

    def close(self):
        """End session: prune dormant nodes, save. Returns SessionWrapResult."""
        if self._unified:
            return self._unified.close()
        return self._memory.session_wrap()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


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
