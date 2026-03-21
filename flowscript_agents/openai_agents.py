"""
FlowScript OpenAI Agents SDK Integration.

Implements the OpenAI Agents SDK Session protocol, making FlowScript memory
available as a session backend for OpenAI agents.

Usage:
    from flowscript_agents.openai_agents import FlowScriptSession

    session = FlowScriptSession("conversation_123", "./agent-memory.json")
    # Use with Runner:
    # result = await Runner.run(agent, "Hello", session=session)

Note: Requires openai-agents package: pip install flowscript-agents[openai-agents]
The Session protocol is for conversation history. For richer FlowScript
capabilities (compression, semantic queries, temporal tiers), access
session.memory directly or expose as agent tools.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, TYPE_CHECKING

_log = logging.getLogger(__name__)

from .memory import Memory, MemoryOptions, NodeRef

if TYPE_CHECKING:
    from .embeddings.providers import EmbeddingProvider
    from .embeddings.extract import ExtractFn
    from .embeddings.consolidate import ConsolidationProvider
    from .unified import UnifiedMemory


class FlowScriptSession:
    """OpenAI Agents SDK Session backed by FlowScript reasoning memory.

    Implements the 4-method Session protocol:
    - get_items(limit?) → conversation history
    - add_items(items) → store new conversation items
    - pop_item() → remove and return last item
    - clear_session() → clear all items

    For vector-powered search, provide an embedder::

        from flowscript_agents.embeddings import OpenAIEmbeddings

        session = FlowScriptSession(
            "conv_123",
            "./agent-memory.json",
            embedder=OpenAIEmbeddings(),
        )

    Access FlowScript queries via .memory property:
        session.memory.query.tensions()
        session.memory.query.blocked()

    Note: ``llm`` and ``consolidation_provider`` create the underlying
    UnifiedMemory infrastructure but are NOT invoked by add_items/get_items.
    This adapter stores conversation items as-is (session protocol).
    To use auto-extraction and consolidation, call ``session.unified.add()``
    directly, or use the Pydantic AI adapter whose ``store()`` method
    routes through the full pipeline.
    """

    def __init__(
        self,
        session_id: str,
        file_path: str | None = None,
        session_settings: Any = None,
        *,
        embedder: EmbeddingProvider | None = None,
        llm: ExtractFn | None = None,
        consolidation_provider: ConsolidationProvider | None = None,
        options: MemoryOptions | None = None,
        **unified_kwargs: Any,
    ) -> None:
        self.session_id = session_id
        self.session_settings = session_settings

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

        # Ordered list of conversation items for this session
        self._items: list[dict[str, Any]] = []
        self._rebuild_items()
        # Start temporal session
        self._memory.session_start()

    @property
    def memory(self) -> Memory:
        return self._memory

    @property
    def unified(self) -> UnifiedMemory | None:
        """Access UnifiedMemory for vector search + extraction. None if not configured."""
        return self._unified

    def resolve(self, content: str) -> NodeRef | None:
        """Resolve a conversation item to a FlowScript NodeRef by content match.

        Searches for a node whose content contains the given string.
        Returns the first match, or None if not found. Use the returned
        NodeRef to build relationships for semantic queries::

            ref = session.resolve("chose Redis for sessions")
            if ref:
                ref.decide(rationale="Speed critical, native TTL")

            session.memory.query.blocked()
        """
        matches = self._memory.find_nodes(content)
        return matches[0] if matches else None

    def _rebuild_items(self) -> None:
        """Rebuild item list from loaded memory nodes."""
        items_with_order: list[tuple[int, dict[str, Any]]] = []
        for ref in self._memory.nodes:
            node = ref.node
            ext = node.ext or {}
            if ext.get("oai_session_id") == self.session_id:
                order = ext.get("oai_order", 0)
                item = ext.get("oai_item", {"role": "user", "content": node.content})
                items_with_order.append((order, item))

        items_with_order.sort(key=lambda x: x[0])
        self._items = [item for _, item in items_with_order]

    async def get_items(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get conversation items, optionally limited."""
        items = self._items[-limit:] if limit is not None else self._items
        # Touch retrieved nodes — retrieval is engagement
        touched_ids = []
        for ref in self._memory.nodes:
            ext = ref.node.ext or {}
            if ext.get("oai_session_id") == self.session_id:
                touched_ids.append(ref.id)
        if touched_ids:
            self._memory.touch_nodes_session_scoped(touched_ids)
        return list(items)

    async def add_items(self, items: list[dict[str, Any]]) -> None:
        """Add conversation items to the session."""
        base_order = len(self._items)
        for i, item in enumerate(items):
            content = _extract_item_content(item)
            if content:
                ref = self._memory.thought(f"[{self.session_id}] {content}")
                node = ref.node
                node.ext = node.ext or {}
                node.ext.update({
                    "oai_session_id": self.session_id,
                    "oai_order": base_order + i,
                    "oai_item": item,
                })
                # Index for vector search if available
                if self._unified and self._unified.vector_index:
                    try:
                        self._unified.vector_index.index_node(ref.id)
                    except Exception as exc:
                        _log.warning("Failed to index node %s: %s", ref.id[:12], exc)

            self._items.append(item)

        if self._file_path:
            if self._unified:
                self._unified.save()
            else:
                self._memory.save()

    async def pop_item(self) -> dict[str, Any] | None:
        """Remove and return the most recent item."""
        if not self._items:
            return None
        item = self._items.pop()

        # Remove the corresponding FlowScript node + its vector
        content = _extract_item_content(item)
        if content:
            node_content = f"[{self.session_id}] {content}"
            matches = self._memory.find_nodes(node_content)
            for ref in matches:
                if ref.content == node_content:
                    if self._unified and self._unified.vector_index:
                        self._unified.vector_index.remove_node(ref.id)
                    self._memory.remove_node(ref.id)
                    break

        if self._file_path:
            if self._unified:
                self._unified.save()
            else:
                self._memory.save()
        return item

    async def clear_session(self) -> None:
        """Clear all items for this session. Removes nodes from graph."""
        self._items.clear()
        # Remove session-specific nodes from the FlowScript graph
        to_remove = []
        for ref in self._memory.nodes:
            ext = ref.node.ext or {}
            if ext.get("oai_session_id") == self.session_id:
                to_remove.append(ref.id)
        for node_id in to_remove:
            if self._unified and self._unified.vector_index:
                self._unified.vector_index.remove_node(node_id)
            self._memory.remove_node(node_id)

        if self._file_path:
            if self._unified:
                self._unified.save()
            else:
                self._memory.save()

    def save(self) -> None:
        """Persist to disk. No-op if no file_path was provided (in-memory mode)."""
        if self._unified:
            self._unified.save()
        elif self._memory.file_path:
            self._memory.save()

    def close(self):
        """End the session: prune dormant nodes, save. Returns SessionWrapResult."""
        if self._unified:
            return self._unified.close()
        return self._memory.session_wrap()


def _extract_item_content(item: dict[str, Any]) -> str | None:
    """Extract text content from an OpenAI response input item."""
    # Standard message format
    content = item.get("content")
    if isinstance(content, str):
        return content

    # Content array format
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "input_text":
                texts.append(part.get("text", ""))
            elif isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        return " ".join(texts) if texts else None

    # Tool output format
    output = item.get("output")
    if isinstance(output, str):
        return output

    # Fallback
    role = item.get("role", "")
    return f"[{role} message]" if role else None
