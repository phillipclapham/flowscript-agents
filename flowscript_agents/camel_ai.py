"""
FlowScript CAMEL-AI Integration.

Provides FlowScript as an AgentMemory for CAMEL-AI's ChatAgent.
First reasoning memory provider for CAMEL-AI — existing memory
implementations are chat history (ChatHistoryMemory) and vector
retrieval (VectorDBMemory). FlowScript adds semantic queries:
tensions, blocked, why, alternatives, whatIf.

Usage:
    from flowscript_agents.camel_ai import FlowScriptCamelMemory

    memory = FlowScriptCamelMemory("./agent-memory.json")
    # Use with CAMEL ChatAgent:
    # agent = ChatAgent(system_message="...", memory=memory)

Note: Requires camel-ai: pip install flowscript-agents[camel-ai]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from .memory import Memory, NodeRef


@dataclass
class MemoryRecord:
    """Minimal CAMEL-compatible MemoryRecord.

    Duck-types the CAMEL MemoryRecord for environments where
    camel-ai is not installed. When camel-ai IS installed,
    real MemoryRecord objects from camel.memories.records work
    transparently.
    """
    content: str
    role: str = "assistant"
    uuid: str = ""
    timestamp: float = 0.0
    agent_id: str = ""
    extra_info: dict = field(default_factory=dict)


@dataclass
class ContextRecord:
    """Minimal CAMEL-compatible ContextRecord."""
    memory_record: MemoryRecord
    score: float = 0.0
    timestamp: float = 0.0


class FlowScriptCamelMemory:
    """CAMEL-AI AgentMemory backed by FlowScript reasoning memory.

    Implements the AgentMemory protocol (duck-typed) for CAMEL's ChatAgent.
    The key differentiation is in retrieve() — instead of returning raw
    chat history or vector matches, it returns FlowScript nodes ordered by
    temporal tier (proven knowledge first) with semantic query enrichment.

    Access FlowScript queries via .memory property::

        memory.memory.query.tensions()
        memory.memory.query.blocked()

    For semantic queries, use resolve()::

        ref = memory.resolve("chose Redis")
        if ref:
            ref.decide(rationale="Speed critical, native TTL")
    """

    def __init__(
        self,
        file_path: str | None = None,
        *,
        window_size: int | None = None,
        max_tokens: int = 4096,
    ) -> None:
        """Initialize FlowScript memory for CAMEL agents.

        Args:
            file_path: Path for persistent memory. None = in-memory.
            window_size: Max recent records to return from retrieve().
                None = all records (within token budget).
            max_tokens: Token budget for context assembly.
        """
        if file_path:
            self._memory = Memory.load_or_create(file_path)
        else:
            self._memory = Memory()
        self._file_path = file_path
        self._window_size = window_size
        self._max_tokens = max_tokens
        self._agent_id: str | None = None
        self._memory.session_start()

    @property
    def memory(self) -> Memory:
        """Access the underlying FlowScript Memory."""
        return self._memory

    @property
    def agent_id(self) -> str | None:
        return self._agent_id

    @agent_id.setter
    def agent_id(self, val: str | None) -> None:
        self._agent_id = val

    def resolve(self, content: str) -> NodeRef | None:
        """Resolve stored content to a FlowScript NodeRef.

        Searches for a node whose content contains the given string.
        Returns the first match, or None.
        """
        matches = self._memory.find_nodes(content)
        return matches[0] if matches else None

    # -- AgentMemory protocol methods --

    def retrieve(self) -> list[ContextRecord]:
        """Return scored records for context creation.

        This is where FlowScript differentiates from ChatHistoryMemory
        and VectorDBMemory. Records are:
        - Ordered by tier (proven > developing > current)
        - Scored by temporal frequency
        - Enriched with semantic query results (tensions, blocked)

        Returns:
            List of ContextRecord objects scored for context assembly.
        """
        records: list[ContextRecord] = []

        # Order by tier priority
        tier_order = {"proven": 0, "foundation": 0, "developing": 1, "current": 2}
        nodes_with_meta: list[tuple[NodeRef, str, int]] = []

        for ref in self._memory.nodes:
            meta = self._memory.temporal_map.get(ref.id)
            tier = meta.tier if meta else "current"
            freq = meta.frequency if meta else 1
            nodes_with_meta.append((ref, tier, freq))

        nodes_with_meta.sort(key=lambda x: (tier_order.get(x[1], 3), -x[2]))

        # Apply window size
        if self._window_size is not None:
            nodes_with_meta = nodes_with_meta[:self._window_size]

        # Touch retrieved nodes
        if nodes_with_meta:
            self._memory.touch_nodes_session_scoped(
                [ref.id for ref, _, _ in nodes_with_meta]
            )

        for ref, tier, freq in nodes_with_meta:
            ext = ref.node.ext or {}
            role = ext.get("camel_role", "assistant")

            # Score: proven=1.0, developing=0.7, current=0.4, plus frequency bonus
            base_score = {"proven": 1.0, "foundation": 1.0, "developing": 0.7, "current": 0.4}
            score = base_score.get(tier, 0.4) + min(freq * 0.05, 0.3)

            record = MemoryRecord(
                content=ref.content,
                role=role,
                uuid=ref.id[:8],
                timestamp=datetime.now(timezone.utc).timestamp(),
                agent_id=self._agent_id or "",
                extra_info={
                    "tier": tier,
                    "frequency": freq,
                    "source": "flowscript",
                },
            )
            records.append(ContextRecord(
                memory_record=record,
                score=score,
                timestamp=record.timestamp,
            ))

        # Append semantic insights as additional records
        try:
            tensions = self._memory.query.tensions()
            if tensions.metadata.get("total_tensions", 0) > 0:
                records.append(ContextRecord(
                    memory_record=MemoryRecord(
                        content=f"[ACTIVE TENSIONS] {tensions}",
                        role="system",
                        extra_info={"source": "flowscript-query", "type": "tensions"},
                    ),
                    score=0.95,  # High priority — tensions are actionable
                ))
        except Exception:
            pass

        try:
            blocked = self._memory.query.blocked()
            if blocked.blockers:
                records.append(ContextRecord(
                    memory_record=MemoryRecord(
                        content=f"[BLOCKED] {blocked}",
                        role="system",
                        extra_info={"source": "flowscript-query", "type": "blocked"},
                    ),
                    score=0.95,
                ))
        except Exception:
            pass

        return records

    def get_context_creator(self) -> Any:
        """Return a context creator.

        Returns a minimal context creator that respects token budget.
        When camel-ai is installed, users can provide their own
        ScoreBasedContextCreator for more sophisticated assembly.
        """
        return _SimpleContextCreator(max_tokens=self._max_tokens)

    def write_records(self, records: list[Any]) -> None:
        """Store records from agent conversation.

        Called by ChatAgent after each turn. Each record becomes a
        FlowScript node.

        Args:
            records: List of MemoryRecord-like objects.
        """
        prev_ref = None
        for record in records:
            content = getattr(record, "content", str(record))
            if not content:
                continue

            role = getattr(record, "role", "assistant")
            agent_id = getattr(record, "agent_id", self._agent_id or "")

            ref = self._memory.thought(content)
            node = ref.node
            node.ext = node.ext or {}
            node.ext.update({
                "camel_role": str(role),
                "camel_agent_id": agent_id,
            })

            extra = getattr(record, "extra_info", None)
            if extra and isinstance(extra, dict):
                node.ext["camel_extra"] = extra

            if prev_ref:
                prev_ref.then(ref)
            prev_ref = ref

        if self._file_path:
            self._memory.save()

    def write_record(self, record: Any) -> None:
        """Store a single record."""
        self.write_records([record])

    def clear(self) -> None:
        """Clear all stored records."""
        # Remove all nodes
        node_ids = [ref.id for ref in self._memory.nodes]
        for nid in node_ids:
            self._memory.remove_node(nid)

    def get_context(self) -> tuple[list[dict[str, Any]], int]:
        """Get chat context for the agent.

        Returns (messages, token_count) tuple. Messages are in
        OpenAI-compatible format.
        """
        records = self.retrieve()
        creator = self.get_context_creator()
        return creator.create_context(records)

    # -- Additional convenience methods --

    def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search memory with word-level matching."""
        query_words = [w.lower() for w in query.split() if len(w) > 2]
        scored: list[tuple[NodeRef, float]] = []
        if query_words:
            for node in self._memory._nodes.values():
                content_lower = node.content.lower()
                hits = sum(1 for w in query_words if w in content_lower)
                if hits > 0:
                    score = hits / len(query_words)
                    scored.append((NodeRef(self._memory, node), score))
            scored.sort(key=lambda x: -x[1])
        matches = [ref for ref, _ in scored[:limit]]
        if matches:
            self._memory.touch_nodes_session_scoped([ref.id for ref in matches])

        results = []
        for ref in matches:
            meta = self._memory.temporal_map.get(ref.id)
            results.append({
                "content": ref.content,
                "id": ref.id,
                "tier": meta.tier if meta else "current",
                "frequency": meta.frequency if meta else 1,
            })
        return results

    def save(self) -> None:
        """Persist to disk."""
        self._memory.save()

    def close(self):
        """End session: prune dormant, save. Returns SessionWrapResult."""
        return self._memory.session_wrap()


class _SimpleContextCreator:
    """Minimal context creator that assembles records into messages."""

    def __init__(self, max_tokens: int = 4096) -> None:
        self._max_tokens = max_tokens

    def create_context(
        self, records: list[ContextRecord]
    ) -> tuple[list[dict[str, Any]], int]:
        """Create context from records.

        Returns (messages, token_count) tuple.
        """
        # Sort by score descending
        sorted_records = sorted(records, key=lambda r: -r.score)

        messages: list[dict[str, Any]] = []
        char_budget = self._max_tokens * 4
        used = 0

        for ctx_rec in sorted_records:
            rec = ctx_rec.memory_record
            content = rec.content
            if used + len(content) > char_budget:
                break

            role = rec.role if rec.role in ("user", "assistant", "system") else "assistant"
            messages.append({
                "role": role,
                "content": content,
            })
            used += len(content)

        # Estimate tokens (rough: chars / 4)
        token_count = used // 4

        return messages, token_count
