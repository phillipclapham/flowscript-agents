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
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

from camel.memories.base import AgentMemory as _CamelAgentMemory

from .memory import Memory, MemoryOptions, NodeRef

if TYPE_CHECKING:
    from .embeddings.providers import EmbeddingProvider
    from .embeddings.extract import ExtractFn
    from .embeddings.consolidate import ConsolidationProvider
    from .unified import UnifiedMemory


@dataclass
class MemoryRecord:
    """Minimal CAMEL-compatible MemoryRecord.

    Duck-types the CAMEL MemoryRecord for environments where
    camel-ai is not installed. When camel-ai IS installed,
    real MemoryRecord objects from camel.memories.records work
    transparently.

    Includes role_at_backend for compatibility with ChatAgent's
    memory setter, which categorizes records by backend role.
    """
    content: str
    role: str = "assistant"
    role_at_backend: str = "assistant"
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


class FlowScriptCamelMemory(_CamelAgentMemory):
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
        embedder: EmbeddingProvider | None = None,
        llm: ExtractFn | None = None,
        consolidation_provider: ConsolidationProvider | None = None,
        options: MemoryOptions | None = None,
        **unified_kwargs: Any,
    ) -> None:
        """Initialize FlowScript memory for CAMEL agents.

        Args:
            file_path: Path for persistent memory. None = in-memory.
            window_size: Max recent records to return from retrieve().
                None = all records (within token budget).
            max_tokens: Token budget for context assembly.
            embedder: Embedding provider for vector search.
            llm: LLM function for auto-extraction.
            consolidation_provider: Tool-calling LLM for consolidation.
            options: MemoryOptions for the underlying Memory.
        """
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
        self._window_size = window_size
        self._max_tokens = max_tokens
        self._agent_id: str | None = None
        self._memory.session_start()

    @property
    def memory(self) -> Memory:
        """Access the underlying FlowScript Memory."""
        return self._memory

    @property
    def unified(self) -> UnifiedMemory | None:
        """Access UnifiedMemory for vector search + extraction. None if not configured."""
        return self._unified

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
                role_at_backend=role,
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

        When UnifiedMemory is configured with an LLM, uses auto-extraction
        for typed reasoning nodes. Otherwise stores as thought nodes.

        Args:
            records: List of MemoryRecord-like objects.
        """
        # Extract content from all records
        contents = []
        for record in records:
            msg = getattr(record, "message", None)
            if msg is not None:
                content = getattr(msg, "content", None) or str(msg)
            else:
                content = getattr(record, "content", str(record))
            if content:
                role_backend = getattr(record, "role_at_backend", None)
                if role_backend is not None:
                    role = getattr(role_backend, "value", str(role_backend))
                else:
                    role = getattr(record, "role", "assistant")
                contents.append((content, role))

        if not contents:
            return

        camel_meta = {"camel_agent_id": self._agent_id or ""}

        # Use auto-extraction when available
        if self._unified and self._unified.extractor:
            combined = "\n".join(f"[{role}] {c}" for c, role in contents)
            roles = list(set(role for _, role in contents))
            actor = "agent"
            result = self._unified.add(combined, metadata=camel_meta, actor=actor)
            # Tag extracted nodes with metadata + roles
            for node_id in result.node_ids:
                node = self._memory.get_node(node_id)
                if node:
                    node.ext = node.ext or {}
                    node.ext.update(camel_meta)
                    node.ext["camel_roles"] = roles
        else:
            prev_ref = None
            for content, role in contents:
                ref = self._memory.thought(content)
                node = ref.node
                node.ext = node.ext or {}
                node.ext.update({
                    "camel_role": str(role),
                    **camel_meta,
                })

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

    def write_record(self, record: Any) -> None:
        """Store a single record."""
        self.write_records([record])

    def clean_tool_calls(self) -> None:
        """Remove tool call messages from memory.

        Called by ChatAgent after tool execution. No-op for FlowScript
        since we store reasoning patterns, not raw tool call messages.
        """
        pass

    def clear(self) -> None:
        """Intentional no-op: preserve reasoning memory across clears.

        ChatAgent calls clear() during init and memory reassignment.
        This is correct for chat history buffers (FIFO, vector stores)
        but destructive for reasoning memory where nodes represent
        graduated knowledge with temporal intelligence.

        FlowScript's content-addressable design means the system message
        written after clear() will either deduplicate (same content → same
        node ID) or create a new node alongside existing ones.

        This is a genuine differentiator: CAMEL's built-in memories
        lose all history on clear. FlowScript preserves what matters.
        """
        # No-op: reasoning memory persists through agent lifecycle changes.
        # Nodes carry graduated temporal state (tier, frequency) that
        # represents learning — destroying it defeats the purpose.
        pass

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
        """Search memory for relevant content.

        Uses unified search (vector + keyword + temporal) when available,
        falls back to word-level matching.
        """
        if self._unified:
            unified_results = self._unified.search(query, top_k=limit)
            if unified_results:
                self._memory.touch_nodes_session_scoped(
                    [r.node_id for r in unified_results]
                )
            results = []
            for r in unified_results:
                meta = self._memory.temporal_map.get(r.node_id)
                results.append({
                    "content": r.content,
                    "id": r.node_id,
                    "tier": meta.tier if meta else "current",
                    "frequency": meta.frequency if meta else 1,
                    "score": r.combined_score,
                })
            return results

        # Fallback: word-level matching
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
        """Persist to disk. No-op if no file_path was provided (in-memory mode)."""
        if self._unified:
            self._unified.save()
        elif self._memory.file_path:
            self._memory.save()

    def close(self):
        """End session: prune dormant, save. Returns SessionWrapResult."""
        if self._unified:
            return self._unified.close()
        return self._memory.session_wrap()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise  # close() failure IS the error when no prior exception


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
