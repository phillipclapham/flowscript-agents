"""
FlowScript Pydantic AI Integration.

Provides FlowScript memory as a dependency + toolset for Pydantic AI agents.
First-mover: no other memory provider has a Pydantic AI integration.

Usage:
    from flowscript_agents.pydantic_ai import FlowScriptDeps, create_memory_tools

    deps = FlowScriptDeps(file_path="./agent-memory.json")
    agent = Agent('anthropic:claude-sonnet-4-6', deps_type=FlowScriptDeps)

    # Register memory tools
    for tool_fn in create_memory_tools():
        agent.tool()(tool_fn)

    result = await agent.run("...", deps=deps)

Note: Requires pydantic-ai package: pip install flowscript-agents[pydantic-ai]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from pydantic_ai import RunContext

from .memory import Memory, MemoryOptions, NodeRef

if TYPE_CHECKING:
    from .embeddings.providers import EmbeddingProvider
    from .embeddings.extract import ExtractFn
    from .embeddings.consolidate import ConsolidationProvider
    from .unified import UnifiedMemory


@dataclass
class FlowScriptDeps:
    """Pydantic AI dependency providing FlowScript reasoning memory.

    Pass as deps to any Pydantic AI agent. Access FlowScript's full
    query engine (tensions, blocked, why, alternatives, whatIf) plus
    temporal intelligence (graduation, garden, pruning).

    Usage::

        deps = FlowScriptDeps(file_path="./agent-memory.json")
        result = await agent.run("What should we use for caching?", deps=deps)

    For vector-powered search and auto-extraction::

        from flowscript_agents.embeddings import OpenAIEmbeddings

        deps = FlowScriptDeps(
            file_path="./agent-memory.json",
            embedder=OpenAIEmbeddings(),
            llm=my_llm_fn,
        )
        # store() now auto-extracts typed reasoning
        # recall() now uses vector + keyword + temporal ranking

    For semantic queries, use resolve() to build relationships::

        ref = deps.resolve("chose Redis")
        if ref:
            ref.decide(rationale="Speed critical, native TTL")

        tensions = deps.memory.query.tensions()
    """

    file_path: str | None = None
    embedder: Any = field(default=None, repr=False)
    llm: Any = field(default=None, repr=False)
    consolidation_provider: Any = field(default=None, repr=False)
    options: Any = field(default=None, repr=False)
    unified_kwargs: dict = field(default_factory=dict, repr=False)
    _memory: Memory = field(init=False, repr=False)
    _unified: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.embedder or self.llm or self.consolidation_provider:
            from .unified import UnifiedMemory as _UM

            self._unified = _UM(
                file_path=self.file_path,
                embedder=self.embedder,
                llm=self.llm,
                consolidation_provider=self.consolidation_provider,
                options=self.options,
                **self.unified_kwargs,
            )
            self._memory = self._unified.memory
        else:
            if self.file_path:
                self._memory = Memory.load_or_create(self.file_path, options=self.options)
            else:
                self._memory = Memory(options=self.options)
        self._memory.session_start()
        self._memory.set_adapter_context("pydantic_ai", "FlowScriptDeps", "init")

    @property
    def memory(self) -> Memory:
        """Access the underlying FlowScript Memory for semantic queries."""
        return self._memory

    @property
    def unified(self) -> UnifiedMemory | None:
        """Access UnifiedMemory for vector search + extraction. None if not configured."""
        return self._unified

    def resolve(self, content: str) -> NodeRef | None:
        """Resolve stored content to a FlowScript NodeRef for semantic operations.

        Searches for a node whose content contains the given string.
        Returns the first match, or None. Use the returned NodeRef to
        build relationships that power semantic queries::

            ref = deps.resolve("chose Redis for sessions")
            if ref:
                ref.decide(rationale="Speed critical")
                ref.tension_with(other_ref, axis="cost vs speed")

            deps.memory.query.tensions()
        """
        matches = self._memory.find_nodes(content)
        return matches[0] if matches else None

    def store(self, content: str, **metadata: Any) -> NodeRef:
        """Store a thought in memory. Returns NodeRef for chaining.

        When UnifiedMemory is configured with an LLM, uses auto-extraction
        to create typed reasoning nodes (decisions, tensions, etc.) instead
        of a single thought node. Returns a NodeRef to the first extracted
        node for method chaining (.decide(), .block(), .tension_with(), etc.).

        Args:
            content: The content to remember.
            **metadata: Optional metadata stored in node.ext.

        Returns:
            NodeRef for building relationships (causes, tension_with, etc.)
        """
        self._memory.set_adapter_operation("store")
        if self._unified and self._unified.extractor:
            result = self._unified.add(content, metadata=metadata if metadata else None)
            # Return first extracted node for chaining
            if result.node_ids:
                try:
                    return self._memory.ref(result.node_ids[0])
                except KeyError:
                    pass
            # Fallback: create simple thought if extraction produced nothing
            return self._memory.thought(content)

        ref = self._memory.thought(content)
        if metadata:
            ref.node.ext = ref.node.ext or {}
            ref.node.ext["pydantic_ai_meta"] = metadata
        # Index for vector search if available
        if self._unified and self._unified.vector_index:
            self._unified.vector_index.index_node(ref.id)
        return ref

    def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search memory for relevant content.

        When UnifiedMemory is configured, uses vector + keyword + temporal
        ranking for much better search quality. Falls back to word-level
        matching without embedder.

        Args:
            query: Text to search for.
            limit: Maximum results to return.

        Returns:
            List of dicts with 'content', 'id', 'tier', 'frequency' keys.
        """
        self._memory.set_adapter_operation("recall")
        # Use unified search when available (vector + keyword + temporal)
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

    def get_context(self, max_tokens: int = 4000) -> str:
        """Get memory context formatted for prompt injection.

        Returns a structured summary of memory nodes with tier info.
        Use in @agent.instructions for automatic context injection.
        When UnifiedMemory is configured, uses its get_context method.
        """
        if self._unified:
            return self._unified.get_context(max_tokens=max_tokens)

        if self._memory.size == 0:
            return ""

        lines = []
        char_budget = max_tokens * 4  # rough chars-to-tokens
        used = 0
        for ref in self._memory.nodes:
            meta = self._memory.temporal_map.get(ref.id)
            tier = meta.tier if meta else "current"
            freq = meta.frequency if meta else 1
            line = f"[{tier}, freq={freq}] {ref.content}"
            if used + len(line) > char_budget:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    def save(self) -> None:
        """Persist memory to disk. No-op if no file_path was provided (in-memory mode)."""
        if self._unified:
            self._unified.save()
        elif self._memory.file_path:
            self._memory.save()

    def close(self):
        """End session: prune dormant nodes, save. Returns SessionWrapResult."""
        try:
            if self._unified:
                return self._unified.close()
            return self._memory.session_wrap()
        finally:
            self._memory.clear_adapter_context()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise  # close() failure IS the error when no prior exception


def create_memory_tools() -> list:
    """Create tool functions for FlowScript memory operations.

    Returns a list of async functions compatible with Pydantic AI's
    @agent.tool() decorator. Each function accepts RunContext[FlowScriptDeps].

    Usage::

        from pydantic_ai import Agent
        from flowscript_agents.pydantic_ai import FlowScriptDeps, create_memory_tools

        agent = Agent('anthropic:claude-sonnet-4-6', deps_type=FlowScriptDeps)
        for tool_fn in create_memory_tools():
            agent.tool()(tool_fn)

    Tools created:
        - store_memory: Store observations, decisions, or insights
        - recall_memory: Search for relevant past context
        - query_tensions: Find active tradeoffs and tensions
        - query_blocked: Find blockers and their downstream impact
        - query_why: Trace causal chains backward (why did this happen?)
        - query_what_if: Explore downstream impact (what would change?)
        - query_alternatives: Find alternatives for decision points
    """

    async def store_memory(ctx: RunContext[FlowScriptDeps], content: str, category: str = "observation") -> str:
        """Store an observation, decision, or insight in persistent memory.

        Args:
            ctx: RunContext with FlowScriptDeps.
            content: What to remember.
            category: Type of memory (observation, decision, concern, insight).

        Returns:
            Confirmation message with node ID.
        """
        ref = ctx.deps.store(content, category=category)
        return f"Stored: {ref.id[:8]}... [{category}]"

    async def recall_memory(ctx: RunContext[FlowScriptDeps], query: str, limit: int = 5) -> str:
        """Search persistent memory for relevant past context.

        Args:
            ctx: RunContext with FlowScriptDeps.
            query: What to search for.
            limit: Maximum results.

        Returns:
            Formatted memory results with tier and frequency info.
        """
        results = ctx.deps.recall(query, limit=limit)
        if not results:
            return "No relevant memories found."
        lines = []
        for r in results:
            lines.append(f"[{r['tier']}, freq={r['frequency']}] {r['content']}")
        return "\n".join(lines)

    async def query_tensions(ctx: RunContext[FlowScriptDeps]) -> str:
        """Find active tradeoffs and tensions in memory.

        Returns:
            Summary of tensions grouped by axis.
        """
        tensions = ctx.deps.memory.query.tensions()
        if tensions.metadata.get("total_tensions", 0) == 0:
            return "No tensions found. Use store.resolve() to build tension relationships."
        return str(tensions)

    async def query_blocked(ctx: RunContext[FlowScriptDeps]) -> str:
        """Find blockers and their downstream impact.

        Returns:
            Summary of blocked items with reasons and impact.
        """
        blocked = ctx.deps.memory.query.blocked()
        if not blocked.blockers:
            return "Nothing blocked. Use store.resolve() to mark items as blocked."
        return str(blocked)

    async def query_why(ctx: RunContext[FlowScriptDeps], node_content: str) -> str:
        """Trace why something happened — follow causal chains backward.

        Args:
            ctx: RunContext with FlowScriptDeps.
            node_content: Content of the node to trace causes for.

        Returns:
            Causal chain from the node back to its root cause.
        """
        ref = ctx.deps.resolve(node_content)
        if not ref:
            return f"No memory found matching '{node_content}'. Store it first, then build causal relationships with resolve()."
        result = ctx.deps.memory.query.why(ref.id)
        return str(result)

    async def query_what_if(ctx: RunContext[FlowScriptDeps], node_content: str) -> str:
        """Explore downstream impact — what would be affected if this changed?

        Args:
            ctx: RunContext with FlowScriptDeps.
            node_content: Content of the node to trace effects from.

        Returns:
            Impact tree showing downstream consequences.
        """
        ref = ctx.deps.resolve(node_content)
        if not ref:
            return f"No memory found matching '{node_content}'. Store it first, then build causal relationships with resolve()."
        result = ctx.deps.memory.query.what_if(ref.id)
        return str(result)

    async def query_alternatives(ctx: RunContext[FlowScriptDeps], question_content: str) -> str:
        """Find alternatives for a question or decision point.

        Args:
            ctx: RunContext with FlowScriptDeps.
            question_content: Content of the question node to find alternatives for.

        Returns:
            List of alternatives with their states (decided, blocked, exploring, etc.)
        """
        ref = ctx.deps.resolve(question_content)
        if not ref:
            return f"No memory found matching '{question_content}'. Store a question first, then add alternatives with resolve()."
        result = ctx.deps.memory.query.alternatives(ref.id)
        return str(result)

    return [store_memory, recall_memory, query_tensions, query_blocked, query_why, query_what_if, query_alternatives]
