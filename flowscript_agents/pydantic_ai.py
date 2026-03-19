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

if TYPE_CHECKING:
    pass  # RunContext[FlowScriptDeps] would go here when pydantic-ai is installed

from .memory import Memory, NodeRef


@dataclass
class FlowScriptDeps:
    """Pydantic AI dependency providing FlowScript reasoning memory.

    Pass as deps to any Pydantic AI agent. Access FlowScript's full
    query engine (tensions, blocked, why, alternatives, whatIf) plus
    temporal intelligence (graduation, garden, pruning).

    Usage::

        deps = FlowScriptDeps(file_path="./agent-memory.json")
        result = await agent.run("What should we use for caching?", deps=deps)

    For semantic queries, use resolve() to build relationships::

        ref = deps.resolve("chose Redis")
        if ref:
            ref.decide(rationale="Speed critical, native TTL")

        tensions = deps.memory.query.tensions()
    """

    file_path: str | None = None
    _memory: Memory = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.file_path:
            self._memory = Memory.load_or_create(self.file_path)
        else:
            self._memory = Memory()
        self._memory.session_start()

    @property
    def memory(self) -> Memory:
        """Access the underlying FlowScript Memory for semantic queries."""
        return self._memory

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

        Args:
            content: The content to remember.
            **metadata: Optional metadata stored in node.ext.

        Returns:
            NodeRef for building relationships (causes, tension_with, etc.)
        """
        ref = self._memory.thought(content)
        if metadata:
            ref.node.ext = ref.node.ext or {}
            ref.node.ext["pydantic_ai_meta"] = metadata
        return ref

    def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search memory for relevant content.

        Uses word-level matching: splits query into words, matches nodes
        containing any query word, scores by proportion of words matched.
        This handles natural language queries that won't match as exact
        substrings in longer content.

        Args:
            query: Text to search for.
            limit: Maximum results to return.

        Returns:
            List of dicts with 'content', 'id', 'tier', 'frequency' keys.
        """
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
        """
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
        """Persist memory to disk."""
        self._memory.save()

    def close(self):
        """End session: prune dormant nodes, save. Returns SessionWrapResult."""
        return self._memory.session_wrap()


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
    """

    # Note: ctx typed as Any to avoid requiring pydantic-ai at import time.
    # At runtime, ctx is RunContext[FlowScriptDeps]. Pydantic AI's @agent.tool()
    # wires dependency injection by positional convention, not type annotation.

    async def store_memory(ctx: Any, content: str, category: str = "observation") -> str:
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

    async def recall_memory(ctx: Any, query: str, limit: int = 5) -> str:
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

    async def query_tensions(ctx: Any) -> str:
        """Find active tradeoffs and tensions in memory.

        Returns:
            Summary of tensions grouped by axis.
        """
        tensions = ctx.deps.memory.query.tensions()
        if tensions.metadata.get("total_tensions", 0) == 0:
            return "No tensions found. Use store.resolve() to build tension relationships."
        return str(tensions)

    async def query_blocked(ctx: Any) -> str:
        """Find blockers and their downstream impact.

        Returns:
            Summary of blocked items with reasons and impact.
        """
        blocked = ctx.deps.memory.query.blocked()
        if not blocked.blockers:
            return "Nothing blocked. Use store.resolve() to mark items as blocked."
        return str(blocked)

    return [store_memory, recall_memory, query_tensions, query_blocked]
