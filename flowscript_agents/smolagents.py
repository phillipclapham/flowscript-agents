"""
FlowScript smolagents Integration.

Provides FlowScript memory as Tool subclasses for HuggingFace smolagents.
Compatible with both CodeAgent and ToolCallingAgent.

Usage:
    from flowscript_agents.smolagents import FlowScriptMemoryTools

    memory_tools = FlowScriptMemoryTools("./agent-memory.json")
    agent = CodeAgent(tools=memory_tools.tools(), model=model)

    # After the run
    memory_tools.close()  # prune dormant, save

Note: Requires smolagents package: pip install flowscript-agents[smolagents]
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .memory import Memory, NodeRef


class FlowScriptMemoryTools:
    """FlowScript memory integration for smolagents.

    Creates Tool-compatible objects that smolagents agents can use to
    store, recall, and query persistent reasoning memory.

    Access FlowScript's semantic queries via the .memory property::

        tools = FlowScriptMemoryTools("./agent-memory.json")
        tensions = tools.memory.query.tensions()
        blocked = tools.memory.query.blocked()

    For semantic queries, use resolve() to build relationships::

        ref = tools.resolve("chose Redis")
        if ref:
            ref.decide(rationale="Speed critical")
    """

    def __init__(self, file_path: str | None = None) -> None:
        if file_path:
            self._memory = Memory.load_or_create(file_path)
        else:
            self._memory = Memory()
        self._file_path = file_path
        self._memory.session_start()

    @property
    def memory(self) -> Memory:
        """Access the underlying FlowScript Memory for semantic queries."""
        return self._memory

    def resolve(self, content: str) -> NodeRef | None:
        """Resolve stored content to a FlowScript NodeRef for semantic operations.

        Searches for a node whose content contains the given string.
        Returns the first match, or None. Use the returned NodeRef to
        build relationships that power semantic queries.
        """
        matches = self._memory.find_nodes(content)
        return matches[0] if matches else None

    def tools(self) -> list:
        """Return list of smolagents-compatible tool instances.

        Each tool follows the smolagents Tool protocol (name, description,
        inputs, output_type, forward method). Pass to CodeAgent or
        ToolCallingAgent via the tools parameter.
        """
        return [
            _StoreMemoryTool(self._memory),
            _RecallMemoryTool(self._memory),
            _QueryTensionsTool(self._memory),
            _QueryBlockedTool(self._memory),
            _GetMemoryContextTool(self._memory),
        ]

    def save(self) -> None:
        """Persist memory to disk."""
        self._memory.save()

    def close(self) -> None:
        """End session: prune dormant, save, return lifecycle stats."""
        return self._memory.session_wrap()


class _BaseFSTool:
    """Base for FlowScript smolagents tools.

    Implements the smolagents Tool protocol without importing smolagents
    (which may not be installed). smolagents uses duck-typing for tools —
    any object with name, description, inputs, output_type, and forward()
    works as a tool.
    """

    name: str = ""
    description: str = ""
    inputs: dict = {}
    output_type: str = "string"

    def __init__(self, memory: Memory) -> None:
        self._memory = memory

    def forward(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def __call__(self, **kwargs: Any) -> str:
        return self.forward(**kwargs)


class _StoreMemoryTool(_BaseFSTool):
    name = "store_memory"
    description = (
        "Store an observation, decision, or insight in persistent reasoning memory. "
        "Use this to remember important context across sessions."
    )
    inputs = {
        "content": {
            "type": "string",
            "description": "What to remember — observation, decision, insight, or concern.",
        },
        "category": {
            "type": "string",
            "description": "Type of memory: observation, decision, concern, or insight.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, content: str, category: str = "observation") -> str:
        ref = self._memory.thought(content)
        if category:
            ref.node.ext = ref.node.ext or {}
            ref.node.ext["smolagents_category"] = category
        return f"Stored in memory: [{category}] {content[:80]}..."


class _RecallMemoryTool(_BaseFSTool):
    name = "recall_memory"
    description = (
        "Search persistent memory for relevant past context. Returns memories "
        "with their tier (current/developing/proven) and engagement frequency."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "What to search for in memory.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum results to return.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, query: str, limit: int = 5) -> str:
        matches = self._memory.find_nodes(query)[:limit]
        if not matches:
            return "No relevant memories found."

        self._memory._touch_nodes_session_scoped([ref.id for ref in matches])

        lines = []
        for ref in matches:
            meta = self._memory.temporal_map.get(ref.id)
            tier = meta.tier if meta else "current"
            freq = meta.frequency if meta else 1
            lines.append(f"[{tier}, freq={freq}] {ref.content}")
        return "\n".join(lines)


class _QueryTensionsTool(_BaseFSTool):
    name = "query_tensions"
    description = (
        "Find active tradeoffs and tensions in memory. Returns tensions "
        "grouped by axis (e.g., 'speed vs safety'). Requires relationships "
        "to have been built via the resolve() API."
    )
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        tensions = self._memory.query.tensions()
        if tensions.metadata.get("total_tensions", 0) == 0:
            return "No tensions found in memory."
        return str(tensions)


class _QueryBlockedTool(_BaseFSTool):
    name = "query_blocked"
    description = (
        "Find blockers and their downstream impact in memory. Returns "
        "blocked items with reasons and affected dependencies."
    )
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        blocked = self._memory.query.blocked()
        if not blocked.blockers:
            return "Nothing blocked in memory."
        return str(blocked)


class _GetMemoryContextTool(_BaseFSTool):
    name = "get_memory_context"
    description = (
        "Get a summary of all persistent memory, formatted and token-budgeted. "
        "Use at the start of a session to orient on past context."
    )
    inputs = {
        "max_tokens": {
            "type": "integer",
            "description": "Maximum tokens for the context summary.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, max_tokens: int = 4000) -> str:
        if self._memory.size == 0:
            return "Memory is empty — no past context available."

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
        return "\n".join(lines) if lines else "Memory is empty — no past context available."
