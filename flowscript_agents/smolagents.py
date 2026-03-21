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
from typing import Any, Optional, TYPE_CHECKING

from smolagents import Tool as _SmolBaseTool

from .memory import Memory, MemoryOptions, NodeRef

if TYPE_CHECKING:
    from .embeddings.providers import EmbeddingProvider
    from .embeddings.extract import ExtractFn
    from .embeddings.consolidate import ConsolidationProvider
    from .unified import UnifiedMemory


class FlowScriptMemoryTools:
    """FlowScript memory integration for smolagents.

    Creates Tool-compatible objects that smolagents agents can use to
    store, recall, and query persistent reasoning memory.

    Access FlowScript's semantic queries via the .memory property::

        tools = FlowScriptMemoryTools("./agent-memory.json")
        tensions = tools.memory.query.tensions()
        blocked = tools.memory.query.blocked()

    For vector-powered search and auto-extraction::

        from flowscript_agents.embeddings import OpenAIEmbeddings

        tools = FlowScriptMemoryTools(
            "./agent-memory.json",
            embedder=OpenAIEmbeddings(),
            llm=my_llm_fn,
        )

    For semantic queries, use resolve() to build relationships::

        ref = tools.resolve("chose Redis")
        if ref:
            ref.decide(rationale="Speed critical")
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
        self._memory.session_start()

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
            _StoreMemoryTool(self._memory, self._unified),
            _RecallMemoryTool(self._memory, self._unified),
            _QueryTensionsTool(self._memory),
            _QueryBlockedTool(self._memory),
            _GetMemoryContextTool(self._memory, self._unified),
            _QueryWhyTool(self._memory),
            _QueryWhatIfTool(self._memory),
            _QueryAlternativesTool(self._memory),
        ]

    def save(self) -> None:
        """Persist memory to disk. No-op if no file_path was provided (in-memory mode)."""
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


class _BaseFSTool(_SmolBaseTool):
    """Base for FlowScript smolagents tools.

    Inherits from smolagents.Tool for isinstance compatibility.
    Each subclass sets name, description, inputs, output_type as
    class attributes and implements forward().
    """

    name: str = ""
    description: str = ""
    inputs: dict = {}
    output_type: str = "string"

    def __init__(self, memory: Memory, unified: Any = None) -> None:
        super().__init__()
        self._memory = memory
        self._unified = unified

    def forward(self, **kwargs: Any) -> str:
        raise NotImplementedError


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
        # Use auto-extraction when available
        if self._unified and self._unified.extractor:
            result = self._unified.add(content, metadata={"category": category})
            return f"Stored in memory: [{category}] {result.nodes_created} nodes extracted"

        ref = self._memory.thought(content)
        if category:
            ref.node.ext = ref.node.ext or {}
            ref.node.ext["smolagents_category"] = category
        # Index for vector search if available
        if self._unified and self._unified.vector_index:
            self._unified.vector_index.index_node(ref.id)
        preview = content[:80] + ("..." if len(content) > 80 else "")
        return f"Stored in memory: [{category}] {preview}"


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
        # Use unified search when available (vector + keyword + temporal)
        if self._unified:
            unified_results = self._unified.search(query, top_k=limit)
            if not unified_results:
                return "No relevant memories found."
            self._memory.touch_nodes_session_scoped(
                [r.node_id for r in unified_results]
            )
            lines = []
            for r in unified_results:
                meta = self._memory.temporal_map.get(r.node_id)
                tier = meta.tier if meta else "current"
                freq = meta.frequency if meta else 1
                lines.append(f"[{tier}, freq={freq}] {r.content}")
            return "\n".join(lines)

        # Fallback: word-level search
        query_words = [w.lower() for w in query.split() if len(w) > 2]
        scored: list[tuple] = []
        if query_words:
            for node in self._memory._nodes.values():
                content_lower = node.content.lower()
                hits = sum(1 for w in query_words if w in content_lower)
                if hits > 0:
                    score = hits / len(query_words)
                    scored.append((NodeRef(self._memory, node), score))
            scored.sort(key=lambda x: -x[1])
        matches = [ref for ref, _ in scored[:limit]]
        if not matches:
            return "No relevant memories found."

        self._memory.touch_nodes_session_scoped([ref.id for ref in matches])

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
        if self._unified:
            ctx = self._unified.get_context(max_tokens=max_tokens)
            return ctx if ctx else "Memory is empty — no past context available."

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


class _QueryWhyTool(_BaseFSTool):
    name = "query_why"
    description = (
        "Trace why something happened — follow causal chains backward from a "
        "node to its root cause. Requires causal relationships to have been "
        "built via the resolve() API."
    )
    inputs = {
        "content": {
            "type": "string",
            "description": "Content of the memory node to trace causes for.",
        },
    }
    output_type = "string"

    def forward(self, content: str) -> str:
        matches = self._memory.find_nodes(content)
        if not matches:
            return f"No memory found matching '{content}'. Store it first, then build causal relationships with resolve()."
        result = self._memory.query.why(matches[0].id)
        return str(result)


class _QueryWhatIfTool(_BaseFSTool):
    name = "query_what_if"
    description = (
        "Explore downstream impact — what would be affected if this node "
        "changed? Traces causal chains forward. Requires relationships to "
        "have been built via the resolve() API."
    )
    inputs = {
        "content": {
            "type": "string",
            "description": "Content of the memory node to trace effects from.",
        },
    }
    output_type = "string"

    def forward(self, content: str) -> str:
        matches = self._memory.find_nodes(content)
        if not matches:
            return f"No memory found matching '{content}'. Store it first, then build causal relationships with resolve()."
        result = self._memory.query.what_if(matches[0].id)
        return str(result)


class _QueryAlternativesTool(_BaseFSTool):
    name = "query_alternatives"
    description = (
        "Find alternatives for a question or decision point. Returns "
        "alternatives with their states (decided, blocked, exploring, etc.). "
        "Requires alternatives to have been created via the resolve() API."
    )
    inputs = {
        "content": {
            "type": "string",
            "description": "Content of the question node to find alternatives for.",
        },
    }
    output_type = "string"

    def forward(self, content: str) -> str:
        matches = self._memory.find_nodes(content)
        if not matches:
            return f"No memory found matching '{content}'. Store a question first, then add alternatives with resolve()."
        result = self._memory.query.alternatives(matches[0].id)
        return str(result)
