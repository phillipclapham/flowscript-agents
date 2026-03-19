"""
FlowScript LlamaIndex Integration.

Implements LlamaIndex's BaseMemoryBlock, making FlowScript memory
available as a composable memory block for LlamaIndex agents.

Usage:
    from flowscript_agents.llamaindex import FlowScriptMemoryBlock

    block = FlowScriptMemoryBlock(file_path="./agent-memory.json")
    memory = Memory.from_defaults(
        session_id="my_session",
        memory_blocks=[block],
    )
    agent = FunctionAgent(llm=llm, tools=tools)
    response = await agent.run("Hello", memory=memory)

The block stores flushed messages as FlowScript nodes with temporal
intelligence. Returns semantic context (tensions, blocked items, proven
knowledge) alongside content matches.

Note: Requires llama-index-core: pip install flowscript-agents[llamaindex]
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import ConfigDict, PrivateAttr

from llama_index.core.memory import BaseMemoryBlock

from .memory import Memory, NodeRef


class FlowScriptMemoryBlock(BaseMemoryBlock[str]):
    """LlamaIndex BaseMemoryBlock backed by FlowScript reasoning memory.

    Inherits from BaseMemoryBlock[str] for full compatibility with
    LlamaIndex's Memory system (Memory.from_defaults, FunctionAgent, etc.).

    The block participates in LlamaIndex's flush pipeline:
    - Short-term messages overflow → _aput() stores them as FlowScript nodes
    - Agent needs context → _aget() returns semantic memory summary

    Access FlowScript queries via .memory property::

        block.memory.query.tensions()
        block.memory.query.blocked()

    For semantic queries, use resolve()::

        ref = block.resolve("chose Redis")
        if ref:
            ref.decide(rationale="Speed critical")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Private attributes (not Pydantic fields — internal state)
    _memory: Memory = PrivateAttr()
    _file_path: Optional[str] = PrivateAttr(default=None)
    _max_tokens: int = PrivateAttr(default=4000)
    _include_queries: bool = PrivateAttr(default=True)

    def __init__(
        self,
        file_path: str | None = None,
        *,
        name: str = "flowscript_reasoning",
        max_tokens: int = 4000,
        include_queries: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize FlowScript memory block.

        Args:
            file_path: Path for persistent memory. None = in-memory only.
            name: Block name (appears in memory XML tags).
            max_tokens: Token budget for context returned by _aget().
            include_queries: Whether to append semantic query results
                (tensions, blocked) to the context output.
        """
        super().__init__(
            name=name,
            description=(
                "Long-term reasoning memory with temporal intelligence. "
                "Tracks decision patterns, tensions, and blockers across sessions."
            ),
            priority=1,
            accept_short_term_memory=True,
            **kwargs,
        )
        self._max_tokens = max_tokens
        self._include_queries = include_queries
        self._file_path = file_path

        if file_path:
            self._memory = Memory.load_or_create(file_path)
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
        Returns the first match, or None.
        """
        matches = self._memory.find_nodes(content)
        return matches[0] if matches else None

    # -- BaseMemoryBlock protocol methods --

    async def _aget(
        self,
        messages: Optional[List[Any]] = None,
        **block_kwargs: Any,
    ) -> str:
        """Retrieve memory context for prompt injection.

        Called by LlamaIndex's Memory when the agent needs context.
        Returns a formatted string with:
        - Memory nodes ordered by tier (proven first, then developing, current)
        - Semantic query results (tensions, blocked) if enabled

        Args:
            messages: Current conversation messages (ChatMessage objects).
            **block_kwargs: Additional keyword arguments.

        Returns:
            Formatted memory context string.
        """
        if self._memory.size == 0:
            return ""

        # Build context with tier info, respecting token budget
        lines: list[str] = []
        char_budget = self._max_tokens * 4  # rough chars-to-tokens
        used = 0

        # Order by tier priority: proven > developing > current
        tier_order = {"proven": 0, "foundation": 0, "developing": 1, "current": 2}
        nodes_with_tier = []
        for ref in self._memory.nodes:
            meta = self._memory.temporal_map.get(ref.id)
            tier = meta.tier if meta else "current"
            freq = meta.frequency if meta else 1
            nodes_with_tier.append((ref, tier, freq))
        nodes_with_tier.sort(key=lambda x: (tier_order.get(x[1], 3), -x[2]))

        for ref, tier, freq in nodes_with_tier:
            line = f"[{tier}, freq={freq}] {ref.content}"
            if used + len(line) > char_budget:
                break
            lines.append(line)
            used += len(line)

        # Touch retrieved nodes — retrieval is engagement
        if nodes_with_tier:
            retrieved_ids = [ref.id for ref, _, _ in nodes_with_tier[:len(lines)]]
            if retrieved_ids:
                self._memory.touch_nodes_session_scoped(retrieved_ids)

        # Append semantic query insights if enabled
        if self._include_queries and lines:
            try:
                tensions = self._memory.query.tensions()
                if tensions.metadata.get("total_tensions", 0) > 0:
                    tension_line = f"\n[TENSIONS] {tensions}"
                    if used + len(tension_line) < char_budget:
                        lines.append(tension_line)
                        used += len(tension_line)
            except Exception:
                pass

            try:
                blocked = self._memory.query.blocked()
                if blocked.blockers:
                    blocked_line = f"\n[BLOCKED] {blocked}"
                    if used + len(blocked_line) < char_budget:
                        lines.append(blocked_line)
                        used += len(blocked_line)
            except Exception:
                pass

        return "\n".join(lines)

    async def _aput(self, messages: List[Any]) -> None:
        """Store flushed messages as FlowScript nodes.

        Called by LlamaIndex's Memory when short-term messages overflow.
        Each message becomes a FlowScript node with metadata about role
        and position. Handles both ChatMessage objects and plain dicts.

        Args:
            messages: List of ChatMessage objects (or dicts for testing).
        """
        prev_ref = None
        for msg in messages:
            content = _extract_message_content(msg)
            if not content:
                continue

            role = _extract_role(msg)
            ref = self._memory.thought(content)
            node = ref.node
            node.ext = node.ext or {}
            node.ext.update({
                "llamaindex_role": role,
                "llamaindex_source": "flush",
            })

            # Chain sequential messages
            if prev_ref:
                prev_ref.then(ref)
            prev_ref = ref

        if self._file_path:
            self._memory.save()

    async def atruncate(self, content: str, tokens_to_truncate: int) -> str | None:
        """Intelligently truncate memory context when token-constrained.

        Instead of removing all content (the default), we reduce the token
        budget and regenerate. This preserves proven/high-frequency nodes
        while dropping current/low-frequency ones.

        Args:
            content: Current memory context string.
            tokens_to_truncate: How many tokens to free up.

        Returns:
            Reduced context string, or None to remove entirely.
        """
        reduced_budget = self._max_tokens - tokens_to_truncate
        if reduced_budget <= 0:
            return None

        # Regenerate with smaller budget
        old_budget = self._max_tokens
        self._max_tokens = reduced_budget
        try:
            result = await self._aget()
            return result if result else None
        finally:
            self._max_tokens = old_budget

    # -- Additional convenience methods --

    def store(self, content: str, **metadata: Any) -> NodeRef:
        """Store a thought directly. Returns NodeRef for chaining."""
        ref = self._memory.thought(content)
        if metadata:
            ref.node.ext = ref.node.ext or {}
            ref.node.ext["llamaindex_meta"] = metadata
        return ref

    def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search memory for relevant content using word-level matching."""
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
        """Persist memory to disk."""
        self._memory.save()

    def close(self):
        """End session: prune dormant nodes, save. Returns SessionWrapResult."""
        return self._memory.session_wrap()


def _extract_message_content(msg: Any) -> str | None:
    """Extract text content from a ChatMessage or dict."""
    # Dict format
    if isinstance(msg, dict):
        content = msg.get("content", "")
        if isinstance(content, str):
            return content if content else None
        # Content blocks format
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text:
                        texts.append(text)
            return " ".join(texts) if texts else None
        return None

    # ChatMessage object — .content is a property returning str
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content if content else None

    # Fallback: content blocks on ChatMessage
    blocks = getattr(msg, "blocks", None)
    if blocks:
        texts = []
        for block in blocks:
            text = getattr(block, "text", None)
            if text:
                texts.append(text)
        return " ".join(texts) if texts else None

    return None


def _extract_role(msg: Any) -> str:
    """Extract role from a ChatMessage or dict."""
    if isinstance(msg, dict):
        role = msg.get("role", "unknown")
        return str(role)
    role = getattr(msg, "role", "unknown")
    # LlamaIndex MessageRole enum
    if hasattr(role, "value"):
        return role.value
    return str(role)
