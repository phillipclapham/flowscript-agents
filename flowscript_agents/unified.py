"""
UnifiedMemory — Complete agent memory in one class.

Wires together Memory (reasoning) + VectorIndex (similarity) +
AutoExtract (ingestion) + UnifiedSearch (merged ranking).

This is the "complete agent memory" API:
- add() = auto-extract + embed + store (Mem0-like, but with typed reasoning)
- search() = vector + keyword + temporal (unified ranking)
- .memory = full reasoning query access (why, tensions, blocked, etc.)

Usage:
    from flowscript_agents import UnifiedMemory
    from flowscript_agents.embeddings import OpenAIEmbeddings

    mem = UnifiedMemory("./agent.json", embedder=OpenAIEmbeddings(), llm=my_llm)

    # Episodic (what happened)
    mem.add("User chose PostgreSQL over MySQL for ACID compliance")

    # Search (vector + keyword + temporal)
    results = mem.search("database decisions")

    # Reasoning (what Mem0 can't do)
    mem.memory.query.why(results[0].node_id)
    mem.memory.query.tensions()
    mem.memory.query.blocked()
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .memory import Memory, MemoryOptions, NodeRef
from .embeddings.providers import EmbeddingProvider
from .embeddings.index import VectorIndex, VectorSearchResult
from .embeddings.search import UnifiedSearch, UnifiedSearchResult
from .embeddings.extract import AutoExtract, ExtractFn, IngestResult
from .embeddings.consolidate import ConsolidationEngine, ConsolidationProvider


class UnifiedMemory:
    """Complete agent memory: reasoning + vector + auto-extraction in one install.

    Composes Memory + VectorIndex + AutoExtract + UnifiedSearch.
    All components are accessible for direct use when needed.
    """

    def __init__(
        self,
        file_path: str | None = None,
        embedder: EmbeddingProvider | None = None,
        llm: ExtractFn | None = None,
        consolidation_provider: ConsolidationProvider | None = None,
        dedup_threshold: float = 0.80,
        candidate_threshold: float = 0.45,
        options: MemoryOptions | None = None,
        vector_weight: float = 0.4,
        keyword_weight: float = 0.4,
        temporal_weight: float = 0.2,
        auto_save: bool = False,
    ) -> None:
        """Create a UnifiedMemory instance.

        Args:
            file_path: Path for persistence. None = in-memory only.
            embedder: Embedding provider for vector search. None = keyword-only.
            llm: LLM function for auto-extraction. (prompt: str) -> str.
                 None = manual node creation only (add_raw still works).
            consolidation_provider: Tool-calling LLM for consolidation decisions.
                 Requires embedder + llm. When provided, add() uses type-aware
                 consolidation (ADD/UPDATE/RELATE/RESOLVE/NONE) instead of simple
                 semantic dedup. This is what makes FlowScript a memory system,
                 not just a vector store.
            dedup_threshold: Similarity threshold for dedup (0-1). Default 0.80.
                 Used when consolidation is NOT active.
            candidate_threshold: Similarity threshold for consolidation candidates (0-1).
                 Default 0.45. Lower than dedup — we want a wider net, LLM decides.
                 Empirically validated: real OpenAI embeddings show related content at
                 0.50-0.71, contradictions at 0.55-0.60, unrelated at 0.11-0.13.
            options: MemoryOptions for the underlying Memory instance.
            vector_weight: Weight for vector similarity in unified search (0-1).
            keyword_weight: Weight for keyword matching in unified search (0-1).
            temporal_weight: Weight for temporal scoring in unified search (0-1).
            auto_save: If True, automatically save memory + embeddings after each
                 add() call. Trades write performance for crash safety — no data lost
                 if process dies between add() calls. Default False (batch save at
                 session_end/close). Enable for long-running agents or crash-sensitive
                 deployments.
        """
        self._auto_save = auto_save
        # Core memory (reasoning layer)
        if file_path:
            self._memory = Memory.load_or_create(file_path, options=options)
        else:
            self._memory = Memory(options=options)

        # Vector index (optional — only if embedder provided)
        self._vector_index: VectorIndex | None = None
        if embedder is not None:
            self._vector_index = VectorIndex(self._memory, embedder)
            # Load existing embeddings from sidecar if available
            self._vector_index.load()
            # Index any nodes that don't have embeddings yet
            self._vector_index.index_all()

        # Consolidation engine (optional — requires embedder + llm + provider)
        self._consolidation_engine: ConsolidationEngine | None = None
        if consolidation_provider is not None and self._vector_index is not None:
            self._consolidation_engine = ConsolidationEngine(
                self._memory,
                provider=consolidation_provider,
                vector_index=self._vector_index,
                candidate_threshold=candidate_threshold,
            )

        # Auto-extractor (optional — only if llm provided)
        self._extractor: AutoExtract | None = None
        if llm is not None:
            self._extractor = AutoExtract(
                self._memory, llm=llm,
                vector_index=self._vector_index,
                dedup_threshold=dedup_threshold,
                consolidation_engine=self._consolidation_engine,
            )

        # Unified search (always available — degrades gracefully without vector)
        self._search = UnifiedSearch(
            self._memory,
            vector_index=self._vector_index,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            temporal_weight=temporal_weight,
        )

    # -- Properties --

    @property
    def memory(self) -> Memory:
        """Access the underlying Memory for reasoning queries, node creation, etc."""
        return self._memory

    @property
    def vector_index(self) -> VectorIndex | None:
        """Access the VectorIndex directly. None if no embedder configured."""
        return self._vector_index

    @property
    def extractor(self) -> AutoExtract | None:
        """Access the AutoExtract directly. None if no LLM configured."""
        return self._extractor

    @property
    def search_engine(self) -> UnifiedSearch:
        """Access the UnifiedSearch directly."""
        return self._search

    # -- High-level API --

    def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        actor: str | None = None,
    ) -> IngestResult:
        """Add text to memory via auto-extraction.

        If an LLM is configured: extracts typed reasoning structure, deduplicates,
        creates nodes/relationships/states.
        If no LLM: creates a single thought node with the text.

        Args:
            text: Text to ingest (conversation, notes, facts, etc.)
            metadata: Optional metadata stored in node.ext
            actor: Source of the text — "user" or "agent". Tunes extraction
                priorities when provided. See AutoExtract.ingest() for details.
        """
        if self._extractor is not None:
            result = self._extractor.ingest(text, metadata=metadata, actor=actor)
            if self._auto_save:
                self.save()
            return result

        # Fallback: create a simple thought node
        ref = self._memory.thought(text)
        if metadata:
            if ref.node.ext is None:
                ref.node.ext = {}
            ref.node.ext.update(metadata)
        # Index if we have an embedder
        if self._vector_index is not None:
            self._vector_index.index_node(ref.id)

        if self._auto_save:
            self.save()

        return IngestResult(
            nodes_created=1, nodes_deduplicated=0,
            relationships_created=0, states_created=0,
            node_ids=[ref.id],
        )

    _VALID_RAW_TYPES = frozenset({
        "thought", "statement", "question", "action", "insight", "completion",
    })

    def add_raw(self, content: str, node_type: str = "thought") -> NodeRef:
        """Add a single node directly (no LLM extraction).

        Bypasses auto-extraction for when you know exactly what to store.
        Still indexes for vector search if embedder is configured.
        Valid types: thought, statement, question, action, insight, completion.
        """
        if node_type not in self._VALID_RAW_TYPES:
            node_type = "thought"
        creator = getattr(self._memory, node_type)
        ref = creator(content)
        if self._vector_index is not None:
            self._vector_index.index_node(ref.id)
        if self._auto_save:
            self.save()
        return ref

    def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[UnifiedSearchResult]:
        """Search across vector similarity, keyword matching, and temporal signals.

        Returns merged, deduplicated results ranked by combined score.
        """
        return self._search.search(query, top_k=top_k, **kwargs)

    def vector_search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Pure vector similarity search (no keyword or temporal mixing).

        Returns empty list if no embedder configured.
        """
        if self._vector_index is None:
            return []
        return self._vector_index.search(query, top_k=top_k, threshold=threshold)

    # -- Lifecycle --

    def session_start(self) -> Any:
        """Start a session. Call at beginning of agent interaction."""
        return self._memory.session_start()

    def session_end(self) -> Any:
        """End a session with pruning + save."""
        return self._memory.session_end()

    def save(self) -> None:
        """Save memory + embeddings to disk. No-op for in-memory mode."""
        if self._memory.file_path:
            self._memory.save()
            if self._vector_index is not None:
                self._vector_index.save()

    def close(self) -> Any:
        """Full session wrap: prune + save memory + save embeddings."""
        result = self._memory.session_wrap()
        if self._memory.file_path and self._vector_index is not None:
            self._vector_index.save()
        return result

    def __enter__(self) -> "UnifiedMemory":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- Convenience delegations --

    @property
    def size(self) -> int:
        """Number of nodes in memory."""
        return self._memory.size

    def get_context(self, max_tokens: int = 4000) -> str:
        """Get memory content formatted for prompt injection.

        Returns all nodes with tier + frequency info, sorted by relevance.
        """
        lines: list[str] = []
        nodes = self._memory.nodes
        # Sort by tier weight then frequency
        tier_order = {"foundation": 0, "proven": 1, "developing": 2, "current": 3}

        def sort_key(ref: NodeRef) -> tuple[int, int]:
            temporal = self._memory.get_temporal(ref.id)
            tier_rank = tier_order.get(temporal.tier if temporal else "current", 3)
            freq = -(temporal.frequency if temporal else 0)
            return (tier_rank, freq)

        sorted_nodes = sorted(nodes, key=sort_key)

        total_chars = 0
        for ref in sorted_nodes:
            temporal = self._memory.get_temporal(ref.id)
            tier = temporal.tier if temporal else "current"
            freq = temporal.frequency if temporal else 1
            line = f"[{tier}|{freq}x] ({ref.type.value}) {ref.content}"
            line_chars = len(line) + 1  # +1 for newline
            if total_chars + line_chars > max_tokens * 4:  # ~4 chars per token
                break
            lines.append(line)
            total_chars += line_chars

        return "\n".join(lines)

    def __repr__(self) -> str:
        has_embedder = self._vector_index is not None
        has_llm = self._extractor is not None
        return (
            f"UnifiedMemory(nodes={self._memory.size}, "
            f"embedder={has_embedder}, llm={has_llm})"
        )
