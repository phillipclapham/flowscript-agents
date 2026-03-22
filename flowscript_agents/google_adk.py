"""
FlowScript Google ADK Integration.

Implements Google ADK's BaseMemoryService interface, making FlowScript memory
available as a memory service for ADK agents.

Usage:
    from flowscript_agents.google_adk import FlowScriptMemoryService

    memory_service = FlowScriptMemoryService("./agent-memory.json")
    # Use with ADK Runner:
    # runner = Runner(agent=agent, memory_service=memory_service, ...)

Note: Requires google-adk package: pip install flowscript-agents[google-adk]
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence, TYPE_CHECKING

from google.adk.memory import BaseMemoryService as _ADKBaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry

from .memory import Memory, MemoryOptions, NodeRef

if TYPE_CHECKING:
    from .embeddings.providers import EmbeddingProvider
    from .embeddings.extract import ExtractFn
    from .embeddings.consolidate import ConsolidationProvider
    from .unified import UnifiedMemory

logger = logging.getLogger(__name__)


class FlowScriptMemoryService(_ADKBaseMemoryService):
    """Google ADK BaseMemoryService backed by FlowScript reasoning memory.

    Implements the two required ADK methods:
    - add_session_to_memory(session) — extracts reasoning from session events
    - search_memory(app_name, user_id, query) — searches FlowScript graph

    For auto-extraction and vector search, provide an embedder + LLM::

        from flowscript_agents.embeddings import OpenAIEmbeddings

        service = FlowScriptMemoryService(
            "./agent-memory.json",
            embedder=OpenAIEmbeddings(),
            llm=my_llm_fn,
        )

    Access FlowScript queries via .memory property:
        service.memory.query.tensions()
        service.memory.query.blocked()
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
        # Start temporal session
        self._memory.session_start()
        self._memory.set_adapter_context("google_adk", "FlowScriptMemoryService", "init")

    @property
    def memory(self) -> Memory:
        return self._memory

    @property
    def unified(self) -> UnifiedMemory | None:
        """Access UnifiedMemory for vector search + extraction. None if not configured."""
        return self._unified

    def resolve(self, node_id: str) -> NodeRef | None:
        """Resolve a node ID to a FlowScript NodeRef for semantic operations.

        Node IDs appear in search_memory results. Use the returned NodeRef to
        build relationships that power FlowScript's semantic queries::

            results = await service.search_memory(
                app_name="myapp", user_id="user1", query="database"
            )
            for mem in results["memories"]:
                ref = service.resolve(mem["id"])
                if ref:
                    ref.block(reason="Waiting on approval")

            service.memory.query.blocked()
        """
        try:
            return self._memory.ref(node_id)
        except KeyError:
            return None

    async def add_session_to_memory(self, session: Any) -> None:
        """Extract reasoning from an ADK session and add to memory.

        When UnifiedMemory is configured with an LLM, uses auto-extraction
        to create typed reasoning nodes from session content. Otherwise,
        stores raw content as thought nodes.
        """
        self._memory.set_adapter_operation("add_session")
        app_name = getattr(session, "app_name", "unknown")
        user_id = getattr(session, "user_id", "unknown")
        session_id = getattr(session, "id", "unknown")

        events = getattr(session, "events", []) or []
        adk_meta = {
            "adk_app": app_name,
            "adk_user": user_id,
            "adk_session": session_id,
        }

        # When unified has extractor, batch all event content and extract
        if self._unified and self._unified.extractor:
            texts = []
            authors = []
            for event in events:
                content = _extract_event_content(event)
                if content:
                    author = getattr(event, "author", "unknown")
                    texts.append(f"[{author}] {content}")
                    authors.append(author)

            if texts:
                combined = "\n".join(texts)
                # Use first non-"unknown" author for actor hint
                actor = next((a for a in authors if a != "unknown"), "agent")
                result = self._unified.add(combined, metadata=adk_meta, actor=actor)
                # Tag all extracted nodes with ADK metadata + authors
                for node_id in result.node_ids:
                    node = self._memory.get_node(node_id)
                    if node:
                        node.ext = node.ext or {}
                        node.ext.update(adk_meta)
                        node.ext["adk_authors"] = list(set(authors))
        else:
            # Fallback: store each event as a thought node
            prev_ref = None
            for event in events:
                content = _extract_event_content(event)
                if not content:
                    continue

                author = getattr(event, "author", "unknown")
                ref = self._memory.thought(content)

                node = ref.node
                node.ext = node.ext or {}
                node.ext.update(adk_meta)
                node.ext["adk_author"] = author

                # Index for vector search if embedder available
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

    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """Search FlowScript memory for relevant content.

        When UnifiedMemory is configured, uses vector + keyword + temporal
        ranking. Otherwise falls back to word-level matching.

        Returns ADK SearchMemoryResponse with MemoryEntry objects.
        """
        self._memory.set_adapter_operation("search_memory")
        # Use unified search when available (vector + keyword + temporal)
        if self._unified:
            unified_results = self._unified.search(query, top_k=10)
            scored_matches = []
            for r in unified_results:
                try:
                    ref = self._memory.ref(r.node_id)
                    scored_matches.append((ref, r.combined_score))
                except KeyError:
                    continue
        else:
            # Fallback: word-level search
            query_words = [w.lower() for w in query.split() if len(w) > 2]
            scored_matches = []
            if query_words:
                for node in self._memory._nodes.values():
                    content_lower = node.content.lower()
                    hits = sum(1 for w in query_words if w in content_lower)
                    if hits > 0:
                        score = hits / len(query_words)
                        scored_matches.append((NodeRef(self._memory, node), score))
                scored_matches.sort(key=lambda x: -x[1])

        matches = [ref for ref, _ in scored_matches[:10]]
        if matches:
            self._memory.touch_nodes_session_scoped([ref.id for ref in matches])

        # Also check if query relates to FlowScript query operations
        memories = []
        for ref, score in scored_matches[:10]:  # limit results
            node = ref.node
            ext = node.ext or {}

            # Filter by app_name/user_id if stored (skip filter for nodes without ADK metadata)
            if ext.get("adk_app") and ext["adk_app"] != app_name:
                continue
            if ext.get("adk_user") and ext["adk_user"] != user_id:
                continue

            memories.append(MemoryEntry(
                content=_make_content(node.content),
                id=node.id,
                author=ext.get("adk_author", "memory"),
                timestamp=node.provenance.timestamp,
                custom_metadata={
                    "node_type": node.type.value,
                    "source": "flowscript",
                },
            ))

        # Add FlowScript query insights if relevant
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["why", "reason", "cause"]):
            # Surface causal chain insights
            for ref in matches[:3]:
                try:
                    result = self._memory.query.why(ref.id)
                    chain = getattr(result, "causal_chain", [])
                    if chain:
                        chain_text = " → ".join(
                            getattr(n, "content", str(n)) for n in chain
                        )
                        memories.append(MemoryEntry(
                            content=_make_content(f"Causal chain: {chain_text}"),
                            id=f"why-{ref.id[:8]}",
                            author="flowscript-query",
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        ))
                except Exception:
                    logger.debug("FlowScript query failed during search enrichment", exc_info=True)

        if any(kw in query_lower for kw in ["tension", "tradeoff", "trade-off"]):
            try:
                tensions = self._memory.query.tensions()
                if tensions.metadata.get("total_tensions", 0) > 0:
                    memories.append(MemoryEntry(
                        content=_make_content(
                            f"Active tensions: {json.dumps(asdict(tensions), default=str)}"
                        ),
                        id="tensions-summary",
                        author="flowscript-query",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ))
            except Exception:
                pass

        if any(kw in query_lower for kw in ["blocked", "stuck", "waiting"]):
            try:
                blocked = self._memory.query.blocked()
                if blocked.blockers:
                    memories.append(MemoryEntry(
                        content=_make_content(
                            f"Blocked items: {json.dumps(asdict(blocked), default=str)}"
                        ),
                        id="blocked-summary",
                        author="flowscript-query",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ))
            except Exception:
                pass

        return SearchMemoryResponse(memories=memories)

    # -- Optional methods --

    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Any],
        session_id: str | None = None,
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Incremental event addition."""
        self._memory.set_adapter_operation("add_events")
        prev_ref = None
        for event in events:
            content = _extract_event_content(event)
            if not content:
                continue

            author = getattr(event, "author", "unknown")
            ref = self._memory.thought(content)
            node = ref.node
            node.ext = node.ext or {}
            node.ext.update({
                "adk_app": app_name,
                "adk_user": user_id,
                "adk_session": session_id or "unknown",
                "adk_author": author,
            })
            if custom_metadata:
                node.ext["custom_metadata"] = dict(custom_metadata)

            if prev_ref:
                prev_ref.then(ref)
            prev_ref = ref

        if self._file_path:
            self._memory.save()

    def save(self) -> None:
        """Persist to disk. No-op if no file_path was provided (in-memory mode)."""
        if self._unified:
            self._unified.save()
        elif self._memory.file_path:
            self._memory.save()

    def close(self):
        """End the session: prune dormant nodes, save. Returns SessionWrapResult."""
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


def _extract_event_content(event: Any) -> str | None:
    """Extract text content from an ADK event."""
    content = getattr(event, "content", None)
    if content is None:
        return None

    # ADK Content has .parts list
    parts = getattr(content, "parts", None)
    if parts:
        texts = []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
        return " ".join(texts) if texts else None

    # Fallback: string content
    if isinstance(content, str):
        return content

    return None


def _make_content(text: str) -> dict[str, Any]:
    """Create ADK-compatible Content structure.

    Returns a dict that can be used to construct google.genai.types.Content.
    When google-adk is installed, callers can wrap this with types.Content().
    """
    return {
        "parts": [{"text": text}],
        "role": "model",
    }
