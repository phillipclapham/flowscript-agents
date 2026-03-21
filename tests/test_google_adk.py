"""Tests for FlowScript Google ADK integration.

Tests the MemoryService implementation without requiring google-adk package.
Uses mock objects for ADK Session/Event types.
"""

from types import SimpleNamespace

import pytest

from flowscript_agents.google_adk import FlowScriptMemoryService


def _make_event(text: str, author: str = "user"):
    """Create a mock ADK Event."""
    part = SimpleNamespace(text=text)
    content = SimpleNamespace(parts=[part])
    return SimpleNamespace(content=content, author=author)


def _make_session(events, app_name="test_app", user_id="user1", session_id="sess1"):
    """Create a mock ADK Session."""
    return SimpleNamespace(
        app_name=app_name,
        user_id=user_id,
        id=session_id,
        events=events,
    )


class TestAddSessionToMemory:
    @pytest.mark.asyncio
    async def test_add_session(self):
        service = FlowScriptMemoryService()
        session = _make_session([
            _make_event("What database should I use?", "user"),
            _make_event("Redis is great for caching", "model"),
        ])
        await service.add_session_to_memory(session)
        assert service.memory.size >= 2

    @pytest.mark.asyncio
    async def test_empty_session(self):
        service = FlowScriptMemoryService()
        session = _make_session([])
        await service.add_session_to_memory(session)
        assert service.memory.size == 0

    @pytest.mark.asyncio
    async def test_creates_temporal_chain(self):
        service = FlowScriptMemoryService()
        session = _make_session([
            _make_event("first"),
            _make_event("second"),
            _make_event("third"),
        ])
        await service.add_session_to_memory(session)
        ir = service.memory.to_ir()
        # Should have temporal relationships between sequential events
        temporal_rels = [r for r in ir.relationships if r.type.value == "temporal"]
        assert len(temporal_rels) >= 2

    @pytest.mark.asyncio
    async def test_stores_metadata(self):
        service = FlowScriptMemoryService()
        session = _make_session(
            [_make_event("test")],
            app_name="myapp",
            user_id="u42",
        )
        await service.add_session_to_memory(session)
        node = service.memory.nodes[0].node
        assert node.ext["adk_app"] == "myapp"
        assert node.ext["adk_user"] == "u42"


class TestSearchMemory:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        service = FlowScriptMemoryService()
        session = _make_session([
            _make_event("Redis is fast for caching"),
            _make_event("Postgres is good for transactions"),
        ])
        await service.add_session_to_memory(session)

        result = await service.search_memory(
            app_name="test_app", user_id="user1", query="Redis"
        )
        assert len(result.memories) >= 1

    @pytest.mark.asyncio
    async def test_search_empty(self):
        service = FlowScriptMemoryService()
        result = await service.search_memory(
            app_name="test_app", user_id="user1", query="anything"
        )
        assert result.memories == []

    @pytest.mark.asyncio
    async def test_search_returns_memory_structure(self):
        service = FlowScriptMemoryService()
        session = _make_session([_make_event("test content")])
        await service.add_session_to_memory(session)

        result = await service.search_memory(
            app_name="test_app", user_id="user1", query="test"
        )
        assert hasattr(result, "memories")
        if result.memories:
            mem = result.memories[0]
            assert hasattr(mem, "content")


class TestAddEventsToMemory:
    @pytest.mark.asyncio
    async def test_incremental_add(self):
        service = FlowScriptMemoryService()
        await service.add_events_to_memory(
            app_name="test_app",
            user_id="user1",
            events=[
                _make_event("incremental event 1"),
                _make_event("incremental event 2"),
            ],
        )
        assert service.memory.size >= 2


class TestQueryIntegration:
    @pytest.mark.asyncio
    async def test_tensions_available(self):
        service = FlowScriptMemoryService()
        speed = service.memory.thought("speed matters")
        cost = service.memory.thought("cost is important")
        service.memory.tension(speed, cost, "performance vs budget")

        result = await service.search_memory(
            app_name="test_app",
            user_id="user1",
            query="tension tradeoff"
        )
        # Should include tension summary
        assert any("tension" in str(m).lower() for m in result.memories)


class TestResolve:
    """Test resolve() bridge from ADK node IDs to FlowScript NodeRef."""

    def test_resolve_existing_node(self):
        service = FlowScriptMemoryService()
        ref = service.memory.thought("Use Redis")
        resolved = service.resolve(ref.id)
        assert resolved is not None
        assert "Redis" in resolved.content

    def test_resolve_nonexistent_returns_none(self):
        service = FlowScriptMemoryService()
        assert service.resolve("nonexistent-id") is None

    def test_resolve_enables_semantic_relationships(self):
        service = FlowScriptMemoryService()
        db = service.memory.thought("Use Redis for sessions")
        cache = service.memory.thought("Use Redis for caching")

        db_ref = service.resolve(db.id)
        cache_ref = service.resolve(cache.id)
        assert db_ref is not None and cache_ref is not None

        cache_ref.causes(db_ref)
        db_ref.tension_with(cache_ref, axis="simplicity vs resilience")

        tensions = service.memory.query.tensions()
        assert tensions.metadata["total_tensions"] >= 1

    def test_resolve_enables_blocking(self):
        service = FlowScriptMemoryService()
        ref = service.memory.thought("Deploy Redis cluster")
        resolved = service.resolve(ref.id)
        assert resolved is not None
        resolved.block(reason="Waiting on Sentinel setup")

        blocked = service.memory.query.blocked()
        assert len(blocked.blockers) >= 1
