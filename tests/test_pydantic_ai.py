"""Tests for FlowScript Pydantic AI integration.

Tests the dependency + tools pattern without requiring pydantic-ai package.
"""

import tempfile
import os

import pytest

from flowscript_agents.pydantic_ai import FlowScriptDeps, create_memory_tools


class TestFlowScriptDeps:
    def test_create_in_memory(self):
        deps = FlowScriptDeps()
        assert deps.memory.size == 0

    def test_create_with_file(self):
        path = os.path.join(tempfile.mkdtemp(), "test.json")
        deps = FlowScriptDeps(file_path=path)
        assert deps.memory.size == 0

    def test_store_and_recall(self):
        deps = FlowScriptDeps()
        deps.store("Redis is fast for session storage")
        deps.store("PostgreSQL is better for complex queries")

        results = deps.recall("Redis")
        assert len(results) >= 1
        assert any("Redis" in r["content"] for r in results)

    def test_recall_returns_tier_and_frequency(self):
        deps = FlowScriptDeps()
        deps.store("Important observation about caching")
        results = deps.recall("caching")
        assert len(results) == 1
        assert results[0]["tier"] in ("current", "developing")  # may graduate on touch
        assert results[0]["frequency"] >= 1
        assert "id" in results[0]

    def test_recall_empty_returns_empty(self):
        deps = FlowScriptDeps()
        results = deps.recall("nothing here")
        assert results == []

    def test_store_with_metadata(self):
        deps = FlowScriptDeps()
        ref = deps.store("decision about Redis", category="decision", priority="high")
        assert ref.node.ext["pydantic_ai_meta"]["category"] == "decision"
        assert ref.node.ext["pydantic_ai_meta"]["priority"] == "high"

    def test_get_context(self):
        deps = FlowScriptDeps()
        deps.store("First observation")
        deps.store("Second observation")
        context = deps.get_context(max_tokens=4000)
        # Should return some FlowScript formatted content
        assert isinstance(context, str)

    def test_save_and_reload(self):
        path = os.path.join(tempfile.mkdtemp(), "test.json")
        deps = FlowScriptDeps(file_path=path)
        deps.store("Persistent memory")
        deps.save()

        # Reload
        deps2 = FlowScriptDeps(file_path=path)
        results = deps2.recall("Persistent")
        assert len(results) >= 1

    def test_close_saves(self):
        path = os.path.join(tempfile.mkdtemp(), "test.json")
        deps = FlowScriptDeps(file_path=path)
        deps.store("Will be saved on close")
        deps.close()

        deps2 = FlowScriptDeps(file_path=path)
        results = deps2.recall("saved on close")
        assert len(results) >= 1


class TestResolve:
    def test_resolve_existing(self):
        deps = FlowScriptDeps()
        deps.store("Use Redis for sessions")
        ref = deps.resolve("Redis")
        assert ref is not None
        assert "Redis" in ref.content

    def test_resolve_nonexistent(self):
        deps = FlowScriptDeps()
        assert deps.resolve("nonexistent xyz") is None

    def test_resolve_enables_semantic_relationships(self):
        deps = FlowScriptDeps()
        deps.store("Use Redis for sessions")
        deps.store("Use Redis for caching too")

        db = deps.resolve("sessions")
        cache = deps.resolve("caching")
        assert db is not None and cache is not None

        cache.causes(db)
        db.tension_with(cache, axis="simplicity vs resilience")

        tensions = deps.memory.query.tensions()
        assert tensions.metadata["total_tensions"] >= 1

    def test_resolve_enables_blocking(self):
        deps = FlowScriptDeps()
        deps.store("Deploy Redis cluster")
        ref = deps.resolve("Deploy Redis")
        assert ref is not None
        ref.block(reason="Waiting on Sentinel")

        blocked = deps.memory.query.blocked()
        assert len(blocked.blockers) >= 1


class TestMemoryTools:
    def test_create_memory_tools_returns_functions(self):
        tools = create_memory_tools()
        assert len(tools) == 7
        assert all(callable(t) for t in tools)

    def test_tool_names(self):
        tools = create_memory_tools()
        names = [t.__name__ for t in tools]
        assert "store_memory" in names
        assert "recall_memory" in names
        assert "query_tensions" in names
        assert "query_blocked" in names
        assert "query_why" in names
        assert "query_what_if" in names
        assert "query_alternatives" in names

    @pytest.mark.asyncio
    async def test_store_tool(self):
        deps = FlowScriptDeps()
        tools = create_memory_tools()
        store_fn = tools[0]  # store_memory

        # Simulate RunContext with deps
        class MockCtx:
            def __init__(self, deps):
                self.deps = deps

        ctx = MockCtx(deps)
        result = await store_fn(ctx, "Test observation", "observation")
        assert "Stored" in result
        assert deps.memory.size >= 1

    @pytest.mark.asyncio
    async def test_recall_tool(self):
        deps = FlowScriptDeps()
        deps.store("Redis is fast")
        tools = create_memory_tools()
        recall_fn = tools[1]  # recall_memory

        class MockCtx:
            def __init__(self, deps):
                self.deps = deps

        ctx = MockCtx(deps)
        result = await recall_fn(ctx, "Redis")
        assert "Redis" in result

    @pytest.mark.asyncio
    async def test_recall_tool_empty(self):
        deps = FlowScriptDeps()
        tools = create_memory_tools()
        recall_fn = tools[1]

        class MockCtx:
            def __init__(self, deps):
                self.deps = deps

        ctx = MockCtx(deps)
        result = await recall_fn(ctx, "nothing")
        assert "No relevant memories" in result

    @pytest.mark.asyncio
    async def test_tensions_tool_empty(self):
        deps = FlowScriptDeps()
        tools = create_memory_tools()
        tensions_fn = tools[2]

        class MockCtx:
            def __init__(self, deps):
                self.deps = deps

        ctx = MockCtx(deps)
        result = await tensions_fn(ctx)
        assert "No tensions" in result

    @pytest.mark.asyncio
    async def test_blocked_tool_empty(self):
        deps = FlowScriptDeps()
        tools = create_memory_tools()
        blocked_fn = tools[3]

        class MockCtx:
            def __init__(self, deps):
                self.deps = deps

        ctx = MockCtx(deps)
        result = await blocked_fn(ctx)
        assert "Nothing blocked" in result
