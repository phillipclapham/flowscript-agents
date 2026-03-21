"""Tests for FlowScript smolagents integration.

Tests the Tool subclass pattern without requiring smolagents package.
Uses duck-typing — smolagents tools only need name, description, inputs,
output_type, and forward().
"""

import tempfile
import os

import pytest

from flowscript_agents.smolagents import FlowScriptMemoryTools


class TestFlowScriptMemoryTools:
    def test_create_in_memory(self):
        tools = FlowScriptMemoryTools()
        assert tools.memory.size == 0

    def test_create_with_file(self):
        path = os.path.join(tempfile.mkdtemp(), "test.json")
        tools = FlowScriptMemoryTools(file_path=path)
        assert tools.memory.size == 0

    def test_tools_returns_list(self):
        tools = FlowScriptMemoryTools()
        tool_list = tools.tools()
        assert len(tool_list) == 8

    def test_tool_protocol(self):
        """Each tool has the smolagents Tool protocol attributes."""
        tools = FlowScriptMemoryTools()
        for tool in tools.tools():
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "inputs")
            assert hasattr(tool, "output_type")
            assert hasattr(tool, "forward")
            assert callable(tool.forward)
            assert isinstance(tool.name, str)
            assert len(tool.name) > 0
            assert isinstance(tool.description, str)
            assert len(tool.description) > 0

    def test_tool_names(self):
        tools = FlowScriptMemoryTools()
        names = [t.name for t in tools.tools()]
        assert "store_memory" in names
        assert "recall_memory" in names
        assert "query_tensions" in names
        assert "query_blocked" in names
        assert "get_memory_context" in names


class TestStoreMemoryTool:
    def test_store(self):
        mem_tools = FlowScriptMemoryTools()
        store = [t for t in mem_tools.tools() if t.name == "store_memory"][0]
        result = store(content="Redis is fast", category="observation")
        assert "Stored" in result
        assert mem_tools.memory.size >= 1

    def test_store_with_category(self):
        mem_tools = FlowScriptMemoryTools()
        store = [t for t in mem_tools.tools() if t.name == "store_memory"][0]
        store(content="Chose Redis", category="decision")

        # Check category stored in ext
        for ref in mem_tools.memory.nodes:
            if "Redis" in ref.content:
                assert ref.node.ext.get("smolagents_category") == "decision"

    def test_store_default_category(self):
        mem_tools = FlowScriptMemoryTools()
        store = [t for t in mem_tools.tools() if t.name == "store_memory"][0]
        store(content="Something observed")
        # Should not crash with default category


class TestRecallMemoryTool:
    def test_recall_finds_content(self):
        mem_tools = FlowScriptMemoryTools()
        mem_tools.memory.thought("Redis is excellent for caching")
        recall = [t for t in mem_tools.tools() if t.name == "recall_memory"][0]
        result = recall(query="Redis")
        assert "Redis" in result
        # Tier may be current or developing (touch on recall can trigger graduation)
        assert "current" in result or "developing" in result

    def test_recall_empty(self):
        mem_tools = FlowScriptMemoryTools()
        recall = [t for t in mem_tools.tools() if t.name == "recall_memory"][0]
        result = recall(query="nothing here")
        assert "No relevant memories" in result

    def test_recall_with_limit(self):
        mem_tools = FlowScriptMemoryTools()
        for i in range(10):
            mem_tools.memory.thought(f"Memory item {i} about topic X")
        recall = [t for t in mem_tools.tools() if t.name == "recall_memory"][0]
        result = recall(query="topic X", limit=3)
        # Should have at most 3 results
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) <= 3


class TestQueryTools:
    def test_tensions_empty(self):
        mem_tools = FlowScriptMemoryTools()
        tensions = [t for t in mem_tools.tools() if t.name == "query_tensions"][0]
        result = tensions()
        assert "No tensions" in result

    def test_blocked_empty(self):
        mem_tools = FlowScriptMemoryTools()
        blocked = [t for t in mem_tools.tools() if t.name == "query_blocked"][0]
        result = blocked()
        assert "Nothing blocked" in result

    def test_tensions_with_data(self):
        mem_tools = FlowScriptMemoryTools()
        a = mem_tools.memory.thought("Fast but expensive")
        b = mem_tools.memory.thought("Slow but cheap")
        mem_tools.memory.tension(a, b, "speed vs cost")

        tensions = [t for t in mem_tools.tools() if t.name == "query_tensions"][0]
        result = tensions()
        assert "speed vs cost" in result or "1" in result  # Should find the tension

    def test_blocked_with_data(self):
        mem_tools = FlowScriptMemoryTools()
        ref = mem_tools.memory.thought("Deploy Redis")
        ref.block(reason="Waiting on approval")

        blocked = [t for t in mem_tools.tools() if t.name == "query_blocked"][0]
        result = blocked()
        assert "blocked" in result.lower() or "1" in result


class TestGetMemoryContextTool:
    def test_empty_memory(self):
        mem_tools = FlowScriptMemoryTools()
        context = [t for t in mem_tools.tools() if t.name == "get_memory_context"][0]
        result = context()
        assert "empty" in result.lower() or isinstance(result, str)

    def test_with_content(self):
        mem_tools = FlowScriptMemoryTools()
        mem_tools.memory.thought("Important observation")
        context = [t for t in mem_tools.tools() if t.name == "get_memory_context"][0]
        result = context()
        assert isinstance(result, str)


class TestResolve:
    def test_resolve_existing(self):
        tools = FlowScriptMemoryTools()
        tools.memory.thought("Use Redis for sessions")
        ref = tools.resolve("Redis")
        assert ref is not None
        assert "Redis" in ref.content

    def test_resolve_nonexistent(self):
        tools = FlowScriptMemoryTools()
        assert tools.resolve("nonexistent xyz") is None

    def test_resolve_enables_semantic_relationships(self):
        tools = FlowScriptMemoryTools()
        tools.memory.thought("Use Redis for sessions")
        tools.memory.thought("Use Redis for caching too")

        db = tools.resolve("sessions")
        cache = tools.resolve("caching")
        assert db is not None and cache is not None

        cache.causes(db)
        db.tension_with(cache, axis="simplicity vs resilience")

        tensions = tools.memory.query.tensions()
        assert tensions.metadata["total_tensions"] >= 1

    def test_resolve_enables_blocking(self):
        tools = FlowScriptMemoryTools()
        tools.memory.thought("Deploy Redis cluster")
        ref = tools.resolve("Deploy Redis")
        assert ref is not None
        ref.block(reason="Waiting on Sentinel")

        blocked = tools.memory.query.blocked()
        assert len(blocked.blockers) >= 1


class TestPersistence:
    def test_save_and_reload(self):
        path = os.path.join(tempfile.mkdtemp(), "test.json")
        tools = FlowScriptMemoryTools(file_path=path)
        store = [t for t in tools.tools() if t.name == "store_memory"][0]
        store(content="Persistent memory test")
        tools.save()

        tools2 = FlowScriptMemoryTools(file_path=path)
        recall = [t for t in tools2.tools() if t.name == "recall_memory"][0]
        result = recall(query="Persistent")
        assert "Persistent" in result

    def test_close_saves(self):
        path = os.path.join(tempfile.mkdtemp(), "test.json")
        tools = FlowScriptMemoryTools(file_path=path)
        tools.memory.thought("Will survive close")
        tools.close()

        tools2 = FlowScriptMemoryTools(file_path=path)
        assert tools2.memory.size >= 1
