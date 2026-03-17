"""Tests for FlowScript OpenAI Agents SDK integration.

Tests the Session protocol implementation without requiring openai-agents package.
"""

import tempfile
from pathlib import Path

import pytest

from flowscript_agents.openai_agents import FlowScriptSession


class TestGetItems:
    @pytest.mark.asyncio
    async def test_empty_session(self):
        session = FlowScriptSession("sess1")
        items = await session.get_items()
        assert items == []

    @pytest.mark.asyncio
    async def test_get_with_limit(self):
        session = FlowScriptSession("sess1")
        await session.add_items([
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ])
        items = await session.get_items(limit=2)
        assert len(items) == 2
        assert items[0]["content"] == "second"  # last 2


class TestAddItems:
    @pytest.mark.asyncio
    async def test_add_single(self):
        session = FlowScriptSession("sess1")
        await session.add_items([{"role": "user", "content": "hello"}])
        items = await session.get_items()
        assert len(items) == 1
        assert items[0]["role"] == "user"
        assert items[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_add_multiple(self):
        session = FlowScriptSession("sess1")
        await session.add_items([
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ])
        items = await session.get_items()
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_add_creates_nodes(self):
        session = FlowScriptSession("sess1")
        await session.add_items([{"role": "user", "content": "stored as node"}])
        assert session.memory.size >= 1

    @pytest.mark.asyncio
    async def test_preserves_order(self):
        session = FlowScriptSession("sess1")
        await session.add_items([{"role": "user", "content": "first"}])
        await session.add_items([{"role": "assistant", "content": "second"}])
        await session.add_items([{"role": "user", "content": "third"}])
        items = await session.get_items()
        assert [i["content"] for i in items] == ["first", "second", "third"]


class TestPopItem:
    @pytest.mark.asyncio
    async def test_pop_last(self):
        session = FlowScriptSession("sess1")
        await session.add_items([
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "last"},
        ])
        item = await session.pop_item()
        assert item["content"] == "last"
        remaining = await session.get_items()
        assert len(remaining) == 1

    @pytest.mark.asyncio
    async def test_pop_empty(self):
        session = FlowScriptSession("sess1")
        item = await session.pop_item()
        assert item is None


class TestClearSession:
    @pytest.mark.asyncio
    async def test_clear(self):
        session = FlowScriptSession("sess1")
        await session.add_items([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])
        await session.clear_session()
        items = await session.get_items()
        assert items == []


class TestSessionIsolation:
    @pytest.mark.asyncio
    async def test_separate_sessions(self):
        """Different session IDs should have independent items."""
        s1 = FlowScriptSession("sess1")
        s2 = FlowScriptSession("sess2")

        await s1.add_items([{"role": "user", "content": "session 1"}])
        await s2.add_items([{"role": "user", "content": "session 2"}])

        items1 = await s1.get_items()
        items2 = await s2.get_items()
        assert len(items1) == 1
        assert len(items2) == 1
        assert items1[0]["content"] == "session 1"
        assert items2[0]["content"] == "session 2"


class TestPersistence:
    @pytest.mark.asyncio
    async def test_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "session.json")

            s1 = FlowScriptSession("sess1", path)
            await s1.add_items([{"role": "user", "content": "persistent"}])

            s2 = FlowScriptSession("sess1", path)
            items = await s2.get_items()
            assert len(items) == 1
            assert items[0]["content"] == "persistent"


class TestMemoryAccess:
    @pytest.mark.asyncio
    async def test_query_access(self):
        """Can use FlowScript queries through session."""
        session = FlowScriptSession("sess1")
        speed = session.memory.thought("speed matters")
        cost = session.memory.thought("cost important")
        session.memory.tension(speed, cost, "performance vs cost")

        tensions = session.memory.query.tensions()
        assert tensions.metadata["total_tensions"] >= 1

    @pytest.mark.asyncio
    async def test_session_id_stored(self):
        session = FlowScriptSession("my-session")
        assert session.session_id == "my-session"
