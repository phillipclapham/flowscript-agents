"""Tests for FlowScript LlamaIndex integration."""

import json
import os
import tempfile
import pytest

from flowscript_agents.llamaindex import FlowScriptMemoryBlock
from llama_index.core.memory import BaseMemoryBlock, Memory
from llama_index.core.base.llms.types import ChatMessage


class TestFlowScriptMemoryBlock:
    """Test BaseMemoryBlock protocol implementation."""

    def test_create_in_memory(self):
        block = FlowScriptMemoryBlock()
        assert block.memory.size == 0
        assert block.name == "flowscript_reasoning"

    def test_create_with_file(self, tmp_path):
        path = str(tmp_path / "mem.json")
        block = FlowScriptMemoryBlock(file_path=path)
        block.store("test content")
        block.save()
        assert os.path.exists(path)

    def test_custom_name(self):
        block = FlowScriptMemoryBlock(name="custom_memory")
        assert block.name == "custom_memory"

    def test_protocol_fields(self):
        block = FlowScriptMemoryBlock()
        assert hasattr(block, "name")
        assert hasattr(block, "description")
        assert hasattr(block, "priority")
        assert hasattr(block, "accept_short_term_memory")
        assert block.accept_short_term_memory is True

    def test_is_instance_of_base_memory_block(self):
        """Critical: Pydantic validates isinstance — duck-typing doesn't work."""
        block = FlowScriptMemoryBlock()
        assert isinstance(block, BaseMemoryBlock)

    def test_accepted_by_memory_from_defaults(self):
        """Critical: Memory.from_defaults must accept our block without error."""
        block = FlowScriptMemoryBlock()
        memory = Memory.from_defaults(
            session_id="test",
            memory_blocks=[block],
        )
        assert memory is not None


class TestAget:
    """Test _aget — context retrieval."""

    @pytest.mark.asyncio
    async def test_empty_memory_returns_empty(self):
        block = FlowScriptMemoryBlock()
        result = await block._aget()
        assert result == ""

    @pytest.mark.asyncio
    async def test_returns_stored_content(self):
        block = FlowScriptMemoryBlock()
        block.store("Redis is fast for caching")
        block.store("PostgreSQL handles complex queries well")
        result = await block._aget()
        assert "Redis" in result
        assert "PostgreSQL" in result

    @pytest.mark.asyncio
    async def test_includes_tier_info(self):
        block = FlowScriptMemoryBlock()
        block.store("Important observation")
        result = await block._aget()
        assert "[current, freq=1]" in result

    @pytest.mark.asyncio
    async def test_includes_tensions_when_enabled(self):
        block = FlowScriptMemoryBlock(include_queries=True)
        ref1 = block.store("Use Redis")
        ref2 = block.store("Use Memcached")
        ref1.tension_with(ref2, axis="speed vs simplicity")
        result = await block._aget()
        assert "TENSIONS" in result

    @pytest.mark.asyncio
    async def test_excludes_queries_when_disabled(self):
        block = FlowScriptMemoryBlock(include_queries=False)
        ref1 = block.store("Use Redis")
        ref2 = block.store("Use Memcached")
        ref1.tension_with(ref2, axis="speed vs simplicity")
        result = await block._aget()
        assert "TENSIONS" not in result

    @pytest.mark.asyncio
    async def test_respects_token_budget(self):
        block = FlowScriptMemoryBlock(max_tokens=50)
        for i in range(20):
            block.store(f"Observation number {i} with some extra content to fill space")
        result = await block._aget()
        # Should be truncated — 50 tokens ≈ 200 chars
        assert len(result) < 500


class TestAput:
    """Test _aput — message storage from flush pipeline."""

    @pytest.mark.asyncio
    async def test_stores_messages_as_nodes(self):
        block = FlowScriptMemoryBlock()
        await block._aput([
            {"role": "user", "content": "What database should we use?"},
            {"role": "assistant", "content": "I recommend PostgreSQL for your use case."},
        ])
        assert block.memory.size == 2

    @pytest.mark.asyncio
    async def test_stores_role_metadata(self):
        block = FlowScriptMemoryBlock()
        await block._aput([
            {"role": "user", "content": "Hello"},
        ])
        ref = list(block.memory.nodes)[0]
        assert ref.node.ext["llamaindex_role"] == "user"
        assert ref.node.ext["llamaindex_source"] == "flush"

    @pytest.mark.asyncio
    async def test_chains_sequential_messages(self):
        block = FlowScriptMemoryBlock()
        await block._aput([
            {"role": "user", "content": "Question one"},
            {"role": "assistant", "content": "Answer one"},
            {"role": "user", "content": "Question two"},
        ])
        assert block.memory.size == 3
        # Relationships should exist (sequential chaining)
        rels = list(block.memory._relationships)
        assert len(rels) == 2  # q1→a1, a1→q2

    @pytest.mark.asyncio
    async def test_skips_empty_content(self):
        block = FlowScriptMemoryBlock()
        await block._aput([
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Real content"},
        ])
        assert block.memory.size == 1

    @pytest.mark.asyncio
    async def test_saves_to_file(self, tmp_path):
        path = str(tmp_path / "mem.json")
        block = FlowScriptMemoryBlock(file_path=path)
        await block._aput([
            {"role": "user", "content": "Test message"},
        ])
        assert os.path.exists(path)

    @pytest.mark.asyncio
    async def test_stores_chat_message_objects(self):
        """Real LlamaIndex integration uses ChatMessage, not dicts."""
        block = FlowScriptMemoryBlock()
        await block._aput([
            ChatMessage(role="user", content="What database?"),
            ChatMessage(role="assistant", content="I recommend PostgreSQL."),
        ])
        assert block.memory.size == 2
        refs = list(block.memory.nodes)
        assert refs[0].node.ext["llamaindex_role"] == "user"
        assert refs[1].node.ext["llamaindex_role"] == "assistant"

    @pytest.mark.asyncio
    async def test_chat_message_role_enum_extracted(self):
        """MessageRole enum .value must be extracted, not the enum itself."""
        block = FlowScriptMemoryBlock()
        await block._aput([
            ChatMessage(role="user", content="Test role extraction"),
        ])
        ref = list(block.memory.nodes)[0]
        role = ref.node.ext["llamaindex_role"]
        assert isinstance(role, str)
        assert role == "user"


class TestAtruncate:
    """Test atruncate — intelligent context reduction."""

    @pytest.mark.asyncio
    async def test_reduces_context(self):
        block = FlowScriptMemoryBlock(max_tokens=4000)
        for i in range(20):
            block.store(f"Observation {i}: " + "x" * 200)
        full = await block._aget()
        # Request truncation of 3000 tokens (= 12000 chars) — should cut content
        truncated = await block.atruncate(full, 3000)
        assert truncated is not None
        # Truncated version should have fewer lines
        assert truncated.count("\n") < full.count("\n")

    @pytest.mark.asyncio
    async def test_returns_none_when_fully_truncated(self):
        block = FlowScriptMemoryBlock(max_tokens=100)
        block.store("Small content")
        full = await block._aget()
        result = await block.atruncate(full, 200)
        assert result is None


class TestResolve:
    """Test resolve — bridging to semantic operations."""

    def test_resolve_existing(self):
        block = FlowScriptMemoryBlock()
        block.store("Use Redis for caching")
        ref = block.resolve("Redis")
        assert ref is not None
        assert "Redis" in ref.content

    def test_resolve_nonexistent(self):
        block = FlowScriptMemoryBlock()
        ref = block.resolve("nonexistent content")
        assert ref is None

    def test_resolve_enables_semantic_relationships(self):
        block = FlowScriptMemoryBlock()
        block.store("Use Redis for caching")
        block.store("Use Memcached for sessions")
        ref1 = block.resolve("Redis")
        ref2 = block.resolve("Memcached")
        assert ref1 is not None
        assert ref2 is not None
        ref1.tension_with(ref2, axis="flexibility vs simplicity")
        tensions = block.memory.query.tensions()
        assert tensions.metadata.get("total_tensions", 0) > 0

    def test_resolve_enables_blocking(self):
        block = FlowScriptMemoryBlock()
        block.store("Deploy to production")
        ref = block.resolve("production")
        assert ref is not None
        ref.block(reason="Need load testing first")
        blocked = block.memory.query.blocked()
        assert len(blocked.blockers) > 0


class TestRecall:
    """Test recall — word-level search."""

    def test_recall_finds_content(self):
        block = FlowScriptMemoryBlock()
        block.store("Redis is excellent for caching")
        block.store("PostgreSQL handles complex queries")
        results = block.recall("Redis caching performance")
        assert len(results) >= 1
        assert "Redis" in results[0]["content"]

    def test_recall_returns_tier_info(self):
        block = FlowScriptMemoryBlock()
        block.store("Test content")
        results = block.recall("test content")
        assert results[0]["tier"] in ("current", "developing")
        assert results[0]["frequency"] >= 1

    def test_recall_empty_returns_empty(self):
        block = FlowScriptMemoryBlock()
        results = block.recall("nonexistent")
        assert results == []


class TestLifecycle:
    """Test save/close lifecycle."""

    def test_save_and_reload(self, tmp_path):
        path = str(tmp_path / "mem.json")
        block1 = FlowScriptMemoryBlock(file_path=path)
        block1.store("Persistent content")
        block1.save()

        block2 = FlowScriptMemoryBlock(file_path=path)
        assert block2.memory.size == 1
        ref = list(block2.memory.nodes)[0]
        assert "Persistent" in ref.content

    def test_close_saves(self, tmp_path):
        path = str(tmp_path / "mem.json")
        block = FlowScriptMemoryBlock(file_path=path)
        block.store("Content to persist")
        result = block.close()
        assert result.saved is True

        # Verify file exists and has content
        block2 = FlowScriptMemoryBlock(file_path=path)
        assert block2.memory.size == 1
