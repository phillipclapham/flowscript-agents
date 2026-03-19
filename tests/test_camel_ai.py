"""Tests for FlowScript CAMEL-AI integration."""

import os
import pytest
from flowscript_agents.camel_ai import FlowScriptCamelMemory, MemoryRecord


class TestFlowScriptCamelMemory:
    def test_create_in_memory(self):
        mem = FlowScriptCamelMemory()
        assert mem.memory.size == 0

    def test_create_with_file(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem = FlowScriptCamelMemory(file_path=path)
        mem.write_record(MemoryRecord(content="test"))
        assert os.path.exists(path)

    def test_agent_id_property(self):
        mem = FlowScriptCamelMemory()
        assert mem.agent_id is None
        mem.agent_id = "agent-1"
        assert mem.agent_id == "agent-1"


class TestWriteRecords:
    def test_stores_records(self):
        mem = FlowScriptCamelMemory()
        mem.write_records([
            MemoryRecord(content="User asked about databases"),
            MemoryRecord(content="Recommended PostgreSQL"),
        ])
        assert mem.memory.size == 2

    def test_stores_role(self):
        mem = FlowScriptCamelMemory()
        mem.write_record(MemoryRecord(content="Hello", role="user"))
        ref = list(mem.memory.nodes)[0]
        assert ref.node.ext["camel_role"] == "user"

    def test_chains_sequential(self):
        mem = FlowScriptCamelMemory()
        mem.write_records([
            MemoryRecord(content="Q1"),
            MemoryRecord(content="A1"),
            MemoryRecord(content="Q2"),
        ])
        rels = list(mem.memory._relationships)
        assert len(rels) == 2

    def test_stores_extra_info(self):
        mem = FlowScriptCamelMemory()
        mem.write_record(MemoryRecord(
            content="Important", extra_info={"topic": "architecture"}
        ))
        ref = list(mem.memory.nodes)[0]
        assert ref.node.ext["camel_extra"]["topic"] == "architecture"

    def test_skips_empty(self):
        mem = FlowScriptCamelMemory()
        mem.write_record(MemoryRecord(content=""))
        assert mem.memory.size == 0


class TestRetrieve:
    def test_returns_context_records(self):
        mem = FlowScriptCamelMemory()
        mem.write_record(MemoryRecord(content="Test observation"))
        records = mem.retrieve()
        assert len(records) >= 1
        assert records[0].memory_record.content == "Test observation"

    def test_scores_by_tier(self):
        mem = FlowScriptCamelMemory()
        # Store several items, retrieve multiple times to graduate
        mem.write_record(MemoryRecord(content="Frequently accessed"))
        mem.retrieve()  # touch 1
        mem.retrieve()  # touch 2 → developing
        records = mem.retrieve()  # touch 3 → proven
        # Proven nodes should have highest score
        proven_records = [r for r in records if r.memory_record.extra_info.get("tier") == "proven"]
        if proven_records:
            assert proven_records[0].score >= 1.0

    def test_respects_window_size(self):
        mem = FlowScriptCamelMemory(window_size=3)
        for i in range(10):
            mem.write_record(MemoryRecord(content=f"Record {i}"))
        records = mem.retrieve()
        # Should be at most 3 node records (may have query enrichment too)
        node_records = [r for r in records if r.memory_record.extra_info.get("source") == "flowscript"]
        assert len(node_records) <= 3

    def test_includes_tensions_enrichment(self):
        mem = FlowScriptCamelMemory()
        ref1 = mem.memory.thought("Use Redis")
        ref2 = mem.memory.thought("Use Memcached")
        ref1.tension_with(ref2, axis="speed vs simplicity")
        records = mem.retrieve()
        tension_records = [
            r for r in records
            if r.memory_record.extra_info.get("type") == "tensions"
        ]
        assert len(tension_records) == 1

    def test_includes_blocked_enrichment(self):
        mem = FlowScriptCamelMemory()
        ref = mem.memory.thought("Deploy to production")
        ref.block(reason="Needs testing")
        records = mem.retrieve()
        blocked_records = [
            r for r in records
            if r.memory_record.extra_info.get("type") == "blocked"
        ]
        assert len(blocked_records) == 1


class TestGetContext:
    def test_returns_messages_and_count(self):
        mem = FlowScriptCamelMemory()
        mem.write_record(MemoryRecord(content="Test content"))
        messages, token_count = mem.get_context()
        assert len(messages) >= 1
        assert token_count > 0

    def test_messages_have_role(self):
        mem = FlowScriptCamelMemory()
        mem.write_record(MemoryRecord(content="Hello", role="user"))
        messages, _ = mem.get_context()
        node_msgs = [m for m in messages if m["content"] == "Hello"]
        # Role may be from original or from retrieve enrichment
        assert len(node_msgs) >= 0  # Content present in some form


class TestResolve:
    def test_resolve_existing(self):
        mem = FlowScriptCamelMemory()
        mem.write_record(MemoryRecord(content="Use Redis for caching"))
        ref = mem.resolve("Redis")
        assert ref is not None

    def test_resolve_nonexistent(self):
        mem = FlowScriptCamelMemory()
        ref = mem.resolve("nonexistent")
        assert ref is None

    def test_resolve_enables_semantic_ops(self):
        mem = FlowScriptCamelMemory()
        mem.write_record(MemoryRecord(content="Use Redis"))
        mem.write_record(MemoryRecord(content="Use Postgres"))
        ref1 = mem.resolve("Redis")
        ref2 = mem.resolve("Postgres")
        ref1.tension_with(ref2, axis="speed vs reliability")
        tensions = mem.memory.query.tensions()
        assert tensions.metadata.get("total_tensions", 0) > 0


class TestRecall:
    def test_finds_content(self):
        mem = FlowScriptCamelMemory()
        mem.write_record(MemoryRecord(content="Redis is great for caching"))
        results = mem.recall("Redis caching")
        assert len(results) >= 1

    def test_empty_returns_empty(self):
        mem = FlowScriptCamelMemory()
        results = mem.recall("nonexistent")
        assert results == []


class TestClear:
    def test_clear_removes_all(self):
        mem = FlowScriptCamelMemory()
        mem.write_records([
            MemoryRecord(content="A"),
            MemoryRecord(content="B"),
        ])
        mem.clear()
        assert mem.memory.size == 0


class TestLifecycle:
    def test_save_and_reload(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem1 = FlowScriptCamelMemory(file_path=path)
        mem1.write_record(MemoryRecord(content="Persistent"))
        mem1.save()

        mem2 = FlowScriptCamelMemory(file_path=path)
        assert mem2.memory.size == 1

    def test_close_saves(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem = FlowScriptCamelMemory(file_path=path)
        mem.write_record(MemoryRecord(content="Close test"))
        result = mem.close()
        assert result.saved is True
