"""Tests for FlowScript Haystack integration."""

import os
import pytest
from flowscript_agents.haystack import FlowScriptMemoryStore
from haystack.dataclasses import ChatMessage as HaystackChatMessage


class TestFlowScriptMemoryStore:
    def test_create_in_memory(self):
        store = FlowScriptMemoryStore()
        assert store.memory.size == 0

    def test_create_with_file(self, tmp_path):
        path = str(tmp_path / "mem.json")
        store = FlowScriptMemoryStore(file_path=path)
        store.add_memories(messages=[{"role": "user", "content": "test"}])
        assert os.path.exists(path)


class TestAddMemories:
    def test_stores_messages(self):
        store = FlowScriptMemoryStore()
        store.add_memories(messages=[
            {"role": "user", "content": "What database should we use?"},
            {"role": "assistant", "content": "I recommend PostgreSQL."},
        ])
        assert store.memory.size == 2

    def test_stores_haystack_chat_messages(self):
        """Real Haystack Agent passes ChatMessage objects, not dicts."""
        store = FlowScriptMemoryStore()
        store.add_memories(messages=[
            HaystackChatMessage.from_user("What database?"),
            HaystackChatMessage.from_assistant("I recommend PostgreSQL."),
        ])
        assert store.memory.size == 2

    def test_stores_user_id(self):
        store = FlowScriptMemoryStore()
        store.add_memories(
            messages=[{"role": "user", "content": "Hello"}],
            user_id="user-1",
        )
        ref = list(store.memory.nodes)[0]
        assert ref.node.ext["haystack_user_id"] == "user-1"

    def test_stores_role(self):
        store = FlowScriptMemoryStore()
        store.add_memories(messages=[{"role": "user", "content": "Hello"}])
        ref = list(store.memory.nodes)[0]
        assert ref.node.ext["haystack_role"] == "user"

    def test_chains_sequential_messages(self):
        store = FlowScriptMemoryStore()
        store.add_memories(messages=[
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ])
        rels = list(store.memory._relationships)
        assert len(rels) == 2

    def test_skips_empty_content(self):
        store = FlowScriptMemoryStore()
        store.add_memories(messages=[
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Real"},
        ])
        assert store.memory.size == 1

    def test_extra_kwargs_stored(self):
        store = FlowScriptMemoryStore()
        store.add_memories(
            messages=[{"role": "user", "content": "Hello"}],
            agent_id="agent-1",
            run_id="run-42",
        )
        ref = list(store.memory.nodes)[0]
        assert ref.node.ext["haystack_agent_id"] == "agent-1"
        assert ref.node.ext["haystack_run_id"] == "run-42"


class TestSearchMemories:
    def test_finds_by_query(self):
        store = FlowScriptMemoryStore()
        store.add_memories(messages=[
            {"role": "assistant", "content": "Redis is great for caching"},
            {"role": "assistant", "content": "PostgreSQL handles complex queries"},
        ])
        results = store.search_memories(query="Redis caching")
        assert len(results) >= 1
        assert isinstance(results[0], HaystackChatMessage)
        assert "Redis" in results[0].text

    def test_returns_all_without_query(self):
        store = FlowScriptMemoryStore()
        store.add_memories(messages=[
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
        ])
        results = store.search_memories()
        assert len(results) == 2

    def test_filters_by_user_id(self):
        store = FlowScriptMemoryStore()
        store.add_memories(
            messages=[{"role": "user", "content": "User 1 data"}],
            user_id="u1",
        )
        store.add_memories(
            messages=[{"role": "user", "content": "User 2 data"}],
            user_id="u2",
        )
        results = store.search_memories(user_id="u1")
        assert len(results) == 1
        assert "User 1" in results[0].text

    def test_respects_top_k(self):
        store = FlowScriptMemoryStore()
        for i in range(10):
            store.add_memories(messages=[
                {"role": "assistant", "content": f"Memory item {i}"},
            ])
        results = store.search_memories(top_k=3)
        assert len(results) == 3

    def test_returns_meta(self):
        store = FlowScriptMemoryStore()
        store.add_memories(
            messages=[{"role": "assistant", "content": "Important insight"}],
            user_id="u1",
        )
        results = store.search_memories(query="important insight")
        assert results[0].meta["source"] == "flowscript"
        assert "tier" in results[0].meta
        assert "frequency" in results[0].meta

    def test_search_touches_nodes(self):
        store = FlowScriptMemoryStore()
        store.add_memories(messages=[
            {"role": "assistant", "content": "Something about databases"},
        ])
        # First search
        store.search_memories(query="databases")
        # Check frequency increased
        ref = list(store.memory.nodes)[0]
        meta = store.memory.temporal_map.get(ref.id)
        assert meta.frequency >= 2


class TestDeleteMemories:
    def test_delete_single(self):
        store = FlowScriptMemoryStore()
        store.add_memories(messages=[
            {"role": "user", "content": "Delete me"},
        ])
        mem_id = list(store._id_map.keys())[0]
        store.delete_memory(mem_id)
        assert store.memory.size == 0

    def test_delete_all(self):
        store = FlowScriptMemoryStore()
        store.add_memories(messages=[
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
        ])
        store.delete_all_memories()
        assert store.memory.size == 0
        assert len(store._id_map) == 0

    def test_delete_by_user(self):
        store = FlowScriptMemoryStore()
        store.add_memories(
            messages=[{"role": "user", "content": "User 1"}],
            user_id="u1",
        )
        store.add_memories(
            messages=[{"role": "user", "content": "User 2"}],
            user_id="u2",
        )
        store.delete_all_memories(user_id="u1")
        assert store.memory.size == 1
        results = store.search_memories(user_id="u2")
        assert len(results) == 1


class TestResolve:
    def test_resolve_existing(self):
        store = FlowScriptMemoryStore()
        store.add_memories(messages=[{"role": "user", "content": "Test"}])
        mem_id = list(store._id_map.keys())[0]
        ref = store.resolve(mem_id)
        assert ref is not None

    def test_resolve_nonexistent(self):
        store = FlowScriptMemoryStore()
        ref = store.resolve("nonexistent")
        assert ref is None

    def test_resolve_enables_semantic_ops(self):
        store = FlowScriptMemoryStore()
        store.add_memories(messages=[
            {"role": "assistant", "content": "Use Redis"},
            {"role": "assistant", "content": "Use Memcached"},
        ])
        ids = list(store._id_map.keys())
        ref1 = store.resolve(ids[0])
        ref2 = store.resolve(ids[1])
        ref1.tension_with(ref2, axis="speed vs simplicity")
        tensions = store.memory.query.tensions()
        assert tensions.metadata.get("total_tensions", 0) > 0


class TestSerialization:
    def test_to_dict(self):
        store = FlowScriptMemoryStore(file_path="/tmp/test.json")
        d = store.to_dict()
        assert d["type"] == "flowscript_agents.haystack.FlowScriptMemoryStore"
        assert d["init_parameters"]["file_path"] == "/tmp/test.json"

    def test_from_dict(self, tmp_path):
        path = str(tmp_path / "mem.json")
        store1 = FlowScriptMemoryStore(file_path=path)
        store1.add_memories(messages=[{"role": "user", "content": "Persist"}])
        store1.save()

        d = store1.to_dict()
        store2 = FlowScriptMemoryStore.from_dict(d)
        assert store2.memory.size == 1


class TestLifecycle:
    def test_save_and_reload(self, tmp_path):
        path = str(tmp_path / "mem.json")
        store1 = FlowScriptMemoryStore(file_path=path)
        store1.add_memories(messages=[{"role": "user", "content": "Persist"}])
        store1.save()

        store2 = FlowScriptMemoryStore(file_path=path)
        assert store2.memory.size == 1
        # Index rebuilt
        assert len(store2._id_map) == 1

    def test_close_saves(self, tmp_path):
        path = str(tmp_path / "mem.json")
        store = FlowScriptMemoryStore(file_path=path)
        store.add_memories(messages=[{"role": "user", "content": "Close test"}])
        result = store.close()
        assert result.saved is True
