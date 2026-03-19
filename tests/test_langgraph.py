"""Tests for FlowScript LangGraph integration."""

import tempfile
from pathlib import Path

import pytest

from flowscript_agents.langgraph import FlowScriptStore


class TestBasicOperations:
    def test_put_and_get(self):
        store = FlowScriptStore()
        store.put(("memories", "user1"), "pref1", {"content": "prefers dark mode"})
        item = store.get(("memories", "user1"), "pref1")
        assert item is not None
        assert item.value["content"] == "prefers dark mode"
        assert item.key == "pref1"
        assert item.namespace == ("memories", "user1")

    def test_get_missing(self):
        store = FlowScriptStore()
        item = store.get(("memories",), "nonexistent")
        assert item is None

    def test_put_update(self):
        store = FlowScriptStore()
        store.put(("ns",), "key1", {"content": "original"})
        store.put(("ns",), "key1", {"content": "updated"})
        item = store.get(("ns",), "key1")
        assert item.value["content"] == "updated"

    def test_delete(self):
        store = FlowScriptStore()
        store.put(("ns",), "key1", {"content": "deleteme"})
        store.delete(("ns",), "key1")
        item = store.get(("ns",), "key1")
        assert item is None

    def test_multiple_namespaces(self):
        store = FlowScriptStore()
        store.put(("a", "b"), "k1", {"content": "ab"})
        store.put(("a", "c"), "k1", {"content": "ac"})
        assert store.get(("a", "b"), "k1").value["content"] == "ab"
        assert store.get(("a", "c"), "k1").value["content"] == "ac"


class TestSearch:
    def test_search_by_namespace_prefix(self):
        store = FlowScriptStore()
        store.put(("docs", "user1"), "d1", {"content": "doc1"})
        store.put(("docs", "user1"), "d2", {"content": "doc2"})
        store.put(("docs", "user2"), "d3", {"content": "doc3"})
        store.put(("other",), "o1", {"content": "other"})

        results = store.search(("docs",))
        assert len(results) == 3

    def test_search_with_filter(self):
        store = FlowScriptStore()
        store.put(("ns",), "k1", {"content": "a", "type": "article"})
        store.put(("ns",), "k2", {"content": "b", "type": "note"})
        store.put(("ns",), "k3", {"content": "c", "type": "article"})

        results = store.search(("ns",), filter={"type": "article"})
        assert len(results) == 2

    def test_search_with_query(self):
        store = FlowScriptStore()
        store.put(("ns",), "k1", {"content": "Redis is fast"})
        store.put(("ns",), "k2", {"content": "Postgres is reliable"})
        store.put(("ns",), "k3", {"content": "Redis cluster setup"})

        results = store.search(("ns",), query="Redis")
        assert len(results) == 2

    def test_search_with_limit(self):
        store = FlowScriptStore()
        for i in range(10):
            store.put(("ns",), f"k{i}", {"content": f"item {i}"})

        results = store.search(("ns",), limit=3)
        assert len(results) == 3

    def test_search_with_offset(self):
        store = FlowScriptStore()
        for i in range(5):
            store.put(("ns",), f"k{i}", {"content": f"item {i}"})

        all_results = store.search(("ns",), limit=100)
        offset_results = store.search(("ns",), offset=2, limit=100)
        assert len(offset_results) == len(all_results) - 2


class TestListNamespaces:
    def test_list_all(self):
        store = FlowScriptStore()
        store.put(("a", "b"), "k1", {"content": "1"})
        store.put(("a", "c"), "k2", {"content": "2"})
        store.put(("d",), "k3", {"content": "3"})

        ns = store.list_namespaces()
        assert ("a", "b") in ns
        assert ("a", "c") in ns
        assert ("d",) in ns

    def test_list_with_prefix(self):
        store = FlowScriptStore()
        store.put(("a", "b"), "k1", {"content": "1"})
        store.put(("a", "c"), "k2", {"content": "2"})
        store.put(("d",), "k3", {"content": "3"})

        ns = store.list_namespaces(prefix=("a",))
        assert len(ns) == 2
        assert all(n[0] == "a" for n in ns)

    def test_list_with_max_depth(self):
        store = FlowScriptStore()
        store.put(("a", "b", "c"), "k1", {"content": "1"})
        store.put(("a", "b", "d"), "k2", {"content": "2"})

        ns = store.list_namespaces(max_depth=2)
        assert all(len(n) <= 2 for n in ns)


class TestPersistence:
    def test_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "store.json")

            store1 = FlowScriptStore(path)
            store1.put(("memories",), "key1", {"content": "persistent"})
            store1.save()

            store2 = FlowScriptStore(path)
            item = store2.get(("memories",), "key1")
            assert item is not None
            assert item.value["content"] == "persistent"


class TestFlowScriptIntegration:
    def test_memory_access(self):
        """Can access FlowScript queries through the store."""
        store = FlowScriptStore()
        store.put(("ns",), "k1", {"content": "speed matters"})
        store.put(("ns",), "k2", {"content": "cost is important"})

        # Build reasoning through the memory API
        speed = store.memory.thought("speed matters")
        cost = store.memory.thought("cost is important")
        store.memory.tension(speed, cost, "performance vs budget")

        # Query tensions
        tensions = store.memory.query.tensions()
        assert tensions.metadata["total_tensions"] >= 1

    def test_store_items_become_nodes(self):
        """Items stored through LangGraph API are visible as FlowScript nodes."""
        store = FlowScriptStore()
        store.put(("ns",), "k1", {"content": "stored via langgraph"})
        assert store.memory.size >= 1


class TestResolve:
    """Test resolve() bridge from LangGraph items to FlowScript NodeRef."""

    def test_resolve_existing_item(self):
        store = FlowScriptStore()
        store.put(("arch",), "db_choice", {"content": "Use Redis"})
        ref = store.resolve(("arch",), "db_choice")
        assert ref is not None
        assert "Redis" in ref.content

    def test_resolve_nonexistent_returns_none(self):
        store = FlowScriptStore()
        ref = store.resolve(("arch",), "nonexistent")
        assert ref is None

    def test_resolve_enables_semantic_relationships(self):
        """Core validation: resolve() bridges store items to semantic queries."""
        store = FlowScriptStore()
        store.put(("arch",), "db", {"content": "Use Redis for sessions"})
        store.put(("arch",), "cache", {"content": "Use Redis for caching too"})

        db = store.resolve(("arch",), "db")
        cache = store.resolve(("arch",), "cache")
        assert db is not None and cache is not None

        # Build semantic relationships via NodeRef API
        cache.causes(db)
        db.tension_with(cache, axis="simplicity vs resilience")
        db.decide(rationale="Redis for both", on="2026-03-18")

        # Semantic queries now find these relationships
        tensions = store.memory.query.tensions()
        assert tensions.metadata["total_tensions"] >= 1

        blocked = store.memory.query.blocked()
        # Nothing blocked — but the query works
        assert blocked is not None

    def test_resolve_enables_blocking(self):
        store = FlowScriptStore()
        store.put(("deploy",), "redis", {"content": "Deploy Redis cluster"})
        ref = store.resolve(("deploy",), "redis")
        assert ref is not None

        ref.block(reason="Waiting on Sentinel setup", since="2026-03-18")
        blocked = store.memory.query.blocked()
        assert len(blocked.blockers) >= 1

    def test_resolve_enables_alternatives(self):
        store = FlowScriptStore()
        # Create a question via memory API, then alternatives via store
        q = store.memory.question("Which caching strategy?")
        alt1 = store.memory.alternative(q, "Redis")
        alt2 = store.memory.alternative(q, "Varnish")
        alt1.decide(rationale="Already in stack")

        alts = store.memory.query.alternatives(q.id)
        assert alts is not None

    def test_resolve_after_save_load(self):
        """Resolve works after save/load cycle."""
        import tempfile, os
        path = os.path.join(tempfile.mkdtemp(), "test.json")
        store = FlowScriptStore(path)
        store.put(("arch",), "db", {"content": "Use Redis"})
        ref = store.resolve(("arch",), "db")
        assert ref is not None
        ref.decide(rationale="Speed critical")
        store.save()

        # Reload
        store2 = FlowScriptStore(path)
        ref2 = store2.resolve(("arch",), "db")
        assert ref2 is not None
        assert "Redis" in ref2.content


class TestAsync:
    @pytest.mark.asyncio
    async def test_async_put_and_get(self):
        store = FlowScriptStore()
        await store.aput(("ns",), "k1", {"content": "async item"})
        item = await store.aget(("ns",), "k1")
        assert item is not None
        assert item.value["content"] == "async item"

    @pytest.mark.asyncio
    async def test_async_search(self):
        store = FlowScriptStore()
        await store.aput(("ns",), "k1", {"content": "findme"})
        results = await store.asearch(("ns",), query="findme")
        assert len(results) == 1
