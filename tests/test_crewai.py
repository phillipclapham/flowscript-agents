"""Tests for FlowScript CrewAI integration.

Tests the StorageBackend implementation without requiring crewai package.
Uses duck-typing (dict-based records) since CrewAI requires Python <3.14.
"""

from types import SimpleNamespace

import pytest

from flowscript_agents.crewai import FlowScriptStorage


def _make_record(**kwargs):
    """Create a mock MemoryRecord-like object."""
    defaults = {
        "id": "rec-1",
        "content": "test content",
        "scope": "/",
        "categories": [],
        "metadata": {},
        "importance": 0.5,
        "created_at": "2026-03-17T00:00:00Z",
        "last_accessed": "2026-03-17T00:00:00Z",
        "embedding": None,
        "source": None,
        "private": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestSave:
    def test_save_record(self):
        storage = FlowScriptStorage()
        storage.save([_make_record(id="r1", content="hello world")])
        assert storage.count() == 1

    def test_save_multiple(self):
        storage = FlowScriptStorage()
        storage.save([
            _make_record(id="r1", content="first"),
            _make_record(id="r2", content="second"),
        ])
        assert storage.count() == 2

    def test_save_creates_flowscript_node(self):
        storage = FlowScriptStorage()
        storage.save([_make_record(id="r1", content="node content")])
        assert storage.memory.size >= 1


class TestSearch:
    def test_search_all(self):
        storage = FlowScriptStorage()
        storage.save([
            _make_record(id="r1", content="alpha", embedding=[1.0, 0.0]),
            _make_record(id="r2", content="beta", embedding=[0.0, 1.0]),
        ])
        results = storage.search([0.5, 0.5])
        assert len(results) == 2

    def test_search_by_scope(self):
        storage = FlowScriptStorage()
        storage.save([
            _make_record(id="r1", content="a", scope="/project/a"),
            _make_record(id="r2", content="b", scope="/project/b"),
            _make_record(id="r3", content="c", scope="/other"),
        ])
        results = storage.search([], scope_prefix="/project")
        assert len(results) == 2

    def test_search_by_category(self):
        storage = FlowScriptStorage()
        storage.save([
            _make_record(id="r1", content="a", categories=["decision"]),
            _make_record(id="r2", content="b", categories=["insight"]),
        ])
        results = storage.search([], categories=["decision"])
        assert len(results) == 1

    def test_search_with_limit(self):
        storage = FlowScriptStorage()
        storage.save([_make_record(id=f"r{i}", content=f"item {i}") for i in range(10)])
        results = storage.search([], limit=3)
        assert len(results) == 3


class TestDelete:
    def test_delete_by_id(self):
        storage = FlowScriptStorage()
        storage.save([
            _make_record(id="r1", content="keep"),
            _make_record(id="r2", content="delete"),
        ])
        count = storage.delete(record_ids=["r2"])
        assert count == 1
        assert storage.count() == 1

    def test_delete_by_scope(self):
        storage = FlowScriptStorage()
        storage.save([
            _make_record(id="r1", content="a", scope="/old"),
            _make_record(id="r2", content="b", scope="/new"),
        ])
        count = storage.delete(scope_prefix="/old")
        assert count == 1


class TestUpdate:
    def test_update_content(self):
        storage = FlowScriptStorage()
        storage.save([_make_record(id="r1", content="original")])
        storage.update(_make_record(id="r1", content="updated"))
        rec = storage.get_record("r1")
        assert rec["content"] == "updated"


class TestGetRecord:
    def test_get_existing(self):
        storage = FlowScriptStorage()
        storage.save([_make_record(id="r1", content="findme")])
        rec = storage.get_record("r1")
        assert rec is not None
        assert rec["content"] == "findme"

    def test_get_missing(self):
        storage = FlowScriptStorage()
        assert storage.get_record("nope") is None


class TestListRecords:
    def test_list_all(self):
        storage = FlowScriptStorage()
        storage.save([
            _make_record(id="r1", content="a"),
            _make_record(id="r2", content="b"),
        ])
        records = storage.list_records()
        assert len(records) == 2

    def test_list_with_scope(self):
        storage = FlowScriptStorage()
        storage.save([
            _make_record(id="r1", scope="/a"),
            _make_record(id="r2", scope="/b"),
        ])
        records = storage.list_records(scope_prefix="/a")
        assert len(records) == 1


class TestScopes:
    def test_list_scopes(self):
        storage = FlowScriptStorage()
        storage.save([
            _make_record(id="r1", scope="/project/a"),
            _make_record(id="r2", scope="/project/b"),
            _make_record(id="r3", scope="/other"),
        ])
        scopes = storage.list_scopes("/project")
        assert "/project/a" in scopes
        assert "/project/b" in scopes

    def test_list_categories(self):
        storage = FlowScriptStorage()
        storage.save([
            _make_record(id="r1", categories=["decision", "insight"]),
            _make_record(id="r2", categories=["decision"]),
        ])
        cats = storage.list_categories()
        assert cats["decision"] == 2
        assert cats["insight"] == 1


class TestReset:
    def test_reset_all(self):
        storage = FlowScriptStorage()
        storage.save([_make_record(id="r1"), _make_record(id="r2")])
        storage.reset()
        assert storage.count() == 0

    def test_reset_scoped(self):
        storage = FlowScriptStorage()
        storage.save([
            _make_record(id="r1", scope="/old"),
            _make_record(id="r2", scope="/new"),
        ])
        storage.reset(scope_prefix="/old")
        assert storage.count() == 1
