"""Tests for FlowScript Memory — Python implementation."""

import json
import tempfile
from pathlib import Path

import pytest

from flowscript_agents import Memory, NodeRef
from flowscript_agents.types import IR, Node, NodeType, Provenance


class TestNodeCreation:
    def test_create_thought(self):
        mem = Memory()
        t = mem.thought("caching improves latency")
        assert isinstance(t, NodeRef)
        assert t.content == "caching improves latency"
        assert t.type.value == "thought"
        assert mem.size == 1

    def test_create_all_types(self):
        mem = Memory()
        mem.thought("a thought")
        mem.statement("a statement")
        mem.question("a question?")
        mem.action("do something")
        mem.insight("an insight")
        mem.completion("done")
        assert mem.size == 6

    def test_deduplication(self):
        mem = Memory()
        t1 = mem.thought("same content")
        t2 = mem.thought("same content")
        assert t1.id == t2.id
        assert mem.size == 1

    def test_different_types_different_ids(self):
        mem = Memory()
        t = mem.thought("content")
        s = mem.statement("content")
        assert t.id != s.id
        assert mem.size == 2


class TestAlternatives:
    def test_create_alternative(self):
        mem = Memory()
        q = mem.question("Which database?")
        alt = mem.alternative(q, "Redis")
        assert alt.type.value == "alternative"
        assert mem.size == 2

    def test_alternative_creates_relationship(self):
        mem = Memory()
        q = mem.question("Which database?")
        mem.alternative(q, "Redis")
        ir = mem.to_ir()
        assert len(ir.relationships) == 1
        assert ir.relationships[0].type.value == "alternative"


class TestRelationships:
    def test_causes(self):
        mem = Memory()
        a = mem.thought("cause")
        b = mem.thought("effect")
        a.causes(b)
        ir = mem.to_ir()
        assert len(ir.relationships) == 1
        assert ir.relationships[0].type.value == "causes"
        assert ir.relationships[0].source == a.id
        assert ir.relationships[0].target == b.id

    def test_tension(self):
        mem = Memory()
        a = mem.thought("speed")
        b = mem.thought("cost")
        mem.tension(a, b, "performance vs budget")
        ir = mem.to_ir()
        assert len(ir.relationships) == 1
        assert ir.relationships[0].type.value == "tension"
        assert ir.relationships[0].axis_label == "performance vs budget"

    def test_chaining(self):
        mem = Memory()
        a = mem.thought("A")
        b = mem.thought("B")
        c = mem.thought("C")
        result = a.causes(b).causes(c)
        # causes returns self (the source), so chaining works from a
        assert result.id == a.id

    def test_relationship_deduplication(self):
        mem = Memory()
        a = mem.thought("A")
        b = mem.thought("B")
        a.causes(b)
        a.causes(b)
        ir = mem.to_ir()
        assert len(ir.relationships) == 1

    def test_string_resolution(self):
        """String args get resolved to thought nodes."""
        mem = Memory()
        t = mem.thought("existing")
        t.causes("new thought")
        assert mem.size == 2


class TestStates:
    def test_decide(self):
        mem = Memory()
        t = mem.thought("use Redis")
        t.decide(rationale="speed critical")
        ir = mem.to_ir()
        assert len(ir.states) == 1
        assert ir.states[0].type.value == "decided"
        assert ir.states[0].fields.rationale == "speed critical"

    def test_block(self):
        mem = Memory()
        t = mem.thought("deploy")
        t.block(reason="waiting on API keys")
        ir = mem.to_ir()
        assert ir.states[0].type.value == "blocked"
        assert ir.states[0].fields.reason == "waiting on API keys"

    def test_park(self):
        mem = Memory()
        t = mem.thought("optimize later")
        t.park(why="not urgent", until="next sprint")
        ir = mem.to_ir()
        assert ir.states[0].type.value == "parking"

    def test_explore(self):
        mem = Memory()
        t = mem.thought("try this approach")
        t.explore()
        ir = mem.to_ir()
        assert ir.states[0].type.value == "exploring"

    def test_unblock(self):
        mem = Memory()
        t = mem.thought("deploy")
        t.block(reason="waiting")
        t.unblock()
        ir = mem.to_ir()
        assert len(ir.states) == 0

    def test_state_replacement(self):
        """Setting same state type replaces, doesn't duplicate."""
        mem = Memory()
        t = mem.thought("decision")
        t.decide(rationale="reason 1")
        t.decide(rationale="reason 2")
        ir = mem.to_ir()
        assert len(ir.states) == 1
        assert ir.states[0].fields.rationale == "reason 2"


class TestQueries:
    def test_tensions_query(self):
        mem = Memory()
        a = mem.thought("speed")
        b = mem.thought("cost")
        mem.tension(a, b, "performance vs budget")
        result = mem.query.tensions()
        assert result.metadata["total_tensions"] >= 1

    def test_blocked_query(self):
        mem = Memory()
        t = mem.thought("deploy")
        t.block(reason="waiting on keys", since="2026-03-17")
        result = mem.query.blocked()
        assert len(result.blockers) >= 1

    def test_why_query(self):
        mem = Memory()
        a = mem.thought("root cause")
        b = mem.thought("consequence")
        a.causes(b)
        result = mem.query.why(b.id)
        assert result.metadata["total_ancestors"] >= 1

    def test_alternatives_query(self):
        mem = Memory()
        q = mem.question("Which DB?")
        mem.alternative(q, "Redis").decide(rationale="fast")
        mem.alternative(q, "SQLite").block(reason="no concurrency")
        result = mem.query.alternatives(q.id)
        assert len(result.alternatives) == 2


class TestSerialization:
    def test_to_json(self):
        mem = Memory()
        mem.thought("test")
        data = mem.to_json()
        assert "flowscript_memory" in data
        assert "ir" in data
        assert "temporal" in data
        assert "config" in data
        assert len(data["ir"]["nodes"]) == 1

    def test_to_json_string(self):
        mem = Memory()
        mem.thought("test")
        s = mem.to_json_string()
        parsed = json.loads(s)
        assert len(parsed["ir"]["nodes"]) == 1

    def test_from_json_roundtrip(self):
        mem = Memory()
        mem.thought("idea")
        mem.question("why?")
        data = mem.to_json()
        mem2 = Memory.from_json(data)
        assert mem2.size == 2

    def test_from_json_string(self):
        mem = Memory()
        mem.thought("test")
        s = mem.to_json_string()
        mem2 = Memory.from_json(s)
        assert mem2.size == 1


class TestPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "memory.json")
            mem = Memory()
            mem.thought("persistent thought")
            mem.save(path)

            mem2 = Memory.load(path)
            assert mem2.size == 1
            assert mem2.nodes[0].content == "persistent thought"

    def test_load_or_create_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "new.json")
            mem = Memory.load_or_create(path)
            assert mem.size == 0
            assert mem.file_path is not None

    def test_load_or_create_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "existing.json")
            mem1 = Memory()
            mem1.thought("existing")
            mem1.save(path)

            mem2 = Memory.load_or_create(path)
            assert mem2.size == 1

    def test_save_no_arg(self):
        """save() with no arg uses stored path."""
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "memory.json")
            mem = Memory.load_or_create(path)
            mem.thought("new idea")
            mem.save()  # no arg — uses stored path

            mem2 = Memory.load(path)
            assert mem2.size == 1

    def test_save_creates_parents(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "nested" / "dir" / "memory.json")
            mem = Memory()
            mem.thought("deep")
            mem.save(path)
            assert Path(path).exists()


class TestLookup:
    def test_get_node(self):
        mem = Memory()
        t = mem.thought("findme")
        found = mem.get_node(t.id)
        assert found is not None
        assert found.content == "findme"

    def test_get_node_missing(self):
        mem = Memory()
        assert mem.get_node("nonexistent") is None

    def test_ref(self):
        mem = Memory()
        t = mem.thought("findme")
        ref = mem.ref(t.id)
        assert ref.content == "findme"

    def test_ref_missing_raises(self):
        mem = Memory()
        with pytest.raises(KeyError):
            mem.ref("nonexistent")

    def test_find_nodes(self):
        mem = Memory()
        mem.thought("Redis is fast")
        mem.thought("Postgres is reliable")
        mem.thought("SQLite is simple")
        results = mem.find_nodes("is")
        assert len(results) == 3

    def test_find_nodes_case_insensitive(self):
        mem = Memory()
        mem.thought("Redis Is Fast")
        results = mem.find_nodes("redis")
        assert len(results) == 1


class TestFromIR:
    def test_from_ir(self):
        ir = IR(
            nodes=[
                Node(
                    id="abc123",
                    type=NodeType.THOUGHT,
                    content="test",
                    provenance=Provenance(
                        source_file="test.fs",
                        line_number=1,
                        timestamp="2026-03-17T00:00:00Z",
                    ),
                )
            ],
            relationships=[],
            states=[],
        )
        mem = Memory.from_ir(ir)
        assert mem.size == 1


class TestIntegration:
    def test_complete_workflow(self):
        """Full agent reasoning workflow."""
        mem = Memory()

        # Build reasoning
        q = mem.question("Which database for agent memory?")
        redis = mem.alternative(q, "Redis")
        redis.decide(rationale="speed critical for real-time agents")

        sqlite = mem.alternative(q, "SQLite")
        sqlite.block(reason="no concurrent write support")

        speed = mem.thought("Redis gives sub-ms reads")
        cost = mem.thought("cluster costs $200/mo")
        mem.tension(speed, cost, "performance vs cost")

        # Query
        tensions = mem.query.tensions()
        assert tensions.metadata["total_tensions"] >= 1

        blocked = mem.query.blocked()
        assert len(blocked.blockers) >= 1

        alts = mem.query.alternatives(q.id)
        assert len(alts.alternatives) == 2

        # Persist
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "agent-memory.json")
            mem.save(path)
            mem2 = Memory.load(path)
            assert mem2.size == mem.size

            # Queries work after reload
            tensions2 = mem2.query.tensions()
            assert tensions2.metadata["total_tensions"] >= 1
