"""Tests for FlowScript Memory — Python implementation."""

import json
import tempfile
from pathlib import Path

import pytest

from flowscript_agents import Memory, NodeRef
from flowscript_agents.memory import UpdateResult
from flowscript_agents.types import IR, Node, NodeType, Provenance, RelationType


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


class TestUpdateNode:
    """Tests for Memory.update_node() — content modification with reference preservation."""

    def test_basic_update(self):
        """Update content, verify new content and node count unchanged."""
        mem = Memory()
        t = mem.thought("initial content")
        old_id = t.id
        result = mem.update_node(old_id, "updated content")
        assert isinstance(result, UpdateResult)
        assert result.ref.content == "updated content"
        assert result.ref.id != old_id  # content hash changed
        assert result.old_id == old_id
        assert result.merged is False
        assert mem.size == 1  # node count unchanged

    def test_update_preserves_type(self):
        """Updated node keeps its original type."""
        mem = Memory()
        s = mem.statement("a fact")
        result = mem.update_node(s.id, "a better fact")
        assert result.ref.type == NodeType.STATEMENT

    def test_update_preserves_relationships(self):
        """Relationships are re-pointed to the new node ID."""
        mem = Memory()
        a = mem.thought("cause")
        b = mem.thought("effect")
        mem.relate(a, b, RelationType.CAUSES)

        old_b_id = b.id
        updated_b = mem.update_node(old_b_id, "updated effect").ref

        # Relationship should now point to the updated node
        rels = [r for r in mem._relationships if r.target == updated_b.id]
        assert len(rels) == 1
        assert rels[0].source == a.id
        assert rels[0].type.value == "causes"

        # No relationships should reference the old ID
        old_refs = [r for r in mem._relationships if r.source == old_b_id or r.target == old_b_id]
        assert len(old_refs) == 0

    def test_update_preserves_states(self):
        """States are re-pointed to the new node ID."""
        mem = Memory()
        t = mem.thought("blocked thing")
        t.block("waiting on approval")
        old_id = t.id

        updated = mem.update_node(old_id, "updated blocked thing").ref

        # State should be on the new node
        states = [s for s in mem._states if s.node_id == updated.id]
        assert len(states) == 1
        assert states[0].type.value == "blocked"

        # No states on old ID
        old_states = [s for s in mem._states if s.node_id == old_id]
        assert len(old_states) == 0

    def test_update_preserves_temporal_created_at(self):
        """Temporal metadata preserves created_at, updates last_touched."""
        mem = Memory()
        mem.session_start()
        t = mem.thought("original")
        old_temporal = mem.get_temporal(t.id)
        old_created = old_temporal.created_at

        updated = mem.update_node(t.id, "modified").ref
        new_temporal = mem.get_temporal(updated.id)

        assert new_temporal.created_at == old_created  # preserved
        assert new_temporal.frequency == old_temporal.frequency  # preserved

    def test_update_same_content_noop(self):
        """Updating with identical content returns same node (no-op)."""
        mem = Memory()
        t = mem.thought("same content")
        old_id = t.id
        result = mem.update_node(old_id, "same content")
        assert result.ref.id == old_id  # no change
        assert result.merged is False

    def test_update_nonexistent_raises(self):
        """Updating a nonexistent node raises KeyError."""
        mem = Memory()
        with pytest.raises(KeyError):
            mem.update_node("nonexistent_id", "new content")

    def test_update_writes_audit_trail(self):
        """Update writes to audit log with old/new content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "mem.json")
            mem = Memory.load_or_create(path)
            t = mem.thought("before update")
            mem.update_node(t.id, "after update", reason="consolidation merge")

            audit_path = path.replace(".json", ".audit.jsonl")
            assert Path(audit_path).exists()
            entries = [json.loads(line) for line in Path(audit_path).read_text().splitlines()]
            update_entries = [e for e in entries if e["event"] == "update_node"]
            assert len(update_entries) == 1
            assert update_entries[0]["data"]["old_content"] == "before update"
            assert update_entries[0]["data"]["new_content"] == "after update"
            assert update_entries[0]["data"]["reason"] == "consolidation merge"

    def test_update_with_multiple_relationships(self):
        """Node with relationships as both source AND target updates correctly."""
        mem = Memory()
        a = mem.thought("upstream cause")
        b = mem.thought("the node to update")
        c = mem.thought("downstream effect")

        mem.relate(a, b, RelationType.CAUSES)  # b is target
        mem.relate(b, c, RelationType.CAUSES)  # b is source

        old_b_id = b.id
        updated = mem.update_node(old_b_id, "updated middle node").ref

        # Both relationships should now reference updated.id
        incoming = [r for r in mem._relationships if r.target == updated.id]
        outgoing = [r for r in mem._relationships if r.source == updated.id]
        assert len(incoming) == 1  # a → updated
        assert len(outgoing) == 1  # updated → c
        assert incoming[0].source == a.id
        assert outgoing[0].target == c.id

    def test_update_tension_preserves_axis(self):
        """Tension relationships preserve axis_label through update."""
        mem = Memory()
        a = mem.thought("microservices")
        b = mem.thought("monolith")
        mem.tension(a, b, "architecture_complexity")

        updated_a = mem.update_node(a.id, "microservices with service mesh").ref

        tensions = [r for r in mem._relationships if r.type.value == "tension"]
        assert len(tensions) == 1
        assert tensions[0].axis_label == "architecture_complexity"
        assert tensions[0].source == updated_a.id
        assert tensions[0].target == b.id

    def test_update_content_hash_collision(self):
        """If update content matches ANOTHER existing node, merge into it."""
        mem = Memory()
        a = mem.thought("node A")
        b = mem.thought("node B")
        c = mem.thought("downstream")
        mem.relate(a, c, RelationType.CAUSES)  # a → c

        # Update a's content to match b's content exactly
        result = mem.update_node(a.id, "node B")
        assert result.ref.id == b.id  # merged into b
        assert result.merged is True
        assert result.old_id == a.id
        assert mem.size == 2  # a removed, b and c remain

        # The relationship should now be b → c
        rels = [r for r in mem._relationships if r.source == b.id]
        assert len(rels) == 1
        assert rels[0].target == c.id

    def test_update_collision_writes_audit(self):
        """Content-hash collision merge writes audit trail entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "mem.json")
            mem = Memory.load_or_create(path)
            a = mem.thought("node A content")
            b = mem.thought("node B content")
            # Update a to match b's content → triggers merge
            mem.update_node(a.id, "node B content", reason="merge test")

            audit_path = path.replace(".json", ".audit.jsonl")
            assert Path(audit_path).exists()
            entries = [json.loads(line) for line in Path(audit_path).read_text().splitlines()]
            merge_entries = [e for e in entries if e["event"] == "update_node_merge"]
            assert len(merge_entries) == 1
            assert merge_entries[0]["data"]["merged_into"] == b.id
            assert merge_entries[0]["data"]["reason"] == "merge test"

    def test_update_collision_deduplicates_relationships(self):
        """When merging nodes that both relate to the same third node, no duplicate rels."""
        mem = Memory()
        a = mem.thought("node A")
        b = mem.thought("node B")
        c = mem.thought("shared target")

        mem.relate(a, c, RelationType.CAUSES)  # a → c
        mem.relate(b, c, RelationType.CAUSES)  # b → c

        assert mem.relationship_count == 2

        # Merge a into b (update a's content to match b)
        mem.update_node(a.id, "node B")

        # Should have ONE relationship (b → c), not two duplicates
        rels_to_c = [r for r in mem._relationships if r.target == c.id]
        assert len(rels_to_c) == 1
        assert rels_to_c[0].source == b.id

    def test_update_collision_preserves_target_relationships(self):
        """When merging into existing node, that node's pre-existing relationships survive."""
        mem = Memory()
        a = mem.thought("node A")
        b = mem.thought("node B")
        c = mem.thought("only related to B")

        mem.relate(b, c, RelationType.CAUSES)  # b → c (pre-existing on target)

        # Merge a into b
        result = mem.update_node(a.id, "node B")
        assert result.ref.id == b.id

        # b's relationship to c should survive
        rels = [r for r in mem._relationships if r.source == b.id]
        assert len(rels) == 1
        assert rels[0].target == c.id

    def test_update_collision_deduplicates_states(self):
        """When merging nodes that both have the same state type, no duplicate states."""
        mem = Memory()
        a = mem.thought("node A blocked")
        b = mem.thought("node B blocked")
        a_ref = mem.ref(a.id)
        b_ref = mem.ref(b.id)
        a_ref.block("reason A")
        b_ref.block("reason B")

        # Both are blocked — merge a into b
        mem.update_node(a.id, "node B blocked")

        # Should have exactly ONE blocked state on b, not two
        blocked_states = [s for s in mem._states if s.type.value == "blocked"]
        assert len(blocked_states) == 1

    def test_update_collision_merges_temporal(self):
        """Merge preserves the richer temporal history (min created, max tier, sum freq)."""
        mem = Memory()
        mem.session_start()
        a = mem.thought("old established knowledge")
        b = mem.thought("new recent knowledge")

        # Artificially make a's temporal data richer
        mem._temporal_map[a.id] = mem._temporal_map[a.id].__class__(
            created_at="2026-01-01T00:00:00+00:00",
            last_touched="2026-03-15T00:00:00+00:00",
            frequency=15,
            tier="proven",
        )

        # Merge a into b — b should get a's richer history
        mem.update_node(a.id, "new recent knowledge")
        merged_temporal = mem.get_temporal(b.id)

        assert merged_temporal.created_at == "2026-01-01T00:00:00+00:00"  # min
        assert merged_temporal.frequency == 16  # 15 + 1
        assert merged_temporal.tier == "proven"  # highest tier preserved

    def test_update_collision_filters_self_referential(self):
        """Merging A into B removes any A→B relationship (would become B→B)."""
        mem = Memory()
        a = mem.thought("node A")
        b = mem.thought("node B")
        mem.relate(a, b, RelationType.CAUSES)  # a → b

        # Merge a into b — the a→b relationship would become b→b (self-ref)
        mem.update_node(a.id, "node B")

        # Self-referential relationship should be filtered out
        self_refs = [r for r in mem._relationships if r.source == r.target]
        assert len(self_refs) == 0

    def test_update_round_trip_persistence(self):
        """Updated nodes survive save/load round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "mem.json")
            mem = Memory.load_or_create(path)
            a = mem.thought("original")
            b = mem.thought("related")
            mem.relate(a, b, RelationType.CAUSES)
            updated = mem.update_node(a.id, "modified").ref
            mem.save()

            mem2 = Memory.load(path)
            assert mem2.size == 2
            node = mem2.get_node(updated.id)
            assert node is not None
            assert node.content == "modified"

            # Relationship survived
            rels = [r for r in mem2._relationships if r.source == updated.id]
            assert len(rels) == 1
