"""Tests for temporal intelligence: graduation, garden, session lifecycle, prune, config persistence."""

import json
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from flowscript_agents import Memory, MemoryOptions, TemporalConfig, DormancyConfig


class TestTemporalMeta:
    """Nodes get temporal metadata on creation."""

    def test_node_gets_temporal_meta(self):
        mem = Memory()
        ref = mem.thought("test")
        meta = mem.get_temporal(ref.id)
        assert meta is not None
        assert meta.frequency == 1
        assert meta.tier == "current"
        assert meta.created_at is not None
        assert meta.last_touched is not None

    def test_dedup_does_not_create_new_meta(self):
        mem = Memory()
        ref1 = mem.thought("test")
        ref2 = mem.thought("test")  # same content = same node
        assert ref1.id == ref2.id
        meta = mem.get_temporal(ref1.id)
        assert meta.frequency == 1  # not incremented by dedup

    def test_different_nodes_get_different_meta(self):
        mem = Memory()
        a = mem.thought("A")
        b = mem.thought("B")
        assert mem.get_temporal(a.id) is not mem.get_temporal(b.id)


class TestGraduation:
    """Frequency-based tier promotion."""

    def test_graduation_current_to_developing(self):
        mem = Memory()
        ref = mem.thought("test")
        meta = mem.get_temporal(ref.id)
        assert meta.tier == "current"
        # Touch to reach threshold (default: 2 for current→developing)
        mem.touch_nodes([ref.id])  # freq goes 1→2
        assert meta.tier == "developing"

    def test_graduation_developing_to_proven(self):
        mem = Memory()
        ref = mem.thought("test")
        meta = mem.get_temporal(ref.id)
        # Get to developing first
        mem.touch_nodes([ref.id])  # 1→2, graduates to developing
        assert meta.tier == "developing"
        # Touch to proven threshold (default: 3)
        mem.touch_nodes([ref.id])  # 2→3, graduates to proven
        assert meta.tier == "proven"

    def test_graduation_proven_to_foundation(self):
        mem = Memory()
        ref = mem.thought("test")
        meta = mem.get_temporal(ref.id)
        # Get to proven
        mem.touch_nodes([ref.id])  # → developing
        mem.touch_nodes([ref.id])  # → proven
        assert meta.tier == "proven"
        # Touch to foundation threshold (default: 5)
        mem.touch_nodes([ref.id])  # freq 4
        assert meta.tier == "proven"
        mem.touch_nodes([ref.id])  # freq 5, → foundation
        assert meta.tier == "foundation"

    def test_foundation_stays_foundation(self):
        mem = Memory()
        ref = mem.thought("test")
        meta = mem.get_temporal(ref.id)
        # Get to foundation
        for _ in range(4):
            mem.touch_nodes([ref.id])
        assert meta.tier == "foundation"
        # More touches don't change tier
        mem.touch_nodes([ref.id])
        assert meta.tier == "foundation"

    def test_custom_graduation_thresholds(self):
        from flowscript_agents.memory import TemporalTierConfig
        opts = MemoryOptions(
            temporal=TemporalConfig(
                tiers={
                    "developing": TemporalTierConfig(graduation_threshold=5),
                }
            )
        )
        mem = Memory(options=opts)
        ref = mem.thought("test")
        meta = mem.get_temporal(ref.id)
        # Default threshold is 2, but we set 5
        mem.touch_nodes([ref.id])  # freq 2
        assert meta.tier == "current"  # not yet
        mem.touch_nodes([ref.id])  # freq 3
        mem.touch_nodes([ref.id])  # freq 4
        assert meta.tier == "current"
        mem.touch_nodes([ref.id])  # freq 5, now graduates
        assert meta.tier == "developing"


class TestSessionDedup:
    """Session-scoped touch deduplication."""

    def test_session_touch_dedup(self):
        mem = Memory()
        ref = mem.thought("test")
        mem.session_start()
        meta = mem.get_temporal(ref.id)
        initial_freq = meta.frequency
        # Session-scoped touches should max +1
        mem._touch_nodes_session_scoped([ref.id])
        mem._touch_nodes_session_scoped([ref.id])
        mem._touch_nodes_session_scoped([ref.id])
        assert meta.frequency == initial_freq + 1  # only +1

    def test_session_reset_allows_new_touches(self):
        mem = Memory()
        ref = mem.thought("test")
        mem.session_start()
        mem._touch_nodes_session_scoped([ref.id])
        freq_after_first = mem.get_temporal(ref.id).frequency
        # New session
        mem.session_start()
        mem._touch_nodes_session_scoped([ref.id])
        assert mem.get_temporal(ref.id).frequency == freq_after_first + 1

    def test_explicit_touch_always_increments(self):
        mem = Memory()
        ref = mem.thought("test")
        mem.session_start()
        mem.touch_nodes([ref.id])
        mem.touch_nodes([ref.id])
        mem.touch_nodes([ref.id])
        # Explicit touches are NOT session-scoped
        assert mem.get_temporal(ref.id).frequency == 4  # 1 (creation) + 3

    def test_touch_always_updates_last_touched(self):
        mem = Memory()
        ref = mem.thought("test")
        mem.session_start()
        first_touched = mem.get_temporal(ref.id).last_touched
        time.sleep(0.01)
        mem._touch_nodes_session_scoped([ref.id])
        second_touched = mem.get_temporal(ref.id).last_touched
        assert second_touched >= first_touched
        # Even deduped touches update lastTouched
        time.sleep(0.01)
        mem._touch_nodes_session_scoped([ref.id])
        third_touched = mem.get_temporal(ref.id).last_touched
        assert third_touched >= second_touched


class TestGarden:
    """Garden classification based on lastTouched."""

    def test_new_nodes_are_growing(self):
        mem = Memory()
        mem.thought("A")
        mem.thought("B")
        garden = mem.garden()
        assert garden.stats["growing"] == 2
        assert garden.stats["resting"] == 0
        assert garden.stats["dormant"] == 0

    def test_garden_skips_blocks(self):
        mem = Memory()
        mem.thought("A")
        mem.group("container")
        garden = mem.garden()
        assert garden.stats["total"] == 1  # block not counted

    def test_old_nodes_become_dormant(self):
        mem = Memory(options=MemoryOptions(
            temporal=TemporalConfig(
                dormancy=DormancyConfig(resting="1ms", dormant="2ms")
            )
        ))
        ref = mem.thought("old")
        # Manually backdate the lastTouched
        meta = mem.get_temporal(ref.id)
        old_time = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        meta.last_touched = old_time
        time.sleep(0.01)  # ensure we're past the dormancy threshold
        garden = mem.garden()
        assert ref.id in garden.dormant


class TestSessionLifecycle:
    """session_start, session_end, session_wrap."""

    def test_session_start_returns_summary(self):
        mem = Memory()
        mem.thought("A")
        mem.thought("B")
        result = mem.session_start()
        assert result.total_nodes == 2
        assert result.tier_counts["current"] == 2
        assert result.garden.stats["growing"] == 2

    def test_session_end_prunes_and_saves(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            mem = Memory()
            mem._file_path = str(Path(path).resolve())
            mem.thought("test")
            result = mem.session_end()
            assert result.saved is True
            assert result.path is not None
            assert Path(result.path).exists()

    def test_session_wrap_captures_before_after(self):
        mem = Memory()
        mem.thought("A")
        mem.thought("B")
        result = mem.session_wrap()
        assert result.nodes_before == 2
        assert result.nodes_after == 2  # nothing pruned (all growing)
        assert result.tiers_before["current"] == 2

    def test_session_end_no_path_no_save(self):
        mem = Memory()
        mem.thought("test")
        result = mem.session_end()
        assert result.saved is False
        assert result.path is None


class TestPrune:
    """Prune dormant nodes with audit trail."""

    def test_prune_removes_dormant(self):
        mem = Memory(options=MemoryOptions(
            temporal=TemporalConfig(
                dormancy=DormancyConfig(resting="1ms", dormant="2ms")
            )
        ))
        ref = mem.thought("old")
        meta = mem.get_temporal(ref.id)
        meta.last_touched = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        time.sleep(0.01)

        assert mem.size == 1
        result = mem.prune()
        assert result.count == 1
        assert mem.size == 0

    def test_prune_writes_audit(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            mem = Memory(options=MemoryOptions(
                temporal=TemporalConfig(
                    dormancy=DormancyConfig(resting="1ms", dormant="2ms")
                )
            ))
            mem._file_path = path
            ref = mem.thought("audited")
            meta = mem.get_temporal(ref.id)
            meta.last_touched = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
            time.sleep(0.01)

            mem.prune()

            # Check audit log
            audit_path = str(Path(td) / "mem.audit.jsonl")
            assert Path(audit_path).exists()
            entries = Memory.read_audit_log(audit_path)
            assert len(entries) == 1
            assert entries[0]["event"] == "prune"
            assert len(entries[0]["nodes"]) == 1
            assert entries[0]["nodes"][0]["content"] == "audited"

    def test_prune_preserves_growing(self):
        mem = Memory()
        mem.thought("fresh")
        result = mem.prune()
        assert result.count == 0
        assert mem.size == 1

    def test_prune_removes_associated_data(self):
        mem = Memory(options=MemoryOptions(
            temporal=TemporalConfig(
                dormancy=DormancyConfig(resting="1ms", dormant="2ms")
            )
        ))
        a = mem.thought("old")
        b = mem.thought("also old")
        a.causes(b)
        a.block(reason="test", since="2020-01-01")

        # Backdate both
        for ref in [a, b]:
            meta = mem.get_temporal(ref.id)
            meta.last_touched = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        time.sleep(0.01)

        mem.prune()
        assert mem.size == 0
        ir = mem.to_ir()
        assert len(ir.relationships) == 0
        assert len(ir.states) == 0


class TestConfigPersistence:
    """Config survives save/load cycles."""

    def test_config_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            opts = MemoryOptions(
                touch_on_query=False,
                temporal=TemporalConfig(
                    dormancy=DormancyConfig(resting="5d", dormant="14d", archive="60d")
                ),
            )
            mem = Memory(options=opts)
            mem.thought("test")
            mem.save(path)

            mem2 = Memory.load(path)
            assert mem2._config.touch_on_query is False
            assert mem2._dormancy.resting == "5d"
            assert mem2._dormancy.dormant == "14d"

    def test_author_config_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            opts = MemoryOptions(
                author={"agent": "my-agent", "role": "ai"},
            )
            mem = Memory(options=opts)
            mem.thought("test")
            mem.save(path)

            mem2 = Memory.load(path)
            assert mem2._config.author is not None
            assert mem2._config.author["agent"] == "my-agent"
            assert mem2._config.author["role"] == "ai"

    def test_temporal_meta_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            mem = Memory()
            ref = mem.thought("tracked")
            mem.touch_nodes([ref.id])  # freq → 2, tier → developing
            mem.save(path)

            mem2 = Memory.load(path)
            meta = mem2.get_temporal(ref.id)
            assert meta is not None
            assert meta.frequency == 2
            assert meta.tier == "developing"

    def test_load_or_create_respects_options(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "new.json")
            opts = MemoryOptions(touch_on_query=False)
            mem = Memory.load_or_create(path, options=opts)
            assert mem._config.touch_on_query is False


class TestGroup:
    """group() method creates structural block nodes."""

    def test_group_creates_block_node(self):
        mem = Memory()
        ref = mem.group("my container")
        assert ref.type.value == "block"
        assert ref.content == "my container"

    def test_group_excluded_from_garden(self):
        mem = Memory()
        mem.group("container")
        mem.thought("real node")
        garden = mem.garden()
        assert garden.stats["total"] == 1  # only the thought

    def test_group_not_pruned(self):
        """Block nodes are excluded from garden, so they're never dormant → never pruned."""
        mem = Memory(options=MemoryOptions(
            temporal=TemporalConfig(
                dormancy=DormancyConfig(resting="1ms", dormant="2ms")
            )
        ))
        grp = mem.group("container")
        meta = mem.get_temporal(grp.id)
        meta.last_touched = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        time.sleep(0.01)
        mem.prune()
        assert mem.size == 1  # group still here


class TestCountTiers:
    """_count_tiers helper."""

    def test_count_tiers(self):
        mem = Memory()
        mem.thought("A")
        mem.thought("B")
        ref = mem.thought("C")
        mem.touch_nodes([ref.id])  # graduates to developing
        counts = mem._count_tiers()
        assert counts["current"] == 2
        assert counts["developing"] == 1
        assert counts["proven"] == 0
        assert counts["foundation"] == 0


class TestTouchOnQuery:
    """Query operations touch returned nodes when enabled."""

    def test_query_touches_nodes(self):
        mem = Memory()
        a = mem.thought("A")
        b = mem.thought("B")
        mem.tension(a, b, "test_axis")
        mem.session_start()

        initial_freq = mem.get_temporal(a.id).frequency
        mem.query.tensions()
        # Should have been touched
        assert mem.get_temporal(a.id).frequency >= initial_freq

    def test_touch_on_query_disabled(self):
        mem = Memory(options=MemoryOptions(touch_on_query=False))
        a = mem.thought("A")
        b = mem.thought("B")
        mem.tension(a, b, "test_axis")
        mem.session_start()

        initial_freq = mem.get_temporal(a.id).frequency
        mem.query.tensions()
        # Should NOT have been touched
        assert mem.get_temporal(a.id).frequency == initial_freq


class TestE2ELifecycle:
    """Multi-session lifecycle — the critical E2E test."""

    def test_multi_session_graduation(self):
        """Nodes graduate across sessions, not within a single session."""
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")

            # Session 1: Create nodes
            mem = Memory()
            a = mem.thought("important idea")
            b = mem.thought("less important")
            mem.session_start()
            mem.query.tensions()  # touch within session
            mem.query.tensions()  # deduped within session
            assert mem.get_temporal(a.id).frequency == 1  # creation only
            mem.save(path)

            # Session 2: Load and engage
            mem2 = Memory.load(path)
            a_id = a.id
            mem2.session_start()
            mem2._touch_nodes_session_scoped([a_id])
            assert mem2.get_temporal(a_id).frequency == 2
            assert mem2.get_temporal(a_id).tier == "developing"
            mem2.save(path)

            # Session 3: More engagement
            mem3 = Memory.load(path)
            mem3.session_start()
            mem3._touch_nodes_session_scoped([a_id])
            assert mem3.get_temporal(a_id).frequency == 3
            assert mem3.get_temporal(a_id).tier == "proven"

            # b is still current (never touched beyond creation)
            assert mem3.get_temporal(b.id).tier == "current"


class TestLegacyIRLoading:
    """Loading legacy IR-only format (v0.1.0 files) gets temporal metadata."""

    def test_from_ir_initializes_temporal(self):
        """from_ir() should create temporal metadata for all loaded nodes."""
        from flowscript_agents.types import IR, Node, NodeType, Provenance
        ir = IR(
            nodes=[
                Node(id="abc123", type=NodeType.THOUGHT, content="legacy node",
                     provenance=Provenance(source_file="test", line_number=1, timestamp="2026-01-01T00:00:00Z"))
            ],
            relationships=[],
            states=[],
        )
        mem = Memory.from_ir(ir)
        meta = mem.get_temporal("abc123")
        assert meta is not None
        assert meta.frequency == 1
        assert meta.tier == "current"

    def test_legacy_json_loads_with_temporal(self):
        """Loading legacy IR-only JSON (no flowscript_memory key) gets temporal."""
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "legacy.json")
            # Write a legacy IR-only JSON file (what v0.1.0 saved)
            from flowscript_agents.types import IR, Node, NodeType, Provenance
            ir = IR(
                nodes=[
                    Node(id="legacy1", type=NodeType.THOUGHT, content="old node",
                         provenance=Provenance(source_file="test", line_number=1, timestamp="2026-01-01T00:00:00Z"))
                ],
                relationships=[],
                states=[],
            )
            Path(path).write_text(json.dumps(ir.model_dump(mode="json", exclude_none=True)))
            mem = Memory.load(path)
            assert mem.size == 1
            meta = mem.get_temporal("legacy1")
            assert meta is not None
            assert meta.tier == "current"
            # Prune should NOT remove it (it's freshly loaded = growing)
            result = mem.prune()
            assert result.count == 0

    def test_legacy_load_then_session_end_safe(self):
        """Loading legacy file then calling session_end() should NOT prune everything."""
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "legacy2.json")
            from flowscript_agents.types import IR, Node, NodeType, Provenance
            ir = IR(
                nodes=[
                    Node(id="n1", type=NodeType.THOUGHT, content="node 1",
                         provenance=Provenance(source_file="test", line_number=1, timestamp="2026-01-01T00:00:00Z")),
                    Node(id="n2", type=NodeType.STATEMENT, content="node 2",
                         provenance=Provenance(source_file="test", line_number=2, timestamp="2026-01-01T00:00:00Z")),
                ],
                relationships=[],
                states=[],
            )
            Path(path).write_text(json.dumps(ir.model_dump(mode="json", exclude_none=True)))
            mem = Memory.load(path)
            result = mem.session_end()
            assert result.pruned.count == 0  # nothing pruned!
            assert mem.size == 2  # both nodes survive


class TestTiersConfigPersistence:
    """Custom graduation thresholds survive save/load."""

    def test_tiers_config_round_trip(self):
        from flowscript_agents.memory import TemporalTierConfig
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "tiers.json")
            opts = MemoryOptions(
                temporal=TemporalConfig(
                    tiers={
                        "developing": TemporalTierConfig(graduation_threshold=5),
                        "proven": TemporalTierConfig(graduation_threshold=10),
                    },
                    dormancy=DormancyConfig(resting="5d", dormant="14d"),
                )
            )
            mem = Memory(options=opts)
            ref = mem.thought("test")
            mem.save(path)

            mem2 = Memory.load(path)
            # Verify thresholds restored
            assert mem2._get_graduation_threshold("current") == 5  # developing threshold
            assert mem2._get_graduation_threshold("developing") == 10  # proven threshold
            assert mem2._get_graduation_threshold("proven") == 5  # foundation default (not set)

            # Verify graduation uses restored thresholds
            meta = mem2.get_temporal(ref.id)
            mem2.touch_nodes([ref.id])  # freq 2
            assert meta.tier == "current"  # needs 5, only at 2
            mem2.touch_nodes([ref.id])  # freq 3
            mem2.touch_nodes([ref.id])  # freq 4
            assert meta.tier == "current"  # still needs 5
            mem2.touch_nodes([ref.id])  # freq 5 → developing
            assert meta.tier == "developing"
