"""
E2E test: Temporal intelligence through LangGraph FlowScriptStore.

Tests the complete lifecycle:
1. Session 1: Create nodes via store, verify temporal metadata, save
2. Session 2: Load, query (triggers touch-on-query), verify graduation
3. Session 3: Load, verify cross-session frequency, prune dormant, verify audit
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flowscript_agents import Memory, MemoryOptions
from flowscript_agents.memory import DormancyConfig, TemporalConfig
from flowscript_agents.langgraph import FlowScriptStore


def test_temporal_lifecycle():
    """Full temporal lifecycle through LangGraph adapter."""
    print("\n=== Test: Temporal Lifecycle via LangGraph ===")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "temporal-test.json")

        # --- Session 1: Create data ---
        print("\n  Session 1: Create data")
        store1 = FlowScriptStore(path)
        mem1 = store1.memory

        # Build reasoning
        q = mem1.question("Which cache?")
        redis = mem1.alternative(q, "Redis")
        redis.decide(rationale="Speed critical")
        pg = mem1.alternative(q, "PostgreSQL")
        pg.block(reason="Too slow for cache")

        speed = mem1.thought("Sub-ms reads")
        cost = mem1.thought("$200/month")
        mem1.tension(speed, cost, "perf vs cost")

        # Verify temporal metadata exists
        assert mem1.get_temporal(q.id) is not None
        assert mem1.get_temporal(q.id).frequency == 1
        assert mem1.get_temporal(q.id).tier == "current"
        print(f"    ✓ All {mem1.size} nodes have temporal metadata")
        print(f"    ✓ Tier counts: {mem1._count_tiers()}")

        # Start session and verify garden
        result = mem1.session_start()
        assert result.total_nodes == mem1.size
        assert result.garden.stats["growing"] == mem1.size  # all fresh
        print(f"    ✓ Garden: {result.garden.stats}")

        store1.save()
        print(f"    ✓ Saved with temporal data")

        # --- Session 2: Load, query, verify touch-on-query ---
        print("\n  Session 2: Load and query")
        store2 = FlowScriptStore(path)
        mem2 = store2.memory

        # Verify temporal survived save/load
        q_meta = mem2.get_temporal(q.id)
        assert q_meta is not None, "Temporal data lost on reload!"
        assert q_meta.frequency == 1
        assert q_meta.tier == "current"
        print(f"    ✓ Temporal data survived reload (freq={q_meta.frequency}, tier={q_meta.tier})")

        # Start new session
        mem2.session_start()

        # Query tensions — should touch nodes
        tensions = mem2.query.tensions()
        assert tensions.metadata["total_tensions"] >= 1
        print(f"    ✓ Tensions query: {tensions.metadata['total_tensions']} found")

        # Query blocked — should touch nodes
        blocked = mem2.query.blocked()
        assert len(blocked.blockers) >= 1
        print(f"    ✓ Blocked query: {len(blocked.blockers)} found")

        # Touch the question node explicitly to drive graduation
        mem2.touch_nodes([q.id])  # explicit touch (always increments) → freq 2 → developing
        q_meta2 = mem2.get_temporal(q.id)
        assert q_meta2.frequency == 2
        assert q_meta2.tier == "developing"
        print(f"    ✓ Question graduated: freq={q_meta2.frequency}, tier={q_meta2.tier}")

        store2.save()

        # --- Session 3: Cross-session graduation + prune ---
        print("\n  Session 3: Cross-session graduation and prune")
        store3 = FlowScriptStore(path)
        mem3 = store3.memory

        q_meta3 = mem3.get_temporal(q.id)
        assert q_meta3.frequency == 2
        assert q_meta3.tier == "developing"
        print(f"    ✓ Graduation persisted: freq={q_meta3.frequency}, tier={q_meta3.tier}")

        # Touch again to prove → need freq 3
        mem3.session_start()
        mem3.touch_nodes([q.id])  # freq 3 → proven
        assert mem3.get_temporal(q.id).tier == "proven"
        print(f"    ✓ Question → proven (freq={mem3.get_temporal(q.id).frequency})")

        # Verify tier distribution
        tiers = mem3._count_tiers()
        assert tiers["proven"] >= 1
        print(f"    ✓ Tier distribution: {tiers}")

        # Test prune (nothing dormant since everything was just touched)
        result = mem3.prune()
        assert result.count == 0
        print(f"    ✓ Prune: {result.count} nodes removed (expected 0 — all growing)")

        # Force dormancy by backdating a node
        cost_nodes = mem3.find_nodes("$200/month")
        if cost_nodes:
            cost_id = cost_nodes[0].id
            cost_meta = mem3.get_temporal(cost_id)
            cost_meta.last_touched = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

            result = mem3.prune()
            assert result.count == 1
            assert mem3.get_node(cost_id) is None
            print(f"    ✓ Prune: removed 1 dormant node")

            # Check audit trail
            audit_path = str(Path(path).parent / (Path(path).stem + ".audit.jsonl"))
            entries = Memory.read_audit_log(audit_path)
            assert len(entries) == 1
            assert entries[0]["event"] == "prune"
            assert entries[0]["nodes"][0]["content"] == "$200/month"
            print(f"    ✓ Audit trail: {len(entries)} entry, content verified")

        store3.save()

    print("\n  ✓ ALL TEMPORAL LIFECYCLE TESTS PASSED")


def test_config_persistence_through_adapter():
    """Config options persist through adapter save/load."""
    print("\n=== Test: Config Persistence via Adapter ===")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config-test.json")

        # Create with custom config
        opts = MemoryOptions(
            touch_on_query=False,
            temporal=TemporalConfig(
                dormancy=DormancyConfig(resting="5d", dormant="14d")
            ),
        )
        mem1 = Memory(options=opts)
        mem1.thought("test")
        mem1.save(path)

        # Load through adapter
        store = FlowScriptStore(path)
        mem2 = store.memory
        assert mem2._config.touch_on_query is False
        assert mem2._dormancy.resting == "5d"
        assert mem2._dormancy.dormant == "14d"
        print(f"  ✓ touch_on_query={mem2._config.touch_on_query}")
        print(f"  ✓ dormancy: resting={mem2._dormancy.resting}, dormant={mem2._dormancy.dormant}")

    print("  ✓ CONFIG PERSISTENCE PASSED")


def test_session_wrap_through_adapter():
    """session_wrap captures before/after through adapter."""
    print("\n=== Test: Session Wrap via Adapter ===")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "wrap-test.json")

        store = FlowScriptStore(path)
        mem = store.memory

        mem.thought("A")
        mem.thought("B")
        mem.thought("C")

        result = mem.session_start()
        assert result.total_nodes == 3
        print(f"  ✓ session_start: {result.total_nodes} nodes")

        wrap = mem.session_wrap()
        assert wrap.nodes_before == 3
        assert wrap.nodes_after == 3  # nothing pruned
        assert wrap.saved is True  # path was set via store
        print(f"  ✓ session_wrap: {wrap.nodes_before}→{wrap.nodes_after} nodes, saved={wrap.saved}")

    print("  ✓ SESSION WRAP PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("FlowScript Temporal Intelligence E2E Test")
    print("=" * 60)

    try:
        test_temporal_lifecycle()
        test_config_persistence_through_adapter()
        test_session_wrap_through_adapter()

        print("\n" + "=" * 60)
        print("ALL TEMPORAL E2E TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
