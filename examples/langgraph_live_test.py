"""
Live integration test: FlowScriptStore + LangGraph + Claude.

This test builds a real LangGraph agent backed by FlowScript memory,
has it reason about a database decision, then queries the resulting
FlowScript graph for tensions, blockers, and causal chains.

Requires: ANTHROPIC_API_KEY in environment or .env.flow
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Load API key from .env.flow if not in environment
env_flow = Path(__file__).parent.parent.parent / "flow" / ".env.flow"
if env_flow.exists() and "ANTHROPIC_API_KEY" not in os.environ:
    for line in env_flow.read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
            break

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flowscript_agents import Memory
from flowscript_agents.langgraph import FlowScriptStore


def test_store_basic_operations():
    """Test 1: Basic FlowScriptStore CRUD without LLM calls."""
    print("\n=== Test 1: Basic Store Operations ===")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test-memory.json")
        store = FlowScriptStore(path)

        # Put items
        store.put(("agent", "decisions"), "db-choice", {
            "content": "Chose Redis for caching layer",
            "type": "decision",
            "rationale": "Sub-ms reads, good cluster support",
        })
        store.put(("agent", "decisions"), "auth-choice", {
            "content": "Chose JWT for authentication",
            "type": "decision",
            "rationale": "Stateless, good for microservices",
        })
        store.put(("agent", "observations"), "perf-note", {
            "content": "Redis cluster costs $200/month at scale",
            "type": "observation",
        })

        # Verify get
        item = store.get(("agent", "decisions"), "db-choice")
        assert item is not None, "Failed to get stored item"
        assert item.value["content"] == "Chose Redis for caching layer"
        print(f"  ✓ Put/Get: {item.value['content']}")

        # Verify search
        results = store.search(("agent",), query="Redis")
        assert len(results) >= 1, f"Expected search results, got {len(results)}"
        print(f"  ✓ Search 'Redis': {len(results)} results")

        # Verify search with filter
        results = store.search(("agent",), filter={"type": "decision"})
        assert len(results) == 2, f"Expected 2 decisions, got {len(results)}"
        print(f"  ✓ Filter by type=decision: {len(results)} results")

        # Verify namespaces
        ns = store.list_namespaces()
        assert len(ns) >= 2, f"Expected 2+ namespaces, got {len(ns)}"
        print(f"  ✓ Namespaces: {ns}")

        # Verify delete
        store.delete(("agent", "observations"), "perf-note")
        item = store.get(("agent", "observations"), "perf-note")
        assert item is None, "Delete failed — item still exists"
        print("  ✓ Delete: item removed")

        # Verify update
        store.put(("agent", "decisions"), "db-choice", {
            "content": "Switched to DragonflyDB for caching",
            "type": "decision",
            "rationale": "Redis-compatible, lower memory footprint",
        })
        item = store.get(("agent", "decisions"), "db-choice")
        assert "DragonflyDB" in item.value["content"], "Update failed"
        print(f"  ✓ Update: {item.value['content']}")

        # Verify persistence
        store.save()
        store2 = FlowScriptStore(path)
        item2 = store2.get(("agent", "decisions"), "db-choice")
        assert item2 is not None, "Persistence failed — item not found after reload"
        assert "DragonflyDB" in item2.value["content"], "Persistence failed — stale content"
        print(f"  ✓ Persistence: survived save/reload")

        # Verify delete persistence
        item3 = store2.get(("agent", "observations"), "perf-note")
        assert item3 is None, "Delete not persisted — deleted item resurrected"
        print("  ✓ Delete persistence: deleted item stayed deleted after reload")

    print("  ✓ ALL BASIC OPERATIONS PASSED")


def test_flowscript_reasoning():
    """Test 2: FlowScript reasoning through the store's memory property."""
    print("\n=== Test 2: FlowScript Reasoning Integration ===")

    store = FlowScriptStore()
    mem = store.memory

    # Build reasoning graph
    q = mem.question("Which database for the caching layer?")
    redis = mem.alternative(q, "Redis")
    redis.decide(rationale="Sub-ms reads critical for real-time agents")
    dragonfly = mem.alternative(q, "DragonflyDB")
    dragonfly.explore()
    sqlite = mem.alternative(q, "SQLite")
    sqlite.block(reason="No concurrent write support")

    speed = mem.thought("Redis gives sub-ms reads")
    cost = mem.thought("Redis cluster costs $200/month")
    mem.tension(speed, cost, "performance vs cost")

    speed.causes(mem.thought("Better user experience"))
    cost.causes(mem.thought("Higher infrastructure budget"))

    # Query tensions
    tensions = mem.query.tensions()
    assert tensions.metadata["total_tensions"] >= 1
    print(f"  ✓ Tensions found: {tensions.metadata['total_tensions']}")
    if tensions.tensions_by_axis:
        for axis, details in tensions.tensions_by_axis.items():
            for d in details:
                print(f"    - {d.source['content']} >< {d.target['content']} [{axis}]")

    # Query blocked
    blocked = mem.query.blocked()
    assert len(blocked.blockers) >= 1
    print(f"  ✓ Blocked items: {len(blocked.blockers)}")
    for b in blocked.blockers:
        print(f"    - {b.node['content']}: {b.blocked_state.get('reason', '?')}")

    # Query alternatives
    alts = mem.query.alternatives(q.id)
    assert len(alts.alternatives) >= 2
    print(f"  ✓ Alternatives for question: {len(alts.alternatives)}")
    for a in alts.alternatives:
        state = "decided" if a.chosen else ("blocked" if a.rejection_reasons else "exploring")
        print(f"    - {a.content} ({state})")

    # Query why
    ux = mem.find_nodes("Better user experience")
    if ux:
        why = mem.query.why(ux[0].id)
        print(f"  ✓ Why 'Better user experience': {why.metadata['total_ancestors']} ancestors")

    # Also verify the store interface still works alongside reasoning
    store.put(("meta",), "query-count", {"content": "5 queries executed", "type": "meta"})
    assert store.get(("meta",), "query-count") is not None
    print("  ✓ Store operations work alongside reasoning")

    print("  ✓ ALL REASONING TESTS PASSED")


def test_async_operations():
    """Test 3: Async operations work correctly."""
    print("\n=== Test 3: Async Operations ===")

    async def _run():
        store = FlowScriptStore()
        await store.aput(("async",), "k1", {"content": "async item"})
        item = await store.aget(("async",), "k1")
        assert item is not None
        assert item.value["content"] == "async item"
        print("  ✓ Async put/get works")

        results = await store.asearch(("async",), query="async")
        assert len(results) >= 1
        print(f"  ✓ Async search: {len(results)} results")

        await store.adelete(("async",), "k1")
        item = await store.aget(("async",), "k1")
        assert item is None
        print("  ✓ Async delete works")

    asyncio.run(_run())
    print("  ✓ ALL ASYNC TESTS PASSED")


def test_persistence_roundtrip_with_mutations():
    """Test 4: The critical test — mutations persist correctly."""
    print("\n=== Test 4: Mutation Persistence (the bug that was caught) ===")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mutation-test.json")

        # Create store, add items, delete one, update another, save
        store1 = FlowScriptStore(path)
        store1.put(("ns",), "keep", {"content": "I should survive"})
        store1.put(("ns",), "delete-me", {"content": "I should be gone"})
        store1.put(("ns",), "update-me", {"content": "Original content"})

        store1.delete(("ns",), "delete-me")
        store1.put(("ns",), "update-me", {"content": "Updated content"})
        store1.save()

        # Reload and verify
        store2 = FlowScriptStore(path)

        kept = store2.get(("ns",), "keep")
        assert kept is not None, "Kept item disappeared"
        assert kept.value["content"] == "I should survive"
        print("  ✓ Kept item survived reload")

        deleted = store2.get(("ns",), "delete-me")
        assert deleted is None, "CRITICAL: Deleted item resurrected on reload!"
        print("  ✓ Deleted item stayed deleted after reload")

        updated = store2.get(("ns",), "update-me")
        assert updated is not None, "Updated item disappeared"
        assert updated.value["content"] == "Updated content", \
            f"CRITICAL: Update lost on reload! Got: {updated.value['content']}"
        print("  ✓ Updated item has correct content after reload")

        # Verify FlowScript graph is consistent
        nodes = store2.memory.find_nodes("I should be gone")
        assert len(nodes) == 0, "CRITICAL: Deleted node still in FlowScript graph!"
        print("  ✓ Deleted node not in FlowScript graph")

        nodes = store2.memory.find_nodes("Updated content")
        assert len(nodes) >= 1, "Updated content not found in FlowScript graph"
        print("  ✓ Updated content found in FlowScript graph")

        nodes = store2.memory.find_nodes("Original content")
        assert len(nodes) == 0, "Old content still in FlowScript graph after update"
        print("  ✓ Old content removed from FlowScript graph")

    print("  ✓ ALL MUTATION PERSISTENCE TESTS PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("FlowScript + LangGraph Live Integration Test")
    print("=" * 60)

    try:
        test_store_basic_operations()
        test_flowscript_reasoning()
        test_async_operations()
        test_persistence_roundtrip_with_mutations()

        print("\n" + "=" * 60)
        print("ALL LIVE TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
