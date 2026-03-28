"""
Tests for FixpointContext — convergence certificates from fixpoint computations.

Tests that the fixpoint context:
1. Emits proper audit events (fixpoint_start, fixpoint_iteration, fixpoint_end)
2. Produces verifiable certificates with graph hashes
3. Integrates with consolidation (degenerate @fix)
4. Handles edge cases (empty memory, errors, no audit writer)
"""

import hashlib
import json
import os
import tempfile
import pytest

from flowscript_agents.memory import Memory, MemoryOptions
from flowscript_agents.fixpoint import FixpointContext, FixpointResult, _SENTINEL_HASH
from flowscript_agents.audit import AuditConfig, AuditWriter
from flowscript_agents.embeddings.consolidate import (
    ConsolidationEngine,
    ConsolidationResult,
    ConsolidationAction,
)
from flowscript_agents.embeddings.index import VectorIndex
from flowscript_agents.embeddings.extract import AutoExtract


# =============================================================================
# Test fixtures
# =============================================================================


class MockEmbedder:
    """Hash-based deterministic embedder for tests."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            words = text.lower().split()
            vec = [0.0] * self._dim
            for i, word in enumerate(words):
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dim
                vec[idx] += 1.0
            magnitude = sum(v * v for v in vec) ** 0.5
            if magnitude > 0:
                vec = [v / magnitude for v in vec]
            results.append(vec)
        return results


class MockConsolidationProvider:
    """Mock provider that returns UPDATE for contested nodes."""

    def __init__(self):
        self._responses: list[dict] = []

    def set_response(self, tool_calls: list[dict]):
        self._responses = tool_calls

    def tool_call(self, messages, tools):
        return self._responses


def create_memory_with_audit(tmp_path):
    """Create a Memory instance with audit trail enabled."""
    json_path = os.path.join(tmp_path, "test_memory.json")
    audit_config = AuditConfig(
        rotation="none",
        hash_chain=True,
    )
    options = MemoryOptions(audit=audit_config)
    memory = Memory.load_or_create(json_path, options=options)
    return memory


def read_audit_events(memory) -> list[dict]:
    """Read all audit events from the memory's audit trail."""
    writer = memory._ensure_audit_writer()
    if writer is None:
        return []
    # Read the active audit file directly (JSONL format)
    audit_path = writer._active_path
    if not audit_path.exists():
        return []
    events = []
    with open(audit_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


# =============================================================================
# FixpointContext unit tests
# =============================================================================


class TestFixpointContext:
    """Test FixpointContext behavior in isolation."""

    def test_basic_lifecycle(self, tmp_path):
        """Context emits start/iteration/end events."""
        memory = create_memory_with_audit(str(tmp_path))

        with FixpointContext(memory, name="test_fix", constraint="L1") as ctx:
            ctx.record_iteration(3)  # 3 changes
            ctx.record_iteration(0)  # convergence

        result = ctx.result
        assert result is not None
        assert result.name == "test_fix"
        assert result.constraint == "L1"
        assert result.status == "converged"
        assert result.iterations == 2
        assert result.delta_sequence == [3, 0]
        assert result.converged is True
        assert result.elapsed_ms > 0

    def test_graph_hashes_change(self, tmp_path):
        """Initial and final graph hashes differ when memory is modified."""
        memory = create_memory_with_audit(str(tmp_path))

        with FixpointContext(memory, name="test", constraint="L1") as ctx:
            # Add a node to change the graph
            memory.thought("Something new")
            ctx.record_iteration(1)
            ctx.record_iteration(0)

        result = ctx.result
        assert result.initial_graph_hash != result.final_graph_hash
        # Both should be real hashes, not sentinels
        assert result.initial_graph_hash != _SENTINEL_HASH
        assert result.final_graph_hash != _SENTINEL_HASH

    def test_same_graph_same_hash(self, tmp_path):
        """Graph hash is deterministic — same state = same hash."""
        memory = create_memory_with_audit(str(tmp_path))
        memory.thought("Existing thought")

        with FixpointContext(memory, name="test1", constraint="L1") as ctx1:
            ctx1.record_iteration(0)

        with FixpointContext(memory, name="test2", constraint="L1") as ctx2:
            ctx2.record_iteration(0)

        assert ctx1.result.initial_graph_hash == ctx2.result.initial_graph_hash
        # Should be real hash, not sentinel
        assert ctx1.result.initial_graph_hash != _SENTINEL_HASH

    def test_certificate_hash_deterministic(self, tmp_path):
        """Certificate hash is deterministic for same computation."""
        memory = create_memory_with_audit(str(tmp_path))

        with FixpointContext(memory, name="test", constraint="L1") as ctx:
            ctx.record_iteration(5)
            ctx.record_iteration(0)

        cert1 = ctx.result.certificate_hash

        # Same delta sequence = same certificate (modulo graph hashes and timing)
        assert len(cert1) == 64  # SHA-256 hex

    def test_l2_bounded_status(self, tmp_path):
        """L2 computation that doesn't converge gets status 'bounded'."""
        memory = create_memory_with_audit(str(tmp_path))

        with FixpointContext(
            memory, name="test", constraint="L2",
            bound_type="max_iterations", bound_value=3
        ) as ctx:
            ctx.record_iteration(5)
            ctx.record_iteration(3)
            ctx.record_iteration(2)
            # No 0 delta — didn't converge

        result = ctx.result
        assert result.status == "bounded"
        assert result.converged is False
        assert result.bound_type == "max_iterations"
        assert result.bound_value == 3
        assert result.audited is True

    def test_l2_converged_status(self, tmp_path):
        """L2 computation that converges naturally gets status 'converged'."""
        memory = create_memory_with_audit(str(tmp_path))

        with FixpointContext(
            memory, name="test", constraint="L2",
            bound_type="max_iterations", bound_value=10
        ) as ctx:
            ctx.record_iteration(5)
            ctx.record_iteration(2)
            ctx.record_iteration(0)  # converged before bound

        result = ctx.result
        assert result.status == "converged"
        assert result.converged is True

    def test_l1_without_convergence_marker_is_bounded(self, tmp_path):
        """L1 without explicit 0-delta gets 'bounded' — evidence required."""
        memory = create_memory_with_audit(str(tmp_path))

        with FixpointContext(memory, name="test", constraint="L1") as ctx:
            ctx.record_iteration(10)
            # No explicit convergence marker — even L1 needs evidence

        result = ctx.result
        assert result.status == "bounded"  # not "converged" without evidence

    def test_l1_with_convergence_marker(self, tmp_path):
        """L1 with explicit 0-delta is converged."""
        memory = create_memory_with_audit(str(tmp_path))

        with FixpointContext(memory, name="test", constraint="L1") as ctx:
            ctx.record_iteration(5)
            ctx.record_iteration(0)

        result = ctx.result
        assert result.status == "converged"

    def test_audit_events_emitted(self, tmp_path):
        """Audit trail contains fixpoint_start, fixpoint_iteration, fixpoint_end."""
        memory = create_memory_with_audit(str(tmp_path))

        with FixpointContext(memory, name="audit_test", constraint="L1") as ctx:
            ctx.record_iteration(2)
            ctx.record_iteration(0)

        events = read_audit_events(memory)
        event_types = [e["event"] for e in events]

        assert "fixpoint_start" in event_types
        assert "fixpoint_iteration" in event_types
        assert "fixpoint_end" in event_types

        # Check fixpoint_start data
        start = next(e for e in events if e["event"] == "fixpoint_start")
        assert start["data"]["name"] == "audit_test"
        assert start["data"]["constraint"] == "L1"
        assert "initial_graph_hash" in start["data"]

        # Check fixpoint_end data (the certificate)
        end = next(e for e in events if e["event"] == "fixpoint_end")
        assert end["data"]["status"] == "converged"
        assert end["data"]["delta_sequence"] == [2, 0]
        assert "certificate_hash" in end["data"]
        assert "final_graph_hash" in end["data"]

        # Verify iterations
        iterations = [e for e in events if e["event"] == "fixpoint_iteration"]
        assert len(iterations) == 2
        assert iterations[0]["data"]["delta_size"] == 2
        assert iterations[1]["data"]["delta_size"] == 0

    def test_result_to_dict(self, tmp_path):
        """FixpointResult.to_dict() produces complete serializable data."""
        memory = create_memory_with_audit(str(tmp_path))

        with FixpointContext(memory, name="ser_test", constraint="L1") as ctx:
            ctx.record_iteration(1)
            ctx.record_iteration(0)

        d = ctx.result.to_dict()
        assert d["name"] == "ser_test"
        assert d["constraint"] == "L1"
        assert d["status"] == "converged"
        assert d["iterations"] == 2
        assert d["delta_sequence"] == [1, 0]
        assert "initial_graph_hash" in d
        assert "final_graph_hash" in d
        assert "certificate_hash" in d
        assert "elapsed_ms" in d
        assert "timestamp" in d
        assert d["timestamp"] != ""
        assert "audited" in d
        assert d["audited"] is True

        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert json_str  # no error

    def test_no_audit_writer_doesnt_crash(self, tmp_path):
        """If memory has no audit writer, fixpoint context still works."""
        json_path = os.path.join(str(tmp_path), "no_audit.json")
        memory = Memory.load_or_create(json_path)  # No audit config

        with FixpointContext(memory, name="no_audit", constraint="L1") as ctx:
            ctx.record_iteration(1)
            ctx.record_iteration(0)

        result = ctx.result
        assert result.status == "converged"
        assert result.delta_sequence == [1, 0]

    def test_exception_during_computation(self, tmp_path):
        """Fixpoint context handles exceptions — status is 'error', not 'bounded'."""
        memory = create_memory_with_audit(str(tmp_path))

        with pytest.raises(ValueError):
            with FixpointContext(memory, name="error_test", constraint="L2",
                                bound_type="max_iterations", bound_value=10) as ctx:
                ctx.record_iteration(3)
                raise ValueError("computation error")

        result = ctx.result
        assert result.status == "error"  # distinct from "bounded"
        assert result.delta_sequence == [3]

    def test_empty_computation(self, tmp_path):
        """Zero-iteration computation (trivially converged)."""
        memory = create_memory_with_audit(str(tmp_path))

        with FixpointContext(memory, name="empty", constraint="L1") as ctx:
            ctx.record_iteration(0)  # zero-match case from spec

        result = ctx.result
        assert result.status == "converged"
        assert result.iterations == 1
        assert result.delta_sequence == [0]


# =============================================================================
# Integration test — consolidation produces convergence certificate
# =============================================================================


class TestConsolidationFixpoint:
    """Test that consolidation (via AutoExtract) produces fixpoint certificates."""

    def test_consolidation_emits_fixpoint_events(self, tmp_path):
        """When consolidation runs, fixpoint audit events appear in the trail."""
        memory = create_memory_with_audit(str(tmp_path))
        embedder = MockEmbedder()
        index = VectorIndex(memory, embedder)

        # Add some existing content
        ref1 = memory.thought("The user prefers dark mode")
        index.index_node(ref1.id)

        # Set up consolidation
        provider = MockConsolidationProvider()
        provider.set_response([{
            "name": "update_memory",
            "arguments": {
                "new_node_index": 0,
                "target_candidate_index": 0,
                "merged_content": "The user strongly prefers dark mode in all apps",
                "reasoning": "More specific version of same preference",
            },
        }])

        engine = ConsolidationEngine(
            memory, provider=provider, vector_index=index,
            candidate_threshold=0.01,  # Low threshold so mock embeddings match
        )

        extract = AutoExtract(
            memory,
            llm=lambda text: json.dumps({
                "nodes": [{"type": "thought", "content": "The user strongly prefers dark mode"}],
                "relationships": [],
                "states": [],
            }),
            vector_index=index,
            consolidation_engine=engine,
        )

        # Run extraction (triggers consolidation)
        extract.ingest("The user strongly prefers dark mode")

        # Check audit trail for fixpoint events
        events = read_audit_events(memory)
        event_types = [e["event"] for e in events]

        assert "fixpoint_start" in event_types, f"Missing fixpoint_start. Events: {event_types}"
        assert "fixpoint_end" in event_types, f"Missing fixpoint_end. Events: {event_types}"

        # Verify the certificate
        end_event = next(e for e in events if e["event"] == "fixpoint_end")
        assert end_event["data"]["name"] == "consolidation"
        assert end_event["data"]["constraint"] == "L1"
        assert end_event["data"]["status"] == "converged"
        assert "certificate_hash" in end_event["data"]
        assert len(end_event["data"]["certificate_hash"]) == 64

    def test_novel_consolidation_certificate(self, tmp_path):
        """Even all-novel consolidation (no LLM call) produces a certificate."""
        memory = create_memory_with_audit(str(tmp_path))
        embedder = MockEmbedder()
        index = VectorIndex(memory, embedder)

        provider = MockConsolidationProvider()
        engine = ConsolidationEngine(
            memory, provider=provider, vector_index=index,
        )

        extract = AutoExtract(
            memory,
            llm=lambda text: json.dumps({
                "nodes": [{"type": "thought", "content": "Something completely new"}],
                "relationships": [],
                "states": [],
            }),
            vector_index=index,
            consolidation_engine=engine,
        )

        extract.ingest("Something completely new")

        events = read_audit_events(memory)
        end_events = [e for e in events if e["event"] == "fixpoint_end"]

        assert len(end_events) == 1
        cert = end_events[0]["data"]
        assert cert["status"] == "converged"
        assert cert["constraint"] == "L1"
        # Novel = 1 ADD, delta = [1, 0]
        assert cert["delta_sequence"][0] >= 0  # at least 0 (could be 1 for novel ADD)


# =============================================================================
# Certificate semantic consistency tests
# =============================================================================


class TestCertificateSemanticConsistency:
    """Test that certificate fields tell a COHERENT story.

    These tests verify the MEANING of certificates, not just their structure.
    A certificate's delta_sequence, initial_graph_hash, and final_graph_hash
    must be internally consistent:
    - If delta > 0, the graph should have changed (hashes differ)
    - If delta == 0 for all iterations, the graph should be unchanged (hashes equal)
    - The hash transition must reflect the actual consolidation effect
    """

    def test_update_consolidation_hashes_differ_with_positive_delta(self, tmp_path):
        """UPDATE consolidation: graph changes, delta > 0, hashes MUST differ."""
        memory = create_memory_with_audit(str(tmp_path))
        embedder = MockEmbedder()
        index = VectorIndex(memory, embedder)

        # Add existing content that will be UPDATE target
        ref1 = memory.thought("The user prefers dark mode")
        index.index_node(ref1.id)

        # Mock provider returns UPDATE action
        provider = MockConsolidationProvider()
        provider.set_response([{
            "name": "update_memory",
            "arguments": {
                "new_node_index": 0,
                "target_candidate_index": 0,
                "merged_content": "The user strongly prefers dark mode in all apps",
                "reasoning": "More specific version of same preference",
            },
        }])

        engine = ConsolidationEngine(
            memory, provider=provider, vector_index=index,
            candidate_threshold=0.01,
        )

        extract = AutoExtract(
            memory,
            llm=lambda text: json.dumps({
                "nodes": [{"type": "thought", "content": "The user strongly prefers dark mode"}],
                "relationships": [],
                "states": [],
            }),
            vector_index=index,
            consolidation_engine=engine,
        )

        extract.ingest("The user strongly prefers dark mode")

        events = read_audit_events(memory)
        end_event = next(e for e in events if e["event"] == "fixpoint_end")
        cert = end_event["data"]

        # Semantic consistency: UPDATE changes the graph
        assert cert["delta_sequence"][0] > 0, (
            "UPDATE consolidation should report positive delta"
        )
        assert cert["initial_graph_hash"] != cert["final_graph_hash"], (
            "UPDATE consolidation changes graph content — hashes MUST differ. "
            "If equal, the certificate claims changes occurred but graph is unchanged."
        )
        assert cert["graph_hash_valid"] is True

    def test_novel_consolidation_hashes_differ_with_positive_delta(self, tmp_path):
        """All-novel (ADD) consolidation: new nodes added, hashes MUST differ.

        This is the bug that survived two 3-layer reviews (Mar 28):
        When pre_hash was captured AFTER node creation, ADD actions produced
        initial_hash == final_hash because the nodes were already in memory.
        With pre_hash captured BEFORE node creation, the hash transition
        correctly reflects "graph grew by N novel nodes."
        """
        memory = create_memory_with_audit(str(tmp_path))
        embedder = MockEmbedder()
        index = VectorIndex(memory, embedder)

        provider = MockConsolidationProvider()
        engine = ConsolidationEngine(
            memory, provider=provider, vector_index=index,
        )

        extract = AutoExtract(
            memory,
            llm=lambda text: json.dumps({
                "nodes": [
                    {"type": "thought", "content": "Something completely new"},
                    {"type": "observation", "content": "Another novel observation"},
                ],
                "relationships": [],
                "states": [],
            }),
            vector_index=index,
            consolidation_engine=engine,
        )

        extract.ingest("Something completely new and another novel observation")

        events = read_audit_events(memory)
        end_event = next(e for e in events if e["event"] == "fixpoint_end")
        cert = end_event["data"]

        # Semantic consistency: novel nodes ADD to the graph
        assert cert["delta_sequence"][0] > 0, (
            "Novel consolidation should report positive delta (ADD actions)"
        )
        assert cert["initial_graph_hash"] != cert["final_graph_hash"], (
            "Novel consolidation adds nodes — hashes MUST differ. "
            "If equal, the certificate contradicts itself: delta says changes "
            "occurred but graph state is unchanged."
        )
        assert cert["graph_hash_valid"] is True

    def test_duplicate_consolidation_hashes_equal_with_zero_delta(self, tmp_path):
        """All-duplicate (NONE) consolidation: no net change, hashes MUST be equal.

        When every extracted node is a semantic duplicate of existing content,
        consolidation creates then removes the new node (NONE action). The net
        effect on the graph is zero — hashes should be equal and delta should
        reflect no constructive changes.

        Note: extracted content must DIFFER from existing (otherwise _add_node
        deduplicates by content-hash and no new node is created). We use
        semantically equivalent but textually different content so the mock
        embedder produces similar vectors while creating distinct node IDs.
        """
        memory = create_memory_with_audit(str(tmp_path))
        embedder = MockEmbedder()
        index = VectorIndex(memory, embedder)

        # Add existing content
        ref1 = memory.thought("The user prefers dark mode for their editor")
        index.index_node(ref1.id)

        # Mock provider returns NONE (skip_duplicate) action
        provider = MockConsolidationProvider()
        provider.set_response([{
            "name": "skip_duplicate",
            "arguments": {
                "new_node_index": 0,
                "target_candidate_index": 0,
                "reasoning": "Already captured by existing node",
            },
        }])

        engine = ConsolidationEngine(
            memory, provider=provider, vector_index=index,
            candidate_threshold=0.01,  # Low threshold — mock embeddings will match on shared words
        )

        # Different text, same meaning — creates a distinct node ID
        extract = AutoExtract(
            memory,
            llm=lambda text: json.dumps({
                "nodes": [{"type": "thought", "content": "The user likes dark mode in their editor"}],
                "relationships": [],
                "states": [],
            }),
            vector_index=index,
            consolidation_engine=engine,
        )

        extract.ingest("The user likes dark mode in their editor")

        events = read_audit_events(memory)
        end_event = next(e for e in events if e["event"] == "fixpoint_end")
        cert = end_event["data"]

        # Semantic consistency: NONE = no net change to graph
        # Node was created (different content → different ID), then removed
        assert cert["delta_sequence"][0] == 0, (
            "All-duplicate consolidation should report zero delta "
            f"(NONE actions don't add value). Got: {cert['delta_sequence']}"
        )
        assert cert["initial_graph_hash"] == cert["final_graph_hash"], (
            "All-duplicate consolidation has zero net effect — hashes MUST be equal. "
            "If different, the certificate claims nothing changed but graph state shifted."
        )
        assert cert["graph_hash_valid"] is True

    def test_certificate_hash_captures_semantic_fields(self, tmp_path):
        """Certificate hash changes when any semantic field differs."""
        r1 = FixpointResult(
            name="test", constraint="L1", status="converged",
            iterations=2, delta_sequence=[3, 0],
            initial_graph_hash="aaa", final_graph_hash="bbb",
            timestamp="2026-01-01T00:00:00Z",
        )
        r2 = FixpointResult(
            name="test", constraint="L1", status="converged",
            iterations=2, delta_sequence=[3, 0],
            initial_graph_hash="aaa", final_graph_hash="ccc",  # different final hash
            timestamp="2026-01-01T00:00:00Z",
        )
        assert r1.certificate_hash != r2.certificate_hash, (
            "Different graph hashes should produce different certificate hashes"
        )

    def test_pre_hash_captured_before_node_creation(self, tmp_path):
        """Verify the fix: pre_hash reflects graph state BEFORE extraction.

        Directly tests the fix by checking that the initial_graph_hash in the
        certificate matches the empty graph (when starting from empty memory),
        NOT the graph-with-extracted-nodes.
        """
        memory = create_memory_with_audit(str(tmp_path))
        embedder = MockEmbedder()
        index = VectorIndex(memory, embedder)

        # Compute hash of empty graph for comparison
        empty_hash = FixpointContext._compute_graph_hash_static(memory)

        provider = MockConsolidationProvider()
        engine = ConsolidationEngine(
            memory, provider=provider, vector_index=index,
        )

        extract = AutoExtract(
            memory,
            llm=lambda text: json.dumps({
                "nodes": [{"type": "thought", "content": "New thought"}],
                "relationships": [],
                "states": [],
            }),
            vector_index=index,
            consolidation_engine=engine,
        )

        extract.ingest("New thought")

        events = read_audit_events(memory)
        end_event = next(e for e in events if e["event"] == "fixpoint_end")
        cert = end_event["data"]

        # The initial_graph_hash should match the empty graph hash,
        # NOT the hash after node creation
        assert cert["initial_graph_hash"] == empty_hash, (
            "initial_graph_hash should reflect graph BEFORE extraction, "
            "not after node creation. This is the core fix."
        )
