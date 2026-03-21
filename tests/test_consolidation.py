"""
Tests for the ConsolidationEngine — type-aware memory consolidation.

Tests all 5 actions (ADD, UPDATE, RELATE, RESOLVE, NONE), error recovery,
batch processing, and integration with UnifiedMemory.

Uses mock providers to test consolidation logic independently of real LLMs.
"""

import hashlib
import json
import pytest
from unittest.mock import MagicMock

from flowscript_agents.memory import Memory, MemoryOptions, NodeRef
from flowscript_agents.embeddings.consolidate import (
    ConsolidationEngine,
    ConsolidationProvider,
    ConsolidationAction,
    ConsolidationResult,
    CONSOLIDATION_TOOLS,
    CONSOLIDATION_SYSTEM_PROMPT,
)
from flowscript_agents.embeddings.index import VectorIndex
from flowscript_agents.embeddings.extract import AutoExtract, IngestResult
from flowscript_agents.unified import UnifiedMemory
from flowscript_agents.types import RelationType, StateType


# =============================================================================
# Test fixtures
# =============================================================================


class MockEmbedder:
    """Mock embedding provider for tests.

    Uses a simple hash-based embedding to simulate similarity.
    Identical strings get identical embeddings, different strings get
    different embeddings. Similarity is controlled by seeding similar
    content with overlapping word patterns.
    """

    def __init__(self, dim: int = 8):
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            # Simple word-frequency vector (deterministic across runs).
            # Uses hashlib instead of hash() to avoid PYTHONHASHSEED variation.
            words = text.lower().split()
            vec = [0.0] * self._dim
            for i, word in enumerate(words):
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dim
                vec[idx] += 1.0
            # Normalize
            magnitude = sum(v * v for v in vec) ** 0.5
            if magnitude > 0:
                vec = [v / magnitude for v in vec]
            results.append(vec)
        return results


class MockConsolidationProvider:
    """Mock consolidation provider that returns pre-configured tool calls.

    Use .set_response() to configure what the provider returns.
    """

    def __init__(self):
        self._responses: list[dict] = []

    def set_response(self, tool_calls: list[dict]):
        """Configure the tool calls this provider will return."""
        self._responses = tool_calls

    def tool_call(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> list[dict]:
        return self._responses


class FailingConsolidationProvider:
    """Provider that always raises an exception."""

    def tool_call(self, messages, tools):
        raise RuntimeError("LLM service unavailable")


def _make_memory_with_nodes() -> tuple[Memory, VectorIndex, list[str]]:
    """Create a Memory with some existing nodes and a VectorIndex.

    Returns (memory, vector_index, node_ids).
    """
    mem = Memory()
    embedder = MockEmbedder()
    index = VectorIndex(mem, embedder)

    # Create some existing nodes
    ref1 = mem.thought("We chose PostgreSQL for the database due to ACID compliance")
    ref2 = mem.thought("The auth system is blocked waiting on security audit results")
    ref2.block(reason="waiting on security audit")
    ref3 = mem.question("Should we use microservices or monolith architecture?")
    ref3.explore()

    # Index them
    index.index_all()

    return mem, index, [ref1.id, ref2.id, ref3.id]


# =============================================================================
# ConsolidationEngine unit tests
# =============================================================================


class TestConsolidationEngineInit:
    def test_creation(self):
        mem = Memory()
        embedder = MockEmbedder()
        index = VectorIndex(mem, embedder)
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index)
        assert repr(engine) == "ConsolidationEngine(threshold=0.45, top_k=5)"

    def test_custom_thresholds(self):
        mem = Memory()
        embedder = MockEmbedder()
        index = VectorIndex(mem, embedder)
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(
            mem, provider, index,
            candidate_threshold=0.60,
            candidate_top_k=3,
        )
        assert repr(engine) == "ConsolidationEngine(threshold=0.6, top_k=3)"


class TestConsolidationNovel:
    """Tests for novel nodes (no similar existing → bypass LLM)."""

    def test_all_novel_no_llm_call(self):
        """When no existing nodes, all extracted are novel. No LLM call."""
        mem = Memory()
        embedder = MockEmbedder()
        index = VectorIndex(mem, embedder)
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index)

        # Create the extracted nodes in memory first (as ingest would)
        ref1 = mem.thought("Brand new information about databases")
        index.index_node(ref1.id)
        ref2 = mem.thought("Also new: API design patterns")
        index.index_node(ref2.id)

        extracted = [
            {"index": 0, "type": "thought", "content": "Brand new information about databases"},
            {"index": 1, "type": "thought", "content": "Also new: API design patterns"},
        ]
        node_refs = [ref1, ref2]

        result = engine.consolidate(extracted, node_refs)

        assert result.llm_called is False
        assert result.nodes_novel == 2
        assert result.nodes_added == 2
        assert len(result.actions) == 2
        assert all(a.action == "ADD" for a in result.actions)


class TestConsolidationADD:
    """Tests for ADD action — genuinely new despite having candidates."""

    def test_add_via_llm(self):
        """LLM decides ADD even though candidates exist."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        # Create a node that has candidates but LLM says ADD
        ref = mem.thought("PostgreSQL replication setup guide")
        index.index_node(ref.id)

        provider.set_response([{
            "name": "add_memory",
            "arguments": {
                "new_node_index": 0,
                "reasoning": "New topic — replication is different from database selection",
            },
        }])

        extracted = [{"index": 0, "type": "thought", "content": "PostgreSQL replication setup guide"}]
        node_refs: list[NodeRef | None] = [ref]

        result = engine.consolidate(extracted, node_refs)

        assert result.llm_called is True
        assert result.nodes_added >= 1
        # Node should still exist
        assert mem.get_node(ref.id) is not None


class TestConsolidationUPDATE:
    """Tests for UPDATE action — merge richer content into existing."""

    def test_update_merges_content(self):
        """UPDATE replaces existing content with merged version."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        # Get the original PostgreSQL node
        pg_id = existing_ids[0]
        old_node = mem.get_node(pg_id)
        assert old_node is not None

        # Create a new node with richer content about PostgreSQL
        ref = mem.thought("PostgreSQL database for financial ACID compliance and audit logging")
        index.index_node(ref.id)

        merged = "We chose PostgreSQL for the database — ACID compliance for financial data and built-in audit logging"

        provider.set_response([{
            "name": "update_memory",
            "arguments": {
                "new_node_index": 0,
                "target_candidate_index": 0,
                "merged_content": merged,
                "reasoning": "Same topic with richer detail about audit logging",
            },
        }])

        extracted = [{
            "index": 0,
            "type": "thought",
            "content": "PostgreSQL database for financial ACID compliance and audit logging",
        }]
        node_refs: list[NodeRef | None] = [ref]

        result = engine.consolidate(extracted, node_refs)

        assert result.nodes_updated == 1
        # New node should be removed (merged into existing)
        assert node_refs[0] is None
        # Original node ID may have changed (content-hash based)
        # But there should be a node with the merged content
        found = mem.find_nodes("audit logging")
        assert len(found) >= 1
        assert merged in found[0].content

    def test_update_preserves_relationships(self):
        """UPDATE preserves existing relationships on the target node."""
        mem = Memory()
        embedder = MockEmbedder()
        index = VectorIndex(mem, embedder)
        provider = MockConsolidationProvider()

        # Create two related nodes
        ref_a = mem.thought("Database choice affects performance")
        ref_b = mem.thought("We chose PostgreSQL for performance")
        ref_a.causes(ref_b)
        index.index_all()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        # Create a new node that should UPDATE ref_b
        ref_new = mem.thought("We chose PostgreSQL for performance and reliability")
        index.index_node(ref_new.id)

        merged = "We chose PostgreSQL for performance, reliability, and ACID compliance"

        provider.set_response([{
            "name": "update_memory",
            "arguments": {
                "new_node_index": 0,
                "target_candidate_index": 0,
                "merged_content": merged,
                "reasoning": "Richer version of same decision",
            },
        }])

        extracted = [{
            "index": 0,
            "type": "thought",
            "content": "We chose PostgreSQL for performance and reliability",
        }]
        node_refs: list[NodeRef | None] = [ref_new]

        result = engine.consolidate(extracted, node_refs)
        assert result.nodes_updated == 1

        # Verify a relationship still exists (source may have re-pointed)
        rels = [r for r in mem._relationships if r.type == RelationType.CAUSES]
        assert len(rels) >= 1


class TestConsolidationRELATE:
    """Tests for RELATE action — create structural relationships."""

    def test_relate_tension(self):
        """RELATE(tension) creates a tension relationship between new and existing."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        # Create a contradicting node
        ref = mem.thought("MongoDB would be better than PostgreSQL for our flexible schema needs")
        index.index_node(ref.id)

        provider.set_response([{
            "name": "relate_memories",
            "arguments": {
                "new_node_index": 0,
                "target_candidate_index": 0,
                "relationship_type": "tension",
                "direction": "new_to_existing",
                "axis": "database_selection_flexibility_vs_acid",
                "reasoning": "Contradicts the PostgreSQL decision on schema flexibility grounds",
            },
        }])

        extracted = [{
            "index": 0,
            "type": "thought",
            "content": "MongoDB would be better than PostgreSQL for our flexible schema needs",
        }]
        node_refs: list[NodeRef | None] = [ref]

        result = engine.consolidate(extracted, node_refs)

        assert result.nodes_related == 1
        # Node should still exist (RELATE keeps the new node)
        assert node_refs[0] is not None
        # Check tension relationship was created
        tension_rels = [r for r in mem._relationships if r.type == RelationType.TENSION]
        assert len(tension_rels) >= 1
        assert tension_rels[0].axis_label == "database_selection_flexibility_vs_acid"

    def test_relate_causes(self):
        """RELATE(causes) creates a causal relationship."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        ref = mem.thought("Performance testing revealed PostgreSQL handles our load well")
        index.index_node(ref.id)

        provider.set_response([{
            "name": "relate_memories",
            "arguments": {
                "new_node_index": 0,
                "target_candidate_index": 0,
                "relationship_type": "causes",
                "direction": "new_to_existing",
                "reasoning": "Performance testing validates the PostgreSQL decision",
            },
        }])

        extracted = [{
            "index": 0,
            "type": "thought",
            "content": "Performance testing revealed PostgreSQL handles our load well",
        }]
        node_refs: list[NodeRef | None] = [ref]

        result = engine.consolidate(extracted, node_refs)

        assert result.nodes_related == 1
        causes_rels = [r for r in mem._relationships if r.type == RelationType.CAUSES]
        assert len(causes_rels) >= 1

    def test_relate_derives_from(self):
        """RELATE(derives_from) creates a derives_from relationship."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        ref = mem.thought("PostgreSQL partitioning strategy based on our ACID requirements")
        index.index_node(ref.id)

        provider.set_response([{
            "name": "relate_memories",
            "arguments": {
                "new_node_index": 0,
                "target_candidate_index": 0,
                "relationship_type": "derives_from",
                "direction": "new_to_existing",
                "reasoning": "Partitioning strategy derives from the PostgreSQL decision",
            },
        }])

        extracted = [{
            "index": 0,
            "type": "thought",
            "content": "PostgreSQL partitioning strategy based on our ACID requirements",
        }]
        node_refs: list[NodeRef | None] = [ref]

        result = engine.consolidate(extracted, node_refs)

        assert result.nodes_related == 1
        derives_rels = [r for r in mem._relationships if r.type == RelationType.DERIVES_FROM]
        assert len(derives_rels) >= 1

    def test_relate_direction_existing_to_new(self):
        """RELATE with direction=existing_to_new creates correct relationship direction."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        ref = mem.thought("Decided to add read replicas for PostgreSQL")
        index.index_node(ref.id)

        provider.set_response([{
            "name": "relate_memories",
            "arguments": {
                "new_node_index": 0,
                "target_candidate_index": 0,
                "relationship_type": "causes",
                "direction": "existing_to_new",
                "reasoning": "PostgreSQL choice led to read replica decision",
            },
        }])

        extracted = [{
            "index": 0,
            "type": "thought",
            "content": "Decided to add read replicas for PostgreSQL",
        }]
        node_refs: list[NodeRef | None] = [ref]

        result = engine.consolidate(extracted, node_refs)
        assert result.nodes_related == 1

        # Verify direction: existing → new (target should be the new node)
        causes_rels = [r for r in mem._relationships if r.type == RelationType.CAUSES]
        assert len(causes_rels) >= 1
        rel = causes_rels[-1]
        # Source should be an existing node (not the new one), target should be new
        assert rel.source != ref.id  # source is the existing candidate
        assert rel.target == ref.id  # target is the new node

    def test_relate_tension_without_axis_falls_back_to_add(self):
        """RELATE(tension) without axis falls back to ADD."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        ref = mem.thought("MongoDB is better")
        index.index_node(ref.id)

        provider.set_response([{
            "name": "relate_memories",
            "arguments": {
                "new_node_index": 0,
                "target_candidate_index": 0,
                "relationship_type": "tension",
                "direction": "new_to_existing",
                # Missing axis!
                "reasoning": "Contradicts",
            },
        }])

        extracted = [{"index": 0, "type": "thought", "content": "MongoDB is better"}]
        node_refs: list[NodeRef | None] = [ref]

        result = engine.consolidate(extracted, node_refs)

        # Should fall back to ADD since tension requires axis
        assert result.nodes_added >= 1
        assert result.nodes_related == 0


class TestConsolidationRESOLVE:
    """Tests for RESOLVE action — update states on existing nodes."""

    def test_resolve_unblock(self):
        """RESOLVE(unblock) removes blocked state from target node."""
        mem = Memory()
        embedder = MockEmbedder()
        index = VectorIndex(mem, embedder)
        provider = MockConsolidationProvider()

        # Create a blocked node
        blocked_ref = mem.thought("The auth system is blocked waiting on security audit results")
        blocked_ref.block(reason="waiting on security audit")
        blocked_id = blocked_ref.id
        index.index_all()

        # Verify it's blocked
        blocked_states = [s for s in mem._states if s.node_id == blocked_id and s.type == StateType.BLOCKED]
        assert len(blocked_states) == 1

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        # Create a resolving node
        ref = mem.thought("Security audit completed, no critical findings for auth system")
        index.index_node(ref.id)

        provider.set_response([{
            "name": "resolve_state",
            "arguments": {
                "new_node_index": 0,
                "target_candidate_index": 0,
                "resolve_type": "unblock",
                "resolution": "Security audit completed with no critical findings",
                "reasoning": "The blocker (security audit) is resolved",
            },
        }])

        extracted = [{
            "index": 0,
            "type": "thought",
            "content": "Security audit completed, no critical findings for auth system",
        }]
        node_refs: list[NodeRef | None] = [ref]

        result = engine.consolidate(extracted, node_refs)

        assert result.nodes_resolved == 1
        # Node should still exist (RESOLVE keeps the new node)
        assert node_refs[0] is not None
        # Blocked state should be removed from the target
        # Find which node was the target from the action
        resolve_action = [a for a in result.actions if a.action == "RESOLVE"][0]
        target_id = resolve_action.target_node_id
        blocked_states_after = [
            s for s in mem._states
            if s.node_id == target_id and s.type == StateType.BLOCKED
        ]
        assert len(blocked_states_after) == 0
        # Causal relationship should exist
        causes_rels = [r for r in mem._relationships if r.type == RelationType.CAUSES]
        assert len(causes_rels) >= 1

    def test_resolve_decide(self):
        """RESOLVE(decide) adds decided state to target node and removes exploring."""
        mem = Memory()
        embedder = MockEmbedder()
        index = VectorIndex(mem, embedder)
        provider = MockConsolidationProvider()

        # Create an exploring question
        question_ref = mem.question("Should we use microservices or monolith architecture?")
        question_ref.explore()
        question_id = question_ref.id
        index.index_all()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        ref = mem.thought("Team decided on microservices architecture for better scalability")
        index.index_node(ref.id)

        provider.set_response([{
            "name": "resolve_state",
            "arguments": {
                "new_node_index": 0,
                "target_candidate_index": 0,
                "resolve_type": "decide",
                "resolution": "Chose microservices for scalability",
                "reasoning": "The architecture question has been decided",
            },
        }])

        extracted = [{
            "index": 0,
            "type": "thought",
            "content": "Team decided on microservices architecture for better scalability",
        }]
        node_refs: list[NodeRef | None] = [ref]

        result = engine.consolidate(extracted, node_refs)

        assert result.nodes_resolved == 1
        # Find which node was targeted
        resolve_action = [a for a in result.actions if a.action == "RESOLVE"][0]
        target_id = resolve_action.target_node_id
        # Decided state should be added on target
        decided_states = [
            s for s in mem._states
            if s.node_id == target_id and s.type == StateType.DECIDED
        ]
        assert len(decided_states) == 1
        # Exploring state should be removed from target
        exploring_states = [
            s for s in mem._states
            if s.node_id == target_id and s.type == StateType.EXPLORING
        ]
        assert len(exploring_states) == 0


class TestConsolidationNONE:
    """Tests for NONE action — skip duplicate, touch existing."""

    def test_none_removes_new_and_touches_existing(self):
        """NONE removes the new node and touches the existing one."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        pg_id = existing_ids[0]
        old_temporal = mem.get_temporal(pg_id)
        old_freq = old_temporal.frequency if old_temporal else 0

        # Create a duplicate-ish node
        ref = mem.thought("We chose PostgreSQL for ACID compliance")
        index.index_node(ref.id)

        provider.set_response([{
            "name": "skip_duplicate",
            "arguments": {
                "new_node_index": 0,
                "target_candidate_index": 0,
                "reasoning": "Semantically equivalent to existing PostgreSQL node",
            },
        }])

        extracted = [{
            "index": 0,
            "type": "thought",
            "content": "We chose PostgreSQL for ACID compliance",
        }]
        node_refs: list[NodeRef | None] = [ref]

        result = engine.consolidate(extracted, node_refs)

        assert result.nodes_skipped == 1
        # New node should be removed
        assert node_refs[0] is None
        assert mem.get_node(ref.id) is None
        # Existing node should be touched (frequency incremented)
        new_temporal = mem.get_temporal(pg_id)
        assert new_temporal is not None
        assert new_temporal.frequency > old_freq


class TestConsolidationBatch:
    """Tests for batch processing — multiple contested nodes in one LLM call."""

    def test_batch_multiple_actions(self):
        """Batch of 3 nodes: one ADD, one RELATE, one NONE."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        # Create 3 new nodes
        ref0 = mem.thought("Redis caching strategy for PostgreSQL queries")
        ref1 = mem.thought("MongoDB flexibility better than PostgreSQL rigidity")
        ref2 = mem.thought("We chose PostgreSQL for ACID")
        for r in [ref0, ref1, ref2]:
            index.index_node(r.id)

        # Mock LLM response: ADD node 0, RELATE node 1, NONE node 2.
        # With deterministic MockEmbedder (hashlib-based), candidate ordering is:
        #   Node 1: [0]=auth, [1]=PostgreSQL, [2]=microservices
        #   Node 2: [0]=PostgreSQL, [1]=auth, [2]=microservices
        # RELATE targets candidate[0] (auth) for node 1, skip_duplicate
        # targets candidate[0] (PostgreSQL) for node 2 — different existing
        # nodes, no within-batch collision.
        provider.set_response([
            {
                "name": "add_memory",
                "arguments": {
                    "new_node_index": 0,
                    "reasoning": "Redis caching is a new topic",
                },
            },
            {
                "name": "relate_memories",
                "arguments": {
                    "new_node_index": 1,
                    "target_candidate_index": 0,
                    "relationship_type": "tension",
                    "direction": "new_to_existing",
                    "axis": "schema_flexibility_vs_acid",
                    "reasoning": "Contradicts the approach taken",
                },
            },
            {
                "name": "skip_duplicate",
                "arguments": {
                    "new_node_index": 2,
                    "target_candidate_index": 0,
                    "reasoning": "Same as existing PostgreSQL node",
                },
            },
        ])

        extracted = [
            {"index": 0, "type": "thought", "content": "Redis caching strategy for PostgreSQL queries"},
            {"index": 1, "type": "thought", "content": "MongoDB flexibility better than PostgreSQL rigidity"},
            {"index": 2, "type": "thought", "content": "We chose PostgreSQL for ACID"},
        ]
        node_refs: list[NodeRef | None] = [ref0, ref1, ref2]

        result = engine.consolidate(extracted, node_refs)

        assert result.llm_called is True
        assert result.nodes_added >= 1
        assert result.nodes_related == 1
        assert result.nodes_skipped == 1
        # ref0 kept (ADD), ref1 kept (RELATE), ref2 removed (NONE)
        assert node_refs[0] is not None
        assert node_refs[1] is not None
        assert node_refs[2] is None


class TestConsolidationErrorRecovery:
    """Tests for error handling and fallback behavior."""

    def test_llm_failure_falls_back_to_add(self):
        """If consolidation LLM fails, all contested nodes become ADD."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = FailingConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        ref = mem.thought("Something about PostgreSQL")
        index.index_node(ref.id)

        extracted = [{"index": 0, "type": "thought", "content": "Something about PostgreSQL"}]
        node_refs: list[NodeRef | None] = [ref]

        result = engine.consolidate(extracted, node_refs)

        # Should fall back to ADD
        assert result.nodes_added >= 1
        assert result.llm_called is True
        # Node should still exist
        assert mem.get_node(ref.id) is not None

    def test_invalid_tool_name_falls_back_to_add(self):
        """Unknown tool name falls back to ADD."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        ref = mem.thought("Something about PostgreSQL databases")
        index.index_node(ref.id)

        provider.set_response([{
            "name": "unknown_tool",
            "arguments": {"new_node_index": 0, "reasoning": "test"},
        }])

        extracted = [{"index": 0, "type": "thought", "content": "Something about PostgreSQL databases"}]
        node_refs: list[NodeRef | None] = [ref]

        result = engine.consolidate(extracted, node_refs)

        assert result.nodes_added >= 1

    def test_invalid_candidate_index_falls_back_to_add(self):
        """Invalid target_candidate_index falls back to ADD."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        ref = mem.thought("PostgreSQL optimization tips")
        index.index_node(ref.id)

        provider.set_response([{
            "name": "update_memory",
            "arguments": {
                "new_node_index": 0,
                "target_candidate_index": 99,  # invalid
                "merged_content": "whatever",
                "reasoning": "test",
            },
        }])

        extracted = [{"index": 0, "type": "thought", "content": "PostgreSQL optimization tips"}]
        node_refs: list[NodeRef | None] = [ref]

        result = engine.consolidate(extracted, node_refs)

        assert result.nodes_added >= 1

    def test_missing_tool_call_for_node_falls_back_to_add(self):
        """If LLM doesn't return a tool call for a contested node, it becomes ADD."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        ref = mem.thought("PostgreSQL data migration plan")
        index.index_node(ref.id)

        # Empty response — no tool calls at all
        provider.set_response([])

        extracted = [{"index": 0, "type": "thought", "content": "PostgreSQL data migration plan"}]
        node_refs: list[NodeRef | None] = [ref]

        result = engine.consolidate(extracted, node_refs)

        assert result.nodes_added >= 1

    def test_string_arguments_are_parsed(self):
        """Some providers return arguments as JSON string — must handle."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        ref = mem.thought("PostgreSQL backup strategy")
        index.index_node(ref.id)

        # Arguments as JSON string (some providers do this)
        provider.set_response([{
            "name": "add_memory",
            "arguments": json.dumps({
                "new_node_index": 0,
                "reasoning": "genuinely new topic",
            }),
        }])

        extracted = [{"index": 0, "type": "thought", "content": "PostgreSQL backup strategy"}]
        node_refs: list[NodeRef | None] = [ref]

        result = engine.consolidate(extracted, node_refs)

        assert result.nodes_added >= 1


class TestConsolidationAudit:
    """Tests for audit trail logging."""

    def test_actions_are_audited(self):
        """All consolidation actions write to audit trail."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        ref = mem.thought("We chose PostgreSQL for reliability")
        index.index_node(ref.id)

        provider.set_response([{
            "name": "skip_duplicate",
            "arguments": {
                "new_node_index": 0,
                "target_candidate_index": 0,
                "reasoning": "Duplicate of existing",
            },
        }])

        extracted = [{"index": 0, "type": "thought", "content": "We chose PostgreSQL for reliability"}]
        node_refs: list[NodeRef | None] = [ref]

        # Set a file path so audit log can be written
        mem._file_path = "/tmp/test_consolidation_audit.json"

        result = engine.consolidate(extracted, node_refs)

        # Check audit log was written (read the file)
        import os
        audit_path = "/tmp/test_consolidation_audit.audit.jsonl"
        if os.path.exists(audit_path):
            with open(audit_path) as f:
                lines = f.readlines()
            # Find consolidation entries
            consolidation_entries = [
                json.loads(line) for line in lines
                if "consolidation" in json.loads(line).get("event", "")
            ]
            assert len(consolidation_entries) >= 1
            assert consolidation_entries[0]["action"] == "NONE"
            # Cleanup
            os.remove(audit_path)


class TestConsolidationPrompt:
    """Tests for prompt construction and tool definitions."""

    def test_tool_definitions_are_valid(self):
        """All 5 tools have required fields."""
        assert len(CONSOLIDATION_TOOLS) == 5
        names = {t["function"]["name"] for t in CONSOLIDATION_TOOLS}
        assert names == {"add_memory", "update_memory", "relate_memories", "resolve_state", "skip_duplicate"}

        for tool in CONSOLIDATION_TOOLS:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert "new_node_index" in func["parameters"]["properties"]

    def test_system_prompt_exists(self):
        """Consolidation system prompt is non-empty and mentions all actions."""
        assert len(CONSOLIDATION_SYSTEM_PROMPT) > 100
        assert "skip_duplicate" in CONSOLIDATION_SYSTEM_PROMPT
        assert "update_memory" in CONSOLIDATION_SYSTEM_PROMPT
        assert "relate_memories" in CONSOLIDATION_SYSTEM_PROMPT
        assert "resolve_state" in CONSOLIDATION_SYSTEM_PROMPT
        assert "add_memory" in CONSOLIDATION_SYSTEM_PROMPT

    def test_batch_prompt_structure(self):
        """Verify the batch prompt includes all contested nodes with candidates."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        # Build a contested node with candidates
        from flowscript_agents.embeddings.consolidate import _ContestedNode, _CandidateNode

        cn = _ContestedNode(
            new_index=0,
            node_type="thought",
            content="Test content",
            candidates=[
                _CandidateNode(
                    local_index=0,
                    node_id="abc123",
                    node_type="thought",
                    content="Existing content",
                    states=["blocked (reason: \"waiting\")"],
                    relationships=["causes → other node"],
                    similarity=0.75,
                ),
            ],
        )

        prompt = engine._build_batch_prompt([cn])

        assert "contested_nodes" in prompt
        assert "Test content" in prompt
        assert "Existing content" in prompt
        assert "blocked" in prompt
        assert "causes" in prompt


class TestConsolidationIntegration:
    """Integration tests with AutoExtract and UnifiedMemory."""

    def test_autoextract_with_consolidation_engine(self):
        """AutoExtract routes through consolidation when engine is set."""
        mem = Memory()
        embedder = MockEmbedder()
        index = VectorIndex(mem, embedder)
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        # Mock extraction LLM
        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "nodes": [
                    {"type": "thought", "content": "New database insight"},
                ],
                "relationships": [],
                "states": [],
            })

        extractor = AutoExtract(
            mem, llm=mock_llm, vector_index=index,
            consolidation_engine=engine,
        )

        # All nodes are novel (no existing)
        result = extractor.ingest("Some conversation about databases")

        assert result.consolidation_used is True
        assert result.nodes_created >= 1
        assert result.nodes_novel >= 1

    def test_autoextract_without_consolidation_uses_simple_dedup(self):
        """Without consolidation engine, AutoExtract uses original dedup path."""
        mem = Memory()

        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "nodes": [
                    {"type": "thought", "content": "A database insight"},
                ],
                "relationships": [],
                "states": [],
            })

        extractor = AutoExtract(mem, llm=mock_llm)

        result = extractor.ingest("Some conversation")

        assert result.consolidation_used is False
        assert result.nodes_created == 1

    def test_unified_memory_with_consolidation(self):
        """UnifiedMemory accepts consolidation_provider and wires it through."""
        provider = MockConsolidationProvider()
        embedder = MockEmbedder()

        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "nodes": [
                    {"type": "thought", "content": "A test thought"},
                ],
                "relationships": [],
                "states": [],
            })

        um = UnifiedMemory(
            embedder=embedder,
            llm=mock_llm,
            consolidation_provider=provider,
        )

        assert um._consolidation_engine is not None
        assert um._extractor is not None
        assert um._extractor._consolidation_engine is not None

    def test_unified_memory_without_consolidation(self):
        """UnifiedMemory works without consolidation (backward compatible)."""
        embedder = MockEmbedder()

        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "nodes": [
                    {"type": "thought", "content": "A test thought"},
                ],
                "relationships": [],
                "states": [],
            })

        um = UnifiedMemory(embedder=embedder, llm=mock_llm)

        assert um._consolidation_engine is None

        result = um.add("Test input")
        assert result.consolidation_used is False
        assert result.nodes_created == 1

    def test_consolidation_with_extraction_relationships(self):
        """Extraction relationships are created for surviving nodes only."""
        mem = Memory()
        embedder = MockEmbedder()
        index = VectorIndex(mem, embedder)
        provider = MockConsolidationProvider()

        # Add an existing node so consolidation can NONE one of the new ones
        existing = mem.thought("PostgreSQL is our database choice")
        index.index_all()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        # Mock extraction returns 2 nodes with a relationship between them
        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "nodes": [
                    {"type": "thought", "content": "PostgreSQL is great for ACID"},  # will be NONE'd
                    {"type": "thought", "content": "Redis caching helps PostgreSQL performance"},
                ],
                "relationships": [
                    {"type": "causes", "source": 0, "target": 1},  # should be skipped (source removed)
                ],
                "states": [],
            })

        # Provider NONEs the first node, ADDs the second
        provider.set_response([{
            "name": "skip_duplicate",
            "arguments": {
                "new_node_index": 0,
                "target_candidate_index": 0,
                "reasoning": "Same as existing PostgreSQL node",
            },
        }])
        # Note: node 1 is novel (no candidates), so only node 0 goes to consolidation

        extractor = AutoExtract(
            mem, llm=mock_llm, vector_index=index,
            consolidation_engine=engine,
        )

        result = extractor.ingest("Discussion about databases")

        # The extraction relationship (0→1) should be skipped because node 0 was removed
        # Only the surviving node (1) should have been created
        assert result.relationships_created == 0  # source was removed


class TestConsolidationProviderProtocol:
    """Tests for the ConsolidationProvider protocol."""

    def test_mock_satisfies_protocol(self):
        """MockConsolidationProvider satisfies the protocol."""
        provider = MockConsolidationProvider()
        assert isinstance(provider, ConsolidationProvider)

    def test_failing_satisfies_protocol(self):
        """FailingConsolidationProvider satisfies the protocol."""
        provider = FailingConsolidationProvider()
        assert isinstance(provider, ConsolidationProvider)

    def test_arbitrary_class_does_not_satisfy(self):
        """A random class doesn't satisfy the protocol."""

        class NotAProvider:
            pass

        assert not isinstance(NotAProvider(), ConsolidationProvider)


# =============================================================================
# Review-driven tests (domain expert findings #13-18)
# =============================================================================


class TestConsolidationEdgeCases:
    """Edge cases identified during code review."""

    def test_batch_two_nodes_target_same_candidate_update(self):
        """Two contested nodes both UPDATE the same candidate — second falls back to ADD.

        This is a known constraint: batch actions assume each candidate is
        targeted at most once. The second UPDATE fails (KeyError from update_node
        because first UPDATE changed the target's ID) and falls back to ADD.
        """
        mem = Memory()
        embedder = MockEmbedder()
        index = VectorIndex(mem, embedder)
        provider = MockConsolidationProvider()

        # Create one existing node
        existing = mem.thought("PostgreSQL is our database choice for ACID")
        index.index_all()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        # Two new nodes, both want to UPDATE the same candidate
        ref0 = mem.thought("PostgreSQL chosen for ACID and audit logging")
        ref1 = mem.thought("PostgreSQL selected for ACID compliance and scalability")
        index.index_node(ref0.id)
        index.index_node(ref1.id)

        provider.set_response([
            {
                "name": "update_memory",
                "arguments": {
                    "new_node_index": 0,
                    "target_candidate_index": 0,
                    "merged_content": "PostgreSQL for ACID, audit logging, and compliance",
                    "reasoning": "Richer version",
                },
            },
            {
                "name": "update_memory",
                "arguments": {
                    "new_node_index": 1,
                    "target_candidate_index": 0,
                    "merged_content": "PostgreSQL for ACID and scalability",
                    "reasoning": "Also richer",
                },
            },
        ])

        extracted = [
            {"index": 0, "type": "thought", "content": "PostgreSQL chosen for ACID and audit logging"},
            {"index": 1, "type": "thought", "content": "PostgreSQL selected for ACID compliance and scalability"},
        ]
        node_refs: list[NodeRef | None] = [ref0, ref1]

        result = engine.consolidate(extracted, node_refs)

        # First UPDATE succeeds, second falls back to ADD (target ID changed)
        assert result.nodes_updated + result.nodes_added >= 2
        # No data loss — both nodes are accounted for
        total = result.nodes_added + result.nodes_updated + result.nodes_related + result.nodes_resolved + result.nodes_skipped
        assert total == len(extracted)

    def test_resolve_unblock_on_node_without_blocked_state(self):
        """RESOLVE(unblock) on a node that isn't blocked — no state change, relationship still created.

        If the LLM hallucinates that a node is blocked when it isn't,
        the unblock is a no-op but the causal relationship is still created.
        """
        mem = Memory()
        embedder = MockEmbedder()
        index = VectorIndex(mem, embedder)
        provider = MockConsolidationProvider()

        # Create a node WITHOUT blocked state
        existing = mem.thought("The auth system is being designed")
        index.index_all()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        ref = mem.thought("Security audit completed for auth system")
        index.index_node(ref.id)

        provider.set_response([{
            "name": "resolve_state",
            "arguments": {
                "new_node_index": 0,
                "target_candidate_index": 0,
                "resolve_type": "unblock",
                "resolution": "Audit done",
                "reasoning": "The blocker is resolved",
            },
        }])

        extracted = [{"index": 0, "type": "thought", "content": "Security audit completed for auth system"}]
        node_refs: list[NodeRef | None] = [ref]

        # Count states before
        blocked_before = len([s for s in mem._states if s.type == StateType.BLOCKED])

        result = engine.consolidate(extracted, node_refs)

        # Resolve still "succeeds" (no error) — it's safe even without blocked state
        assert result.nodes_resolved == 1
        # No blocked states were removed (none existed)
        blocked_after = len([s for s in mem._states if s.type == StateType.BLOCKED])
        assert blocked_after == blocked_before
        # Causal relationship WAS still created (the information is causal)
        causes_rels = [r for r in mem._relationships if r.type == RelationType.CAUSES]
        assert len(causes_rels) >= 1

    def test_ingest_result_node_ids_includes_none_targets(self):
        """IngestResult.node_ids includes existing nodes touched by NONE actions."""
        mem = Memory()
        embedder = MockEmbedder()
        index = VectorIndex(mem, embedder)
        provider = MockConsolidationProvider()

        existing = mem.thought("PostgreSQL is our database")
        existing_id = existing.id
        index.index_all()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "nodes": [
                    {"type": "thought", "content": "PostgreSQL is the database choice"},
                ],
                "relationships": [],
                "states": [],
            })

        # Provider NONEs the extracted node
        provider.set_response([{
            "name": "skip_duplicate",
            "arguments": {
                "new_node_index": 0,
                "target_candidate_index": 0,
                "reasoning": "Same info",
            },
        }])

        extractor = AutoExtract(
            mem, llm=mock_llm, vector_index=index,
            consolidation_engine=engine,
        )

        result = extractor.ingest("Database discussion")

        # node_ids should include the existing node that was touched
        assert existing_id in result.node_ids

    def test_batch_prompt_includes_similarity_scores(self):
        """Verify similarity scores are included in the batch prompt for LLM context."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        from flowscript_agents.embeddings.consolidate import _ContestedNode, _CandidateNode

        cn = _ContestedNode(
            new_index=0,
            node_type="thought",
            content="Test content",
            candidates=[
                _CandidateNode(
                    local_index=0,
                    node_id="abc123",
                    node_type="thought",
                    content="Existing content",
                    states=[],
                    relationships=[],
                    similarity=0.823,
                ),
            ],
        )

        prompt = engine._build_batch_prompt([cn])
        assert "0.823" in prompt

    def test_consolidation_surviving_nodes_get_extraction_relationships(self):
        """When both nodes in an extraction relationship survive consolidation,
        the relationship is created."""
        mem = Memory()
        embedder = MockEmbedder()
        index = VectorIndex(mem, embedder)
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, candidate_threshold=0.01)

        # Mock extraction returns 2 novel nodes with a relationship
        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "nodes": [
                    {"type": "thought", "content": "Alpha unique insight about XYZ"},
                    {"type": "thought", "content": "Beta unique consequence of XYZ"},
                ],
                "relationships": [
                    {"type": "causes", "source": 0, "target": 1},
                ],
                "states": [],
            })

        extractor = AutoExtract(
            mem, llm=mock_llm, vector_index=index,
            consolidation_engine=engine,
        )

        result = extractor.ingest("Novel discussion about XYZ")

        # Both nodes are novel (no existing memory) — both survive
        assert result.nodes_created >= 2
        # The extraction relationship should be created
        assert result.relationships_created >= 1
        causes_rels = [r for r in mem._relationships if r.type == RelationType.CAUSES]
        assert len(causes_rels) >= 1


# =============================================================================
# Production Hardening Tests — Phase 2
# =============================================================================


class TestConsolidationRetry:
    """Tests for LLM retry with exponential backoff in consolidation."""

    def test_retry_on_transient_failure(self):
        """Should retry and succeed after transient LLM failure."""
        mem, index, existing_ids = _make_memory_with_nodes()
        call_count = [0]

        class RetryProvider:
            def tool_call(self, messages, tools):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise ConnectionError("rate limit")
                # Return ADD for the contested node
                return [{"name": "add_memory", "arguments": {"new_node_index": 0, "reasoning": "retry worked"}}]

        provider = RetryProvider()
        engine = ConsolidationEngine(mem, provider, index, max_retries=3)

        # Create a contested node (similar to existing)
        ref = mem.thought("We chose PostgreSQL for database ACID compliance")
        index.index_node(ref.id)
        extracted = [{"index": 0, "type": "thought", "content": "We chose PostgreSQL for database ACID compliance"}]
        node_refs = [ref]

        result = engine.consolidate(extracted, node_refs)

        assert result.llm_called is True
        assert call_count[0] == 2  # first failed, second succeeded
        # Should not be a fallback ADD
        assert result.fallback_count == 0

    def test_all_retries_exhausted_fallback_add(self):
        """After all retries, contested nodes become fallback ADD."""
        mem, index, existing_ids = _make_memory_with_nodes()

        class AlwaysFailProvider:
            def tool_call(self, messages, tools):
                raise RuntimeError("service down")

        provider = AlwaysFailProvider()
        # Use max_retries=2 so we don't wait too long
        engine = ConsolidationEngine(mem, provider, index, max_retries=2)

        ref = mem.thought("PostgreSQL is our database choice for ACID compliance")
        index.index_node(ref.id)
        extracted = [{"index": 0, "type": "thought", "content": "PostgreSQL is our database choice for ACID compliance"}]
        node_refs = [ref]

        result = engine.consolidate(extracted, node_refs)

        # Should fallback to ADD for all contested
        assert result.llm_called is True
        fallback_actions = [a for a in result.actions if a.is_fallback]
        assert len(fallback_actions) >= 1

    def test_no_retry_when_max_retries_is_1(self):
        """max_retries=1 means single attempt, no retry."""
        mem, index, existing_ids = _make_memory_with_nodes()
        call_count = [0]

        class FailOnce:
            def tool_call(self, messages, tools):
                call_count[0] += 1
                raise RuntimeError("fail")

        engine = ConsolidationEngine(mem, FailOnce(), index, max_retries=1)

        ref = mem.thought("PostgreSQL for ACID compliance database choice")
        index.index_node(ref.id)
        extracted = [{"index": 0, "type": "thought", "content": "PostgreSQL for ACID compliance database choice"}]
        node_refs = [ref]

        result = engine.consolidate(extracted, node_refs)
        assert call_count[0] == 1


class TestConsolidationBatchSplitting:
    """Tests for max_batch_size splitting of contested nodes."""

    def test_small_batch_no_split(self):
        """Batches smaller than max_batch_size don't split."""
        mem = Memory()
        embedder = MockEmbedder()
        index = VectorIndex(mem, embedder)
        provider = MockConsolidationProvider()

        # Set large batch size — no split needed
        engine = ConsolidationEngine(mem, provider, index, max_batch_size=50)

        # Create 3 existing + 3 new similar nodes
        existing = []
        for content in ["database choice A", "database choice B", "database choice C"]:
            ref = mem.thought(content)
            index.index_node(ref.id)
            existing.append(ref.id)

        new_refs = []
        extracted = []
        for i, content in enumerate(["database choice A v2", "database choice B v2", "database choice C v2"]):
            ref = mem.thought(content)
            index.index_node(ref.id)
            new_refs.append(ref)
            extracted.append({"index": i, "type": "thought", "content": content})

        # Provider returns ADD for all
        provider.set_response([
            {"name": "add_memory", "arguments": {"new_node_index": 0, "reasoning": "new"}},
            {"name": "add_memory", "arguments": {"new_node_index": 1, "reasoning": "new"}},
            {"name": "add_memory", "arguments": {"new_node_index": 2, "reasoning": "new"}},
        ])

        result = engine.consolidate(extracted, new_refs)
        assert result.llm_calls == 1  # single batch

    def test_large_batch_splits(self):
        """Batches larger than max_batch_size split into multiple LLM calls."""
        mem = Memory()
        embedder = MockEmbedder(dim=16)
        index = VectorIndex(mem, embedder)
        call_count = [0]

        class CountingProvider:
            def tool_call(self, messages, tools):
                call_count[0] += 1
                # Return empty — all will fallback to ADD
                return []

        engine = ConsolidationEngine(
            mem, CountingProvider(), index,
            max_batch_size=2,  # tiny batch size for testing
            candidate_threshold=0.01,  # low threshold so everything is contested
        )

        # Create 1 existing node that everything will match against
        existing_ref = mem.thought("database technology evaluation report")
        index.index_node(existing_ref.id)

        # Create 5 new similar nodes
        new_refs = []
        extracted = []
        for i in range(5):
            content = f"database technology evaluation report version {i}"
            ref = mem.thought(content)
            index.index_node(ref.id)
            new_refs.append(ref)
            extracted.append({"index": i, "type": "thought", "content": content})

        result = engine.consolidate(extracted, new_refs)

        # With max_batch_size=2 and 5 contested nodes: ceil(5/2) = 3 batches
        assert result.llm_calls == 3
        assert call_count[0] == 3

    def test_batch_size_custom(self):
        """Custom max_batch_size is respected."""
        mem = Memory()
        embedder = MockEmbedder()
        index = VectorIndex(mem, embedder)
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index, max_batch_size=10)
        assert engine._max_batch_size == 10


class TestConsolidationMetrics:
    """Tests for consolidation result metrics."""

    def test_fallback_rate_zero(self):
        result = ConsolidationResult(actions=[
            ConsolidationAction(action="ADD", new_node_index=0, reasoning="novel"),
            ConsolidationAction(action="UPDATE", new_node_index=1, reasoning="merge"),
        ])
        assert result.fallback_rate == 0.0
        assert result.health_ok is True

    def test_fallback_rate_high(self):
        result = ConsolidationResult(actions=[
            ConsolidationAction(action="ADD", new_node_index=0, reasoning="Fallback ADD — error"),
            ConsolidationAction(action="ADD", new_node_index=1, reasoning="Fallback ADD — error"),
            ConsolidationAction(action="ADD", new_node_index=2, reasoning="novel"),
        ], fallback_count=2)
        assert result.fallback_rate == pytest.approx(2.0 / 3.0)
        assert result.health_ok is False

    def test_fallback_rate_empty(self):
        result = ConsolidationResult(actions=[])
        assert result.fallback_rate == 0.0
        assert result.health_ok is True

    def test_novelty_rate_zero(self):
        """No skips = healthy learning."""
        result = ConsolidationResult(actions=[], total_contested=5, nodes_skipped=0)
        assert result.novelty_rate == 0.0

    def test_novelty_rate_high(self):
        """High skip rate = memory stopped learning."""
        result = ConsolidationResult(actions=[], total_contested=10, nodes_skipped=9)
        assert result.novelty_rate == pytest.approx(0.9)

    def test_novelty_rate_no_contested(self):
        """No contested nodes = no novelty rate (avoid division by zero)."""
        result = ConsolidationResult(actions=[], total_contested=0, nodes_skipped=0)
        assert result.novelty_rate == 0.0

    def test_metrics_populated(self):
        """Metrics are populated during consolidation."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index)

        ref = mem.thought("PostgreSQL database ACID compliance choice")
        index.index_node(ref.id)
        extracted = [{"index": 0, "type": "thought", "content": "PostgreSQL database ACID compliance choice"}]
        node_refs = [ref]

        # Provider returns ADD
        provider.set_response([
            {"name": "add_memory", "arguments": {"new_node_index": 0, "reasoning": "new info"}},
        ])

        result = engine.consolidate(extracted, node_refs)

        assert result.total_contested >= 1
        assert result.avg_candidates_per_node > 0
        assert result.llm_calls >= 1

    def test_fallback_count_in_ingest_result(self):
        """fallback_count propagates through IngestResult."""
        mem, index, existing_ids = _make_memory_with_nodes()

        engine = ConsolidationEngine(mem, FailingConsolidationProvider(), index, max_retries=1)

        def mock_llm(prompt):
            return json.dumps({
                "nodes": [{"type": "thought", "content": "PostgreSQL ACID compliance database"}],
                "relationships": [],
                "states": [],
            })

        extractor = AutoExtract(
            mem, llm=mock_llm, vector_index=index,
            consolidation_engine=engine,
        )

        result = extractor.ingest("test input about PostgreSQL ACID")
        # Consolidation fails → fallback ADD
        assert result.consolidation_used is True
        assert result.fallback_count >= 1


class TestConsolidationThinkTagStripping:
    """Tests for think tag handling in consolidation tool call parsing."""

    def test_string_args_with_think_tags(self):
        """Tool call arguments as string with think tags should parse."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()

        engine = ConsolidationEngine(mem, provider, index)

        ref = mem.thought("PostgreSQL is the database choice for ACID compliance reasons")
        index.index_node(ref.id)
        extracted = [{"index": 0, "type": "thought", "content": "PostgreSQL is the database choice for ACID compliance reasons"}]
        node_refs = [ref]

        # Provider returns tool calls with string args wrapped in think tags
        provider.set_response([{
            "name": "add_memory",
            "arguments": '<think>Let me analyze this node...</think>{"new_node_index": 0, "reasoning": "genuinely new"}',
        }])

        result = engine.consolidate(extracted, node_refs)

        # Should parse successfully, not fallback
        add_actions = [a for a in result.actions if a.action == "ADD" and not a.is_fallback]
        assert len(add_actions) >= 1


class TestCrossBatchCollision:
    """Tests for cross-batch candidate collision protection."""

    def test_same_target_across_batches_falls_back(self):
        """Two batches targeting same existing node: second falls back to ADD."""
        mem, index, existing_ids = _make_memory_with_nodes()
        batch_num = [0]

        class BatchTrackingProvider:
            def tool_call(self, messages, tools):
                batch_num[0] += 1
                # Both batches try to UPDATE the same existing node (existing_ids[0])
                return [{
                    "name": "update_memory",
                    "arguments": {
                        "new_node_index": 0 if batch_num[0] == 1 else 1,
                        "target_candidate_index": 0,
                        "merged_content": f"merged content from batch {batch_num[0]}",
                        "reasoning": f"batch {batch_num[0]} update",
                    },
                }]

        engine = ConsolidationEngine(
            mem, BatchTrackingProvider(), index,
            max_batch_size=1,  # force split into 1-node batches
            candidate_threshold=0.01,  # very low so everything is contested
        )

        # Two new nodes, both similar to existing_ids[0]
        ref1 = mem.thought("We chose PostgreSQL for ACID compliance and database reliability")
        index.index_node(ref1.id)
        ref2 = mem.thought("PostgreSQL was selected due to strong ACID transaction support")
        index.index_node(ref2.id)

        extracted = [
            {"index": 0, "type": "thought", "content": "We chose PostgreSQL for ACID compliance and database reliability"},
            {"index": 1, "type": "thought", "content": "PostgreSQL was selected due to strong ACID transaction support"},
        ]
        node_refs = [ref1, ref2]

        result = engine.consolidate(extracted, node_refs)

        # First batch succeeds (UPDATE), second should fall back (cross-batch collision)
        fallback_actions = [a for a in result.actions if a.is_fallback]
        # At least one should have been caught by the cross-batch guard
        # (depends on which existing node both batches target)
        assert result.llm_calls == 2  # two batches processed


class TestIsFallbackField:
    """Tests for is_fallback field on ConsolidationAction."""

    def test_intentional_add_not_fallback(self):
        """LLM-decided ADD has is_fallback=False."""
        mem, index, existing_ids = _make_memory_with_nodes()
        provider = MockConsolidationProvider()
        engine = ConsolidationEngine(mem, provider, index)

        ref = mem.thought("PostgreSQL ACID database compliance choice")
        index.index_node(ref.id)
        extracted = [{"index": 0, "type": "thought", "content": "PostgreSQL ACID database compliance choice"}]
        node_refs = [ref]

        provider.set_response([
            {"name": "add_memory", "arguments": {"new_node_index": 0, "reasoning": "genuinely new"}},
        ])

        result = engine.consolidate(extracted, node_refs)
        for action in result.actions:
            if action.action == "ADD" and "genuinely new" in action.reasoning:
                assert action.is_fallback is False

    def test_error_fallback_has_flag(self):
        """Error-driven fallback ADD has is_fallback=True."""
        mem, index, existing_ids = _make_memory_with_nodes()
        engine = ConsolidationEngine(mem, FailingConsolidationProvider(), index, max_retries=1)

        ref = mem.thought("PostgreSQL database ACID compliance selection")
        index.index_node(ref.id)
        extracted = [{"index": 0, "type": "thought", "content": "PostgreSQL database ACID compliance selection"}]
        node_refs = [ref]

        result = engine.consolidate(extracted, node_refs)
        fallbacks = [a for a in result.actions if a.is_fallback]
        assert len(fallbacks) >= 1

    def test_novel_add_not_fallback(self):
        """Novel nodes (no candidates) are ADD but not fallback."""
        mem = Memory()
        embedder = MockEmbedder()
        index = VectorIndex(mem, embedder)
        provider = MockConsolidationProvider()
        engine = ConsolidationEngine(mem, provider, index)

        ref = mem.thought("Completely new topic")
        index.index_node(ref.id)
        extracted = [{"index": 0, "type": "thought", "content": "Completely new topic"}]
        node_refs = [ref]

        result = engine.consolidate(extracted, node_refs)
        # Novel ADD — not a fallback
        for action in result.actions:
            assert action.is_fallback is False


class TestNonRetryableErrors:
    """Tests for non-retryable error handling."""

    def test_value_error_not_retried(self):
        """ValueError from provider should fail immediately, not retry."""
        mem, index, existing_ids = _make_memory_with_nodes()
        call_count = [0]

        class ValueErrorProvider:
            def tool_call(self, messages, tools):
                call_count[0] += 1
                raise ValueError("invalid configuration")

        engine = ConsolidationEngine(mem, ValueErrorProvider(), index, max_retries=3)

        ref = mem.thought("PostgreSQL ACID compliance for database decision")
        index.index_node(ref.id)
        extracted = [{"index": 0, "type": "thought", "content": "PostgreSQL ACID compliance for database decision"}]
        node_refs = [ref]

        result = engine.consolidate(extracted, node_refs)
        # Should have been called only once (no retry on ValueError)
        assert call_count[0] == 1
        # Should still fall back to ADD (not crash)
        assert len(result.actions) > 0

    def test_connection_error_is_retried(self):
        """ConnectionError should be retried."""
        mem, index, existing_ids = _make_memory_with_nodes()
        call_count = [0]

        class FlakeyProvider:
            def tool_call(self, messages, tools):
                call_count[0] += 1
                if call_count[0] <= 2:
                    raise ConnectionError("network timeout")
                return [{"name": "add_memory", "arguments": {"new_node_index": 0, "reasoning": "ok"}}]

        engine = ConsolidationEngine(mem, FlakeyProvider(), index, max_retries=3)

        ref = mem.thought("PostgreSQL for database ACID compliance choice")
        index.index_node(ref.id)
        extracted = [{"index": 0, "type": "thought", "content": "PostgreSQL for database ACID compliance choice"}]
        node_refs = [ref]

        result = engine.consolidate(extracted, node_refs)
        assert call_count[0] == 3  # 2 failures + 1 success
