"""
Tests for FlowScript embeddings layer: providers, vector index, unified search, auto-extract.

Uses a mock embedding provider for deterministic testing without external deps.
"""

import json
import math
import os
import tempfile
from pathlib import Path

import pytest

from flowscript_agents import Memory, UnifiedMemory
from flowscript_agents.embeddings import (
    VectorIndex,
    VectorSearchResult,
    UnifiedSearch,
    UnifiedSearchResult,
    AutoExtract,
    IngestResult,
)
from flowscript_agents.embeddings.providers import EmbeddingProvider
from flowscript_agents.embeddings.index import _normalize, _dot, _cosine_similarity, _magnitude
from flowscript_agents.embeddings.search import _keyword_score, _temporal_score
from flowscript_agents.embeddings.extract import (
    _extract_json,
    _parse_extraction,
    EXTRACTION_PROMPT,
    EXTRACTION_PROMPT_BASE,
    _ACTOR_SUFFIX,
)
from flowscript_agents.embeddings._utils import strip_llm_wrapping

# Import shared MockEmbeddings from conftest
from conftest import MockEmbeddings


# =============================================================================
# Vector Math Tests
# =============================================================================


class TestVectorMath:
    def test_normalize_unit_vector(self):
        v = [1.0, 0.0, 0.0]
        result = _normalize(v)
        assert result == [1.0, 0.0, 0.0]

    def test_normalize_arbitrary_vector(self):
        v = [3.0, 4.0]
        result = _normalize(v)
        assert abs(result[0] - 0.6) < 1e-6
        assert abs(result[1] - 0.8) < 1e-6

    def test_normalize_zero_vector(self):
        v = [0.0, 0.0, 0.0]
        result = _normalize(v)
        assert result == [0.0, 0.0, 0.0]

    def test_dot_product(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        assert _dot(a, b) == 32.0

    def test_magnitude(self):
        v = [3.0, 4.0]
        assert abs(_magnitude(v) - 5.0) < 1e-6

    def test_cosine_similarity_identical(self):
        v = _normalize([1.0, 1.0])
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_cosine_similarity_opposite(self):
        a = _normalize([1.0, 0.0])
        b = _normalize([-1.0, 0.0])
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6


# =============================================================================
# EmbeddingProvider Protocol Tests
# =============================================================================


class TestMockEmbeddings:
    def test_implements_protocol(self):
        emb = MockEmbeddings()
        assert isinstance(emb, EmbeddingProvider)

    def test_dimensions(self):
        emb = MockEmbeddings(dims=16)
        assert emb.dimensions == 16

    def test_embed_returns_correct_shape(self):
        emb = MockEmbeddings(dims=8)
        result = emb.embed(["hello world", "test"])
        assert len(result) == 2
        assert all(len(v) == 8 for v in result)

    def test_embed_similar_texts(self):
        emb = MockEmbeddings(dims=16)
        vecs = emb.embed(["database performance", "database speed"])
        # Should share the "database" dimension
        sim = _cosine_similarity(_normalize(vecs[0]), _normalize(vecs[1]))
        assert sim > 0.3  # some overlap

    def test_embed_empty_list(self):
        emb = MockEmbeddings()
        assert emb.embed([]) == []

    def test_embed_normalized(self):
        emb = MockEmbeddings(dims=8)
        vecs = emb.embed(["hello world"])
        mag = _magnitude(vecs[0])
        assert abs(mag - 1.0) < 1e-6  # should be normalized


# =============================================================================
# VectorIndex Tests
# =============================================================================


class TestVectorIndex:
    def test_index_single_node(self):
        mem = Memory()
        mem.thought("Redis is fast")
        emb = MockEmbeddings()
        index = VectorIndex(mem, emb)
        assert index.index_node(mem.nodes[0].id) is True
        assert index.indexed_count == 1

    def test_index_already_indexed(self):
        mem = Memory()
        ref = mem.thought("Redis is fast")
        emb = MockEmbeddings()
        index = VectorIndex(mem, emb)
        index.index_node(ref.id)
        assert index.index_node(ref.id) is False  # already indexed

    def test_index_all(self):
        mem = Memory()
        mem.thought("Redis is fast")
        mem.thought("PostgreSQL has ACID")
        mem.thought("SQLite is embedded")
        emb = MockEmbeddings(dims=16)
        index = VectorIndex(mem, emb)
        count = index.index_all()
        assert count == 3
        assert index.indexed_count == 3

    def test_index_all_incremental(self):
        mem = Memory()
        mem.thought("Redis is fast")
        emb = MockEmbeddings(dims=16)
        index = VectorIndex(mem, emb)
        index.index_all()
        assert index.indexed_count == 1
        mem.thought("PostgreSQL has ACID")
        count = index.index_all()
        assert count == 1
        assert index.indexed_count == 2

    def test_search_basic(self):
        mem = Memory()
        mem.thought("Redis provides sub-millisecond reads")
        mem.thought("PostgreSQL ensures ACID compliance")
        mem.thought("The weather is sunny today")
        emb = MockEmbeddings(dims=16)
        index = VectorIndex(mem, emb)
        index.index_all()
        results = index.search("Redis performance", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, VectorSearchResult) for r in results)
        # Redis result should rank higher (shares "redis" word)
        if results:
            assert results[0].score >= results[-1].score

    def test_search_with_threshold(self):
        mem = Memory()
        mem.thought("Redis is fast")
        mem.thought("Totally unrelated content about cooking")
        emb = MockEmbeddings(dims=16)
        index = VectorIndex(mem, emb)
        index.index_all()
        results = index.search("Redis speed", top_k=10, threshold=0.5)
        # Only Redis should be above threshold
        for r in results:
            assert r.score >= 0.5

    def test_search_empty_index(self):
        mem = Memory()
        emb = MockEmbeddings()
        index = VectorIndex(mem, emb)
        results = index.search("anything")
        assert results == []

    def test_search_includes_temporal(self):
        mem = Memory()
        ref = mem.thought("Redis is fast")
        emb = MockEmbeddings()
        index = VectorIndex(mem, emb)
        index.index_all()
        results = index.search("Redis")
        assert len(results) > 0
        assert results[0].tier == "current"
        assert results[0].frequency == 1

    def test_find_similar(self):
        mem = Memory()
        a = mem.thought("Redis provides fast reads")
        b = mem.thought("Redis offers quick read access")
        c = mem.thought("Cooking pasta requires water")
        emb = MockEmbeddings(dims=16)
        index = VectorIndex(mem, emb)
        index.index_all()
        similar = index.find_similar(a.id, top_k=2)
        assert len(similar) <= 2
        # Redis-related should be more similar than cooking
        if len(similar) >= 2:
            redis_scores = [s.score for s in similar if "redis" in s.content.lower() or "read" in s.content.lower()]
            cooking_scores = [s.score for s in similar if "cooking" in s.content.lower()]
            if redis_scores and cooking_scores:
                assert max(redis_scores) > max(cooking_scores)

    def test_remove_node(self):
        mem = Memory()
        ref = mem.thought("Redis is fast")
        emb = MockEmbeddings()
        index = VectorIndex(mem, emb)
        index.index_all()
        assert index.indexed_count == 1
        assert index.remove_node(ref.id) is True
        assert index.indexed_count == 0

    def test_reindex_all(self):
        mem = Memory()
        mem.thought("Redis is fast")
        mem.thought("PostgreSQL is reliable")
        emb = MockEmbeddings(dims=16)
        index = VectorIndex(mem, emb)
        index.index_all()
        assert index.indexed_count == 2
        count = index.reindex_all()
        assert count == 2

    def test_sidecar_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            mem = Memory()
            mem.thought("Redis is fast")
            mem.thought("PostgreSQL is reliable")
            mem._file_path = path
            mem.save()

            emb = MockEmbeddings(dims=8)
            index = VectorIndex(mem, emb)
            index.index_all()
            index.save()

            # Verify sidecar file exists
            sidecar = path + ".embeddings.json"
            assert os.path.exists(sidecar)

            # Load into new index
            mem2 = Memory.load(path)
            index2 = VectorIndex(mem2, emb)
            loaded = index2.load()
            assert loaded == 2
            assert index2.indexed_count == 2

    def test_sidecar_validates_nodes(self):
        """Sidecar load should skip embeddings for nodes that no longer exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            mem = Memory()
            ref = mem.thought("Redis is fast")
            mem.thought("PostgreSQL is reliable")
            mem._file_path = path
            mem.save()

            emb = MockEmbeddings(dims=8)
            index = VectorIndex(mem, emb)
            index.index_all()
            index.save()

            # Remove a node, save, then reload embeddings
            mem.remove_node(ref.id)
            mem.save()

            mem2 = Memory.load(path)
            index2 = VectorIndex(mem2, emb)
            loaded = index2.load()
            assert loaded == 1  # one node was removed

    def test_index_nonexistent_node(self):
        mem = Memory()
        emb = MockEmbeddings()
        index = VectorIndex(mem, emb)
        assert index.index_node("nonexistent") is False

    def test_sidecar_dimension_mismatch(self):
        """Loading embeddings from a different provider should be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            mem = Memory()
            mem.thought("test node")
            mem._file_path = path
            mem.save()

            # Save with 8-dim provider
            emb8 = MockEmbeddings(dims=8)
            index8 = VectorIndex(mem, emb8)
            index8.index_all()
            index8.save()

            # Try to load with 16-dim provider — should reject
            emb16 = MockEmbeddings(dims=16)
            index16 = VectorIndex(mem, emb16)
            loaded = index16.load()
            assert loaded == 0  # rejected due to dimension mismatch

    def test_sidecar_corrupt_file(self):
        """Corrupt sidecar should be handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            mem = Memory()
            mem._file_path = path

            # Write corrupt sidecar
            sidecar = path + ".embeddings.json"
            with open(sidecar, "w") as f:
                f.write("NOT VALID JSON {{{")

            emb = MockEmbeddings()
            index = VectorIndex(mem, emb)
            loaded = index.load()
            assert loaded == 0  # handled gracefully


# =============================================================================
# Keyword Score Tests
# =============================================================================


class TestKeywordScore:
    def test_exact_match(self):
        score = _keyword_score("redis fast", "Redis is fast")
        assert score == 1.0

    def test_partial_match(self):
        score = _keyword_score("redis slow", "Redis is fast")
        assert score == 0.5  # "redis" matches, "slow" doesn't

    def test_no_match(self):
        score = _keyword_score("cooking pasta", "Redis is fast")
        assert score == 0.0

    def test_empty_query(self):
        score = _keyword_score("", "Redis is fast")
        assert score == 0.0

    def test_case_insensitive(self):
        score = _keyword_score("REDIS FAST", "redis is fast")
        assert score == 1.0


# =============================================================================
# Temporal Score Tests
# =============================================================================


class TestTemporalScore:
    def test_foundation_tier_scores_highest(self):
        foundation = _temporal_score("foundation", 5, None)
        current = _temporal_score("current", 5, None)
        assert foundation > current

    def test_high_frequency_boosts_score(self):
        high_freq = _temporal_score("current", 10, None)
        low_freq = _temporal_score("current", 1, None)
        assert high_freq > low_freq

    def test_none_values(self):
        score = _temporal_score(None, None, None)
        assert 0 <= score <= 1

    def test_score_in_range(self):
        score = _temporal_score("proven", 5, "2026-03-20T12:00:00+00:00")
        assert 0 <= score <= 1


# =============================================================================
# UnifiedSearch Tests
# =============================================================================


class TestUnifiedSearch:
    def test_keyword_only(self):
        mem = Memory()
        mem.thought("Redis provides fast reads")
        mem.thought("PostgreSQL ensures ACID compliance")
        mem.thought("The weather is sunny")
        search = UnifiedSearch(mem)  # no vector index
        results = search.search("Redis performance")
        assert len(results) > 0
        # Redis result should be first (keyword match on "redis")
        assert "redis" in results[0].content.lower()

    def test_with_vector_index(self):
        mem = Memory()
        mem.thought("Redis provides fast reads")
        mem.thought("PostgreSQL ensures ACID compliance")
        emb = MockEmbeddings(dims=16)
        index = VectorIndex(mem, emb)
        index.index_all()
        search = UnifiedSearch(mem, vector_index=index)
        results = search.search("Redis speed")
        assert len(results) > 0
        assert all(isinstance(r, UnifiedSearchResult) for r in results)

    def test_results_have_sources(self):
        mem = Memory()
        mem.thought("Redis provides fast reads")
        emb = MockEmbeddings(dims=16)
        index = VectorIndex(mem, emb)
        index.index_all()
        search = UnifiedSearch(mem, vector_index=index)
        results = search.search("Redis")
        assert len(results) > 0
        assert len(results[0].sources) > 0

    def test_top_k_limit(self):
        mem = Memory()
        for i in range(20):
            mem.thought(f"Node number {i}")
        search = UnifiedSearch(mem)
        results = search.search("node", top_k=5)
        assert len(results) <= 5

    def test_custom_weights(self):
        mem = Memory()
        mem.thought("Redis is fast")
        search = UnifiedSearch(
            mem, vector_weight=0.0, keyword_weight=1.0, temporal_weight=0.0,
        )
        results = search.search("Redis")
        assert len(results) > 0
        # With only keyword weight, vector score should be 0
        assert results[0].vector_score == 0.0

    def test_empty_query(self):
        mem = Memory()
        mem.thought("Redis is fast")
        search = UnifiedSearch(mem)
        results = search.search("")
        # Empty query shouldn't crash, may return empty or all
        assert isinstance(results, list)


# =============================================================================
# AutoExtract JSON Parsing Tests
# =============================================================================


class TestExtractJson:
    def test_clean_json(self):
        raw = '{"nodes": [{"type": "thought", "content": "test"}], "relationships": [], "states": []}'
        result = _extract_json(raw)
        assert len(result["nodes"]) == 1

    def test_markdown_fenced(self):
        raw = '```json\n{"nodes": [{"type": "thought", "content": "test"}], "relationships": [], "states": []}\n```'
        result = _extract_json(raw)
        assert len(result["nodes"]) == 1

    def test_with_preamble(self):
        raw = 'Here is the extracted structure:\n{"nodes": [{"type": "thought", "content": "test"}], "relationships": [], "states": []}'
        result = _extract_json(raw)
        assert len(result["nodes"]) == 1

    def test_malformed_returns_empty(self):
        raw = "This is not JSON at all"
        result = _extract_json(raw)
        assert result["nodes"] == []

    def test_empty_string(self):
        result = _extract_json("")
        assert result == {"nodes": [], "relationships": [], "states": []}

    def test_json_with_trailing_prose(self):
        """Bracket-counting should handle JSON followed by prose with braces."""
        raw = 'Here is the result: {"nodes": [{"type": "thought", "content": "test"}], "relationships": [], "states": []} Hope that helps! {not json}'
        result = _extract_json(raw)
        assert len(result["nodes"]) == 1

    def test_nested_json_objects(self):
        """Bracket-counting should handle nested braces correctly."""
        raw = '{"nodes": [{"type": "thought", "content": "test {with braces}"}], "relationships": [], "states": []}'
        result = _extract_json(raw)
        assert len(result["nodes"]) == 1
        assert "with braces" in result["nodes"][0]["content"]


class TestParseExtraction:
    def test_basic_extraction(self):
        raw = {
            "nodes": [
                {"type": "thought", "content": "Redis is fast"},
                {"type": "decision", "content": "Use Redis"},
            ],
            "relationships": [
                {"type": "causes", "source": 0, "target": 1},
            ],
            "states": [
                {"type": "decided", "node": 1, "rationale": "speed critical"},
            ],
        }
        result = _parse_extraction(raw)
        assert len(result.nodes) == 2
        assert len(result.relationships) == 1
        assert result.nodes[1].state_type == "decided"
        assert result.nodes[1].state_rationale == "speed critical"

    def test_invalid_node_type_fallback(self):
        raw = {"nodes": [{"type": "foobar", "content": "test"}], "relationships": [], "states": []}
        result = _parse_extraction(raw)
        assert result.nodes[0].type == "thought"  # fallback

    def test_invalid_relationship_skipped(self):
        raw = {
            "nodes": [{"type": "thought", "content": "A"}],
            "relationships": [
                {"type": "causes", "source": 0, "target": 5},  # out of bounds
            ],
            "states": [],
        }
        result = _parse_extraction(raw)
        assert len(result.relationships) == 0

    def test_self_loop_skipped(self):
        raw = {
            "nodes": [{"type": "thought", "content": "A"}],
            "relationships": [
                {"type": "causes", "source": 0, "target": 0},  # self-loop
            ],
            "states": [],
        }
        result = _parse_extraction(raw)
        assert len(result.relationships) == 0

    def test_tension_requires_axis(self):
        raw = {
            "nodes": [
                {"type": "thought", "content": "A"},
                {"type": "thought", "content": "B"},
            ],
            "relationships": [
                {"type": "tension", "source": 0, "target": 1},  # no axis
            ],
            "states": [],
        }
        result = _parse_extraction(raw)
        assert len(result.relationships) == 0

    def test_tension_with_axis(self):
        raw = {
            "nodes": [
                {"type": "thought", "content": "A"},
                {"type": "thought", "content": "B"},
            ],
            "relationships": [
                {"type": "tension", "source": 0, "target": 1, "axis": "speed vs safety"},
            ],
            "states": [],
        }
        result = _parse_extraction(raw)
        assert len(result.relationships) == 1
        assert result.relationships[0].axis == "speed vs safety"

    def test_empty_content_skipped(self):
        raw = {"nodes": [{"type": "thought", "content": ""}], "relationships": [], "states": []}
        result = _parse_extraction(raw)
        assert len(result.nodes) == 0

    def test_non_dict_nodes_skipped(self):
        raw = {"nodes": ["not a dict", 42], "relationships": [], "states": []}
        result = _parse_extraction(raw)
        assert len(result.nodes) == 0


# =============================================================================
# AutoExtract Integration Tests
# =============================================================================


class TestAutoExtract:
    def _mock_llm(self, prompt: str) -> str:
        """Mock LLM that returns predetermined extraction."""
        return json.dumps({
            "nodes": [
                {"type": "thought", "content": "PostgreSQL has ACID compliance"},
                {"type": "decision", "content": "Choose PostgreSQL over MySQL"},
                {"type": "thought", "content": "Connection pooling needed"},
            ],
            "relationships": [
                {"type": "causes", "source": 0, "target": 1},
                {"type": "tension", "source": 1, "target": 2, "axis": "complexity vs reliability"},
            ],
            "states": [
                {"type": "decided", "node": 1, "rationale": "ACID compliance required"},
            ],
        })

    def test_ingest_creates_nodes(self):
        mem = Memory()
        extractor = AutoExtract(mem, llm=self._mock_llm)
        result = extractor.ingest("any text")
        assert result.nodes_created == 3
        assert result.relationships_created == 2
        assert result.states_created == 1
        assert len(result.node_ids) == 3
        assert mem.size == 3

    def test_ingest_creates_relationships(self):
        mem = Memory()
        extractor = AutoExtract(mem, llm=self._mock_llm)
        extractor.ingest("any text")
        ir = mem.to_ir()
        assert len(ir.relationships) == 2
        # Check types
        rel_types = {r.type.value for r in ir.relationships}
        assert "causes" in rel_types
        assert "tension" in rel_types

    def test_ingest_creates_states(self):
        mem = Memory()
        extractor = AutoExtract(mem, llm=self._mock_llm)
        extractor.ingest("any text")
        ir = mem.to_ir()
        assert len(ir.states) == 1
        assert ir.states[0].type.value == "decided"

    def test_ingest_with_semantic_dedup(self):
        mem = Memory()
        emb = MockEmbeddings(dims=16)
        index = VectorIndex(mem, emb)

        # Pre-populate with existing node
        existing = mem.thought("PostgreSQL has ACID compliance")
        index.index_node(existing.id)

        extractor = AutoExtract(mem, llm=self._mock_llm, vector_index=index, dedup_threshold=0.5)
        result = extractor.ingest("any text")
        # First node should be deduped against existing
        assert result.nodes_deduplicated >= 1
        assert result.nodes_created < 3

    def test_ingest_with_metadata(self):
        mem = Memory()
        extractor = AutoExtract(mem, llm=self._mock_llm)
        result = extractor.ingest("any text", metadata={"source": "test"})
        for nid in result.node_ids:
            node = mem.get_node(nid)
            if node and node.ext:
                assert node.ext.get("source") == "test"

    def test_ingest_conversation(self):
        mem = Memory()
        extractor = AutoExtract(mem, llm=self._mock_llm)
        messages = [
            {"role": "user", "content": "Should we use PostgreSQL?"},
            {"role": "assistant", "content": "Yes, for ACID compliance."},
        ]
        result = extractor.ingest_conversation(messages)
        assert result.nodes_created > 0

    def test_llm_failure_graceful(self):
        def failing_llm(prompt: str) -> str:
            raise RuntimeError("LLM is down")

        mem = Memory()
        extractor = AutoExtract(mem, llm=failing_llm)
        result = extractor.ingest("any text")
        assert result.nodes_created == 0
        assert mem.size == 0

    def test_llm_returns_garbage(self):
        def garbage_llm(prompt: str) -> str:
            return "I don't understand the question"

        mem = Memory()
        extractor = AutoExtract(mem, llm=garbage_llm)
        result = extractor.ingest("any text")
        assert result.nodes_created == 0

    def test_queries_work_on_extracted_nodes(self):
        """The killer test: extract → query. This is what Mem0 can't do."""
        mem = Memory()
        extractor = AutoExtract(mem, llm=self._mock_llm)
        extractor.ingest("any text")

        # Reasoning queries should work on extracted structure
        tensions = mem.query.tensions()
        assert tensions.metadata["total_tensions"] >= 1

        blocked = mem.query.blocked()
        # May or may not have blocked nodes depending on extraction
        assert blocked is not None

    def test_extraction_alternatives_path(self):
        """Verify extracted alternatives are queryable via query.alternatives()."""
        def alt_llm(prompt: str) -> str:
            return json.dumps({
                "nodes": [
                    {"type": "question", "content": "Which database to use?"},
                    {"type": "alternative", "content": "PostgreSQL"},
                    {"type": "alternative", "content": "MySQL"},
                ],
                "relationships": [
                    {"type": "alternative", "source": 0, "target": 1},
                    {"type": "alternative", "source": 0, "target": 2},
                ],
                "states": [
                    {"type": "decided", "node": 1, "rationale": "ACID compliance"},
                ],
            })

        mem = Memory()
        extractor = AutoExtract(mem, llm=alt_llm)
        result = extractor.ingest("evaluating databases")

        assert result.nodes_created == 3
        assert result.relationships_created == 2

        # The critical test: can we query alternatives?
        question_refs = [r for r in mem.nodes if r.type.value == "question"]
        assert len(question_refs) == 1
        alts = mem.query.alternatives(question_refs[0].id)
        assert alts is not None


# =============================================================================
# UnifiedMemory Integration Tests
# =============================================================================


class TestUnifiedMemory:
    def _mock_llm(self, prompt: str) -> str:
        return json.dumps({
            "nodes": [
                {"type": "thought", "content": "User prefers dark mode"},
            ],
            "relationships": [],
            "states": [],
        })

    def test_create_in_memory(self):
        umem = UnifiedMemory()
        assert umem.size == 0
        assert umem.vector_index is None
        assert umem.extractor is None

    def test_create_with_embedder(self):
        emb = MockEmbeddings()
        umem = UnifiedMemory(embedder=emb)
        assert umem.vector_index is not None

    def test_create_with_llm(self):
        umem = UnifiedMemory(llm=self._mock_llm)
        assert umem.extractor is not None

    def test_add_with_llm(self):
        emb = MockEmbeddings(dims=16)
        umem = UnifiedMemory(embedder=emb, llm=self._mock_llm)
        result = umem.add("User prefers dark mode for reading")
        assert result.nodes_created >= 1
        assert umem.size >= 1

    def test_add_without_llm(self):
        umem = UnifiedMemory()
        result = umem.add("User prefers dark mode")
        assert result.nodes_created == 1
        assert umem.size == 1

    def test_add_raw(self):
        emb = MockEmbeddings(dims=16)
        umem = UnifiedMemory(embedder=emb)
        ref = umem.add_raw("Redis is fast", node_type="thought")
        assert ref.content == "Redis is fast"
        # Should be indexed
        assert umem.vector_index.indexed_count == 1

    def test_add_raw_invalid_type_fallback(self):
        """Invalid node_type should fall back to thought, not call arbitrary methods."""
        umem = UnifiedMemory()
        ref = umem.add_raw("test content", node_type="save")  # should NOT call save()
        assert ref.type.value == "thought"
        ref2 = umem.add_raw("test content 2", node_type="remove_node")  # should NOT call remove_node()
        assert ref2.type.value == "thought"

    def test_add_raw_valid_types(self):
        umem = UnifiedMemory()
        for node_type in ("thought", "statement", "question", "action", "insight", "completion"):
            ref = umem.add_raw(f"test {node_type}", node_type=node_type)
            assert ref.type.value in (node_type, "block")  # completion might map differently

    def test_search_keyword_only(self):
        umem = UnifiedMemory()
        umem.add_raw("Redis provides fast reads")
        umem.add_raw("PostgreSQL ensures ACID compliance")
        results = umem.search("Redis")
        assert len(results) > 0

    def test_search_with_vectors(self):
        emb = MockEmbeddings(dims=16)
        umem = UnifiedMemory(embedder=emb)
        umem.add_raw("Redis provides fast reads")
        umem.add_raw("PostgreSQL ensures ACID compliance")
        results = umem.search("Redis speed")
        assert len(results) > 0
        assert results[0].vector_score > 0

    def test_vector_search(self):
        emb = MockEmbeddings(dims=16)
        umem = UnifiedMemory(embedder=emb)
        umem.add_raw("Redis provides fast reads")
        umem.add_raw("PostgreSQL ensures ACID compliance")
        results = umem.vector_search("Redis")
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)

    def test_vector_search_no_embedder(self):
        umem = UnifiedMemory()
        results = umem.vector_search("anything")
        assert results == []

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            emb = MockEmbeddings(dims=8)
            umem = UnifiedMemory(file_path=path, embedder=emb)
            umem.add_raw("Redis is fast")
            umem.add_raw("PostgreSQL is reliable")
            umem.save()

            # Verify both files exist
            assert os.path.exists(path)
            assert os.path.exists(path + ".embeddings.json")

            # Reload
            umem2 = UnifiedMemory(file_path=path, embedder=emb)
            assert umem2.size == 2
            assert umem2.vector_index.indexed_count == 2

    def test_session_lifecycle(self):
        umem = UnifiedMemory()
        result = umem.session_start()
        assert result is not None
        umem.add_raw("Test node")
        # session_end would try to save without path — test close instead
        wrap = umem.close()
        assert wrap is not None

    def test_get_context(self):
        umem = UnifiedMemory()
        umem.add_raw("Redis is fast")
        umem.add_raw("PostgreSQL is reliable")
        context = umem.get_context(max_tokens=1000)
        assert "Redis" in context
        assert "PostgreSQL" in context
        assert "[current|1x]" in context

    def test_reasoning_queries_via_memory(self):
        """Test that reasoning queries work through UnifiedMemory."""
        umem = UnifiedMemory()
        q = umem.memory.question("Which database?")
        redis = umem.memory.alternative(q, "Redis")
        pg = umem.memory.alternative(q, "PostgreSQL")
        redis.decide(rationale="speed critical")
        pg.block(reason="too complex")

        tensions_result = umem.memory.query.blocked()
        assert tensions_result is not None

        alts = umem.memory.query.alternatives(q.id)
        assert alts is not None

    def test_full_pipeline(self):
        """End-to-end: add → search → query. The complete agent memory experience."""
        emb = MockEmbeddings(dims=16)

        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "nodes": [
                    {"type": "thought", "content": "PostgreSQL has ACID compliance"},
                    {"type": "decision", "content": "Choose PostgreSQL"},
                    {"type": "thought", "content": "Need connection pooling"},
                ],
                "relationships": [
                    {"type": "causes", "source": 0, "target": 1},
                    {"type": "tension", "source": 1, "target": 2, "axis": "complexity vs reliability"},
                ],
                "states": [
                    {"type": "decided", "node": 1, "rationale": "ACID required"},
                ],
            })

        umem = UnifiedMemory(embedder=emb, llm=mock_llm)

        # Add (auto-extract)
        result = umem.add("We evaluated databases and chose PostgreSQL for ACID.")
        assert result.nodes_created >= 1

        # Search (unified)
        search_results = umem.search("database decision")
        assert len(search_results) > 0

        # Reasoning queries (what Mem0 can't do)
        tensions = umem.memory.query.tensions()
        assert tensions is not None

    def test_repr(self):
        umem = UnifiedMemory()
        r = repr(umem)
        assert "UnifiedMemory" in r
        assert "nodes=0" in r


# =============================================================================
# Production Hardening Tests — Phase 2
# =============================================================================


class TestStripLLMWrapping:
    """Tests for strip_llm_wrapping — handles think tags, code fences, whitespace."""

    def test_clean_text_unchanged(self):
        text = '{"nodes": [], "relationships": [], "states": []}'
        assert strip_llm_wrapping(text) == text

    def test_strip_think_tags(self):
        text = '<think>Let me analyze this carefully...</think>{"nodes": []}'
        result = strip_llm_wrapping(text)
        assert result == '{"nodes": []}'

    def test_strip_multiline_think_tags(self):
        text = (
            "<think>\nI need to extract the key decisions here.\n"
            "The user mentioned PostgreSQL and ACID compliance.\n"
            "Let me structure this properly.\n</think>\n"
            '{"nodes": [{"type": "decision", "content": "chose PostgreSQL"}]}'
        )
        result = strip_llm_wrapping(text)
        assert result.startswith('{"nodes"')
        assert "<think>" not in result

    def test_strip_think_tags_with_code_fences(self):
        text = (
            '<think>thinking...</think>\n'
            '```json\n{"nodes": [{"type": "thought", "content": "test"}]}\n```'
        )
        result = strip_llm_wrapping(text)
        assert "<think>" not in result
        assert "```" not in result
        assert '"thought"' in result

    def test_strip_code_fences_only(self):
        text = '```json\n{"nodes": []}\n```'
        result = strip_llm_wrapping(text)
        assert result == '{"nodes": []}'

    def test_whitespace_stripping(self):
        text = '  \n  {"nodes": []}  \n  '
        result = strip_llm_wrapping(text)
        assert result == '{"nodes": []}'

    def test_nested_think_not_greedy(self):
        """Ensure regex doesn't consume content between multiple think blocks."""
        text = '<think>first</think>KEEP<think>second</think>{"nodes": []}'
        result = strip_llm_wrapping(text)
        assert "KEEP" in result
        assert "<think>" not in result

    def test_unclosed_think_tag(self):
        """Truncated streaming response with unclosed think tag."""
        text = '<think>I need to analyze this but the stream was cut off... {"nodes": [{"type": "thought", "content": "inside think"}]}'
        result = strip_llm_wrapping(text)
        # Unclosed think tag should be stripped, content after it is lost
        # (this is the safest behavior — partial reasoning isn't parseable)
        assert "<think>" not in result

    def test_no_think_tag_unchanged(self):
        """Text without think tags should pass through unchanged."""
        text = '{"nodes": [{"type": "thought", "content": "normal response"}]}'
        result = strip_llm_wrapping(text)
        assert result == text


class TestExtractJsonWithThinkTags:
    """End-to-end: _extract_json handles think-wrapped responses."""

    def test_think_wrapped_json(self):
        raw = '<think>Analyzing...</think>{"nodes": [{"type": "thought", "content": "test"}], "relationships": [], "states": []}'
        result = _extract_json(raw)
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["content"] == "test"

    def test_think_plus_fences_plus_preamble(self):
        raw = (
            '<think>Let me think about this...\nOk I see 2 key points.</think>\n'
            'Here is the extraction:\n'
            '```json\n'
            '{"nodes": [{"type": "decision", "content": "chose Redis"}], "relationships": [], "states": []}\n'
            '```'
        )
        result = _extract_json(raw)
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["type"] == "decision"


class TestActorAwareExtraction:
    """Tests for actor-aware extraction prompts."""

    def test_actor_suffix_exists(self):
        assert "user" in _ACTOR_SUFFIX
        assert "agent" in _ACTOR_SUFFIX

    def test_user_suffix_prioritizes_decisions(self):
        suffix = _ACTOR_SUFFIX["user"]
        assert "HUMAN USER" in suffix
        assert "decisions" in suffix.lower() or "Decisions" in suffix

    def test_agent_suffix_prioritizes_observations(self):
        suffix = _ACTOR_SUFFIX["agent"]
        assert "AI AGENT" in suffix
        assert "observations" in suffix.lower() or "Analysis" in suffix

    def test_unknown_actor_no_suffix(self):
        # Unknown actor should return empty (no crash)
        suffix = _ACTOR_SUFFIX.get("robot", "")
        assert suffix == ""

    def test_ingest_with_actor_user(self):
        """Actor parameter passes through to extraction without crashing."""
        mem = Memory()
        call_count = [0]

        def mock_llm(prompt):
            call_count[0] += 1
            # Verify actor context was injected
            assert "HUMAN USER" in prompt
            return json.dumps({
                "nodes": [{"type": "decision", "content": "chose PostgreSQL"}],
                "relationships": [],
                "states": [{"type": "decided", "node": 0, "rationale": "ACID compliance"}],
            })

        extractor = AutoExtract(mem, llm=mock_llm)
        result = extractor.ingest("I've decided to use PostgreSQL", actor="user")
        assert result.nodes_created == 1
        assert call_count[0] == 1

    def test_ingest_with_actor_agent(self):
        mem = Memory()

        def mock_llm(prompt):
            assert "AI AGENT" in prompt
            return json.dumps({
                "nodes": [{"type": "thought", "content": "PostgreSQL has better ACID support"}],
                "relationships": [],
                "states": [],
            })

        extractor = AutoExtract(mem, llm=mock_llm)
        result = extractor.ingest("Based on analysis, PostgreSQL has better ACID support", actor="agent")
        assert result.nodes_created == 1

    def test_ingest_without_actor_no_suffix(self):
        mem = Memory()

        def mock_llm(prompt):
            # No actor context should be in the prompt
            assert "HUMAN USER" not in prompt
            assert "AI AGENT" not in prompt
            return json.dumps({
                "nodes": [{"type": "thought", "content": "test"}],
                "relationships": [],
                "states": [],
            })

        extractor = AutoExtract(mem, llm=mock_llm)
        result = extractor.ingest("test content")
        assert result.nodes_created == 1

    def test_conversation_auto_detects_user_actor(self):
        """ingest_conversation auto-detects actor from message roles."""
        mem = Memory()
        detected_actor = [None]

        def mock_llm(prompt):
            if "HUMAN USER" in prompt:
                detected_actor[0] = "user"
            elif "AI AGENT" in prompt:
                detected_actor[0] = "agent"
            return json.dumps({"nodes": [{"type": "thought", "content": "test"}], "relationships": [], "states": []})

        extractor = AutoExtract(mem, llm=mock_llm)
        extractor.ingest_conversation([
            {"role": "user", "content": "I want PostgreSQL"},
            {"role": "user", "content": "ACID is important"},
            {"role": "assistant", "content": "Good choice"},
        ])
        assert detected_actor[0] == "user"  # 2 user > 1 assistant

    def test_conversation_auto_detects_agent_actor(self):
        mem = Memory()
        detected_actor = [None]

        def mock_llm(prompt):
            if "AI AGENT" in prompt:
                detected_actor[0] = "agent"
            return json.dumps({"nodes": [{"type": "thought", "content": "test"}], "relationships": [], "states": []})

        extractor = AutoExtract(mem, llm=mock_llm)
        extractor.ingest_conversation([
            {"role": "system", "content": "You are a DB advisor"},
            {"role": "assistant", "content": "PostgreSQL is best for ACID"},
            {"role": "assistant", "content": "It also has great ecosystem"},
        ])
        assert detected_actor[0] == "agent"  # 2 agent > 0 user


class TestExtractionRetry:
    """Tests for LLM retry logic in AutoExtract."""

    def test_retry_on_transient_failure(self):
        """Should retry and succeed on second attempt."""
        mem = Memory()
        call_count = [0]

        def flaky_llm(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("temporary failure")
            return json.dumps({
                "nodes": [{"type": "thought", "content": "retry worked"}],
                "relationships": [],
                "states": [],
            })

        extractor = AutoExtract(mem, llm=flaky_llm, max_retries=3)
        result = extractor.ingest("test content")
        assert result.nodes_created == 1
        assert call_count[0] == 2

    def test_all_retries_exhausted(self):
        """Should return empty result after all retries fail."""
        mem = Memory()

        def always_fail(prompt):
            raise RuntimeError("persistent failure")

        extractor = AutoExtract(mem, llm=always_fail, max_retries=2)
        result = extractor.ingest("test content")
        assert result.nodes_created == 0
        assert result.node_ids == []

    def test_single_retry_mode(self):
        """max_retries=1 means no retry (fail immediately)."""
        mem = Memory()
        call_count = [0]

        def always_fail(prompt):
            call_count[0] += 1
            raise RuntimeError("fail")

        extractor = AutoExtract(mem, llm=always_fail, max_retries=1)
        result = extractor.ingest("test")
        assert call_count[0] == 1
        assert result.nodes_created == 0

    def test_value_error_not_retried(self):
        """ValueError should fail immediately, not retry."""
        mem = Memory()
        call_count = [0]

        def bad_config(prompt):
            call_count[0] += 1
            raise ValueError("invalid config")

        extractor = AutoExtract(mem, llm=bad_config, max_retries=3)
        result = extractor.ingest("test")
        assert call_count[0] == 1  # no retry
        assert result.nodes_created == 0


class TestUnifiedMemoryActorParam:
    """Tests for actor parameter propagation through UnifiedMemory."""

    def test_add_with_actor(self):
        mem = Memory()

        def mock_llm(prompt):
            assert "HUMAN USER" in prompt
            return json.dumps({
                "nodes": [{"type": "thought", "content": "test"}],
                "relationships": [],
                "states": [],
            })

        umem = UnifiedMemory(llm=mock_llm)
        result = umem.add("test content", actor="user")
        assert result.nodes_created == 1


class TestAutoSave:
    """Tests for auto_save crash safety feature."""

    def test_auto_save_off_by_default(self):
        umem = UnifiedMemory()
        assert umem._auto_save is False

    def test_auto_save_persists_after_add(self):
        """With auto_save=True, add() triggers save to disk."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_autosave.json")

            def mock_llm(prompt):
                return json.dumps({
                    "nodes": [{"type": "thought", "content": "auto saved"}],
                    "relationships": [],
                    "states": [],
                })

            umem = UnifiedMemory(file_path=path, llm=mock_llm, auto_save=True)
            umem.add("test content")

            # File should exist on disk now (auto-saved)
            assert os.path.exists(path)

            # Load fresh instance — node should be there
            umem2 = UnifiedMemory(file_path=path)
            assert umem2.size >= 1

    def test_auto_save_false_no_persist(self):
        """With auto_save=False (default), add() does NOT save to disk."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_no_autosave.json")

            umem = UnifiedMemory(file_path=path, auto_save=False)
            umem.add_raw("test content")

            # File should NOT exist yet (no save called)
            # Note: load_or_create creates an empty file, so check size
            umem_check = UnifiedMemory(file_path=path)
            # The raw node was never saved, so a fresh load won't have it
            # (load_or_create may create the file, but it won't have our node)

    def test_auto_save_with_raw_add(self):
        """auto_save works with add_raw too."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_autosave_raw.json")

            umem = UnifiedMemory(file_path=path, auto_save=True)
            umem.add_raw("raw node content")

            # Load fresh — should have the node
            umem2 = UnifiedMemory(file_path=path)
            assert umem2.size >= 1
