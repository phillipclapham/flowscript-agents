"""Tests for ContinuityManager — Layer 1 session boundary compression."""

import json
import os
import tempfile

import pytest

from flowscript_agents.memory import Memory
from flowscript_agents.continuity import (
    ContinuityManager,
    ContinuityResult,
    _format_session_nodes,
    _build_wrap_prompt,
)


# =============================================================================
# Mock LLM
# =============================================================================


def _make_mock_llm(response: str | None = None):
    """Create a mock LLM that returns a canned response or a valid continuity file."""

    default_response = """# Agent — Memory

## State
Working on database selection for the new API service. PostgreSQL chosen, Redis deferred.
Next: implement connection pooling.

## Patterns
{database_selection:
  thought: ACID compliance outweighs raw speed for this use case | 1x
  thought: connection pooling is the real bottleneck, not query speed | 1x
  ? horizontal scaling strategy — single writer vs multi-writer | 1x
}

## Decisions
[decided(rationale: "ACID compliance required for financial transactions, PostgreSQL provides this natively", on: "2026-03-29")] Use PostgreSQL over Redis for primary storage

## Context
Evaluated database options for the new API service. Started with Redis vs PostgreSQL comparison. Financial transaction requirements made ACID compliance non-negotiable, which ruled out Redis as primary store. PostgreSQL selected. Redis may still serve as cache layer. Next focus is connection pooling architecture."""

    if response is not None:
        return lambda prompt: response
    return lambda prompt: default_response


def _make_tracking_llm(response: str | None = None):
    """Mock LLM that also tracks what prompts it received."""
    calls: list[str] = []
    llm_fn = _make_mock_llm(response)

    def tracking_llm(prompt: str) -> str:
        calls.append(prompt)
        return llm_fn(prompt)

    return tracking_llm, calls


# =============================================================================
# Helper: create a populated Memory
# =============================================================================


def _make_session_memory() -> Memory:
    """Create a Memory with realistic session data for testing."""
    mem = Memory()
    mem.session_start()

    q = mem.question("Which database for the new API?")
    pg = mem.alternative(q, "PostgreSQL — ACID compliant, mature ecosystem")
    redis = mem.alternative(q, "Redis — fast but no ACID")
    pg.decide(rationale="ACID compliance required for financial transactions")

    mem.thought("Connection pooling will be the real bottleneck")
    mem.thought("Horizontal scaling needs investigation")
    mem.action("Implement connection pooling with pgbouncer")

    return mem


# =============================================================================
# Tests: ContinuityManager.produce()
# =============================================================================


class TestProduce:
    """Tests for the core produce() method."""

    def test_produce_returns_result(self):
        mgr = ContinuityManager(llm=_make_mock_llm())
        mem = _make_session_memory()
        result = mgr.produce(mem)

        assert isinstance(result, ContinuityResult)
        assert result.char_count > 0
        assert result.text.startswith("# Agent — Memory")
        assert result.session_nodes_count > 0
        assert not result.truncated

    def test_produce_has_four_sections(self):
        mgr = ContinuityManager(llm=_make_mock_llm())
        mem = _make_session_memory()
        result = mgr.produce(mem)

        assert "## State" in result.text
        assert "## Patterns" in result.text
        assert "## Decisions" in result.text
        assert "## Context" in result.text

    def test_produce_measures_sections(self):
        mgr = ContinuityManager(llm=_make_mock_llm())
        mem = _make_session_memory()
        result = mgr.produce(mem)

        assert "State" in result.section_sizes
        assert "Patterns" in result.section_sizes
        assert "Decisions" in result.section_sizes
        assert "Context" in result.section_sizes
        # Each section should have non-zero size
        for name, size in result.section_sizes.items():
            if name != "_header":
                assert size > 0, f"Section '{name}' has zero size"

    def test_produce_counts_patterns(self):
        mgr = ContinuityManager(llm=_make_mock_llm())
        mem = _make_session_memory()
        result = mgr.produce(mem)

        # The mock response has "| 1x" patterns
        assert result.patterns_extracted >= 1

    def test_produce_with_existing_continuity(self):
        """Produce should pass existing continuity to the LLM."""
        tracking_llm, calls = _make_tracking_llm()
        mgr = ContinuityManager(llm=tracking_llm)
        mem = _make_session_memory()

        existing = "# Agent — Memory\n\n## State\nPrevious state\n"
        result = mgr.produce(mem, existing_continuity=existing)

        assert len(calls) == 1
        assert "Previous state" in calls[0]
        assert "<existing_continuity>" in calls[0]

    def test_produce_without_existing_continuity(self):
        """First session should indicate no existing continuity."""
        tracking_llm, calls = _make_tracking_llm()
        mgr = ContinuityManager(llm=tracking_llm)
        mem = _make_session_memory()

        result = mgr.produce(mem, existing_continuity=None)

        assert len(calls) == 1
        assert "first session" in calls[0].lower()

    def test_produce_sends_session_data(self):
        """Prompt should contain formatted session nodes."""
        tracking_llm, calls = _make_tracking_llm()
        mgr = ContinuityManager(llm=tracking_llm)
        mem = _make_session_memory()

        result = mgr.produce(mem)

        assert len(calls) == 1
        prompt = calls[0]
        # Should contain node content
        assert "PostgreSQL" in prompt or "database" in prompt.lower()
        assert "<session_data>" in prompt

    def test_produce_respects_max_chars(self):
        """Output should not exceed max_chars."""
        mgr = ContinuityManager(llm=_make_mock_llm(), max_chars=500)
        mem = _make_session_memory()
        result = mgr.produce(mem)

        assert result.char_count <= 500
        assert result.truncated  # mock response is longer than 500

    def test_produce_custom_project_name(self):
        mgr = ContinuityManager(llm=_make_mock_llm(), project_name="MyProject")
        mem = _make_session_memory()

        # The project name appears in the prompt
        tracking_llm, calls = _make_tracking_llm()
        mgr2 = ContinuityManager(llm=tracking_llm, project_name="MyProject")
        mgr2.produce(mem)
        assert "MyProject" in calls[0]

    def test_produce_empty_memory(self):
        """Should handle empty memory gracefully."""
        mgr = ContinuityManager(llm=_make_mock_llm())
        mem = Memory()

        result = mgr.produce(mem)
        assert isinstance(result, ContinuityResult)
        assert result.session_nodes_count == 0


class TestProduceErrors:
    """Tests for error handling in produce()."""

    def test_llm_failure_propagates(self):
        """LLM exceptions should propagate — caller handles them."""
        def failing_llm(prompt):
            raise RuntimeError("LLM unavailable")

        mgr = ContinuityManager(llm=failing_llm)
        mem = _make_session_memory()

        with pytest.raises(RuntimeError, match="LLM unavailable"):
            mgr.produce(mem)

    def test_malformed_output_falls_back_to_existing(self):
        """If LLM produces output missing required sections, keep existing continuity."""
        bad_llm = lambda p: "# Agent — Memory\n\n## Summary\nJust a summary, no real sections."
        mgr = ContinuityManager(llm=bad_llm)
        mem = _make_session_memory()

        existing = "# Agent — Memory (v1)\n\n## State\nOld state\n\n## Patterns\nOld\n\n## Decisions\nOld\n\n## Context\nOld"
        result = mgr.produce(mem, existing_continuity=existing)

        # Should fall back to existing
        assert result.text == existing
        assert "Old state" in result.text

    def test_malformed_output_first_session_uses_output(self):
        """If LLM produces bad output on first session (no existing), use it anyway."""
        bad_llm = lambda p: "# Agent — Memory\n\n## Summary\nJust a summary."
        mgr = ContinuityManager(llm=bad_llm)
        mem = _make_session_memory()

        result = mgr.produce(mem, existing_continuity=None)
        # No existing to fall back to — use the output anyway
        assert "Summary" in result.text


class TestStructureValidation:
    """Tests for _validate_structure."""

    def test_valid_structure(self):
        text = "# Agent\n\n## State\nfoo\n\n## Patterns\nbar\n\n## Decisions\nbaz\n\n## Context\nqux"
        assert ContinuityManager._validate_structure(text) is True

    def test_missing_section(self):
        text = "# Agent\n\n## State\nfoo\n\n## Patterns\nbar\n\n## Context\nqux"
        assert ContinuityManager._validate_structure(text) is False

    def test_case_insensitive(self):
        text = "# Agent\n\n## STATE\nfoo\n\n## Patterns\nbar\n\n## DECISIONS\nbaz\n\n## Context\nqux"
        assert ContinuityManager._validate_structure(text) is True

    def test_numbered_sections_still_valid(self):
        """LLMs sometimes add numbers — should still validate."""
        text = "# Agent\n\n## Section 1: State\nfoo\n\n## Section 2: Patterns\nbar\n\n## Section 3: Decisions\nbaz\n\n## Section 4: Context\nqux"
        assert ContinuityManager._validate_structure(text) is True


class TestProduceFromNodes:
    """Tests for the alternative produce_from_nodes() method."""

    def test_produce_from_nodes_works(self):
        mgr = ContinuityManager(llm=_make_mock_llm())
        mem = _make_session_memory()

        result = mgr.produce_from_nodes(
            nodes=list(mem._nodes.values()),
            relationships=list(mem._relationships),
            states=list(mem._states),
        )

        assert isinstance(result, ContinuityResult)
        assert result.text.startswith("# Agent — Memory")


# =============================================================================
# Tests: File I/O
# =============================================================================


class TestFileIO:
    """Tests for save/load sidecar file operations."""

    def test_continuity_path(self):
        assert ContinuityManager.continuity_path("/tmp/agent.json") == "/tmp/agent.continuity.md"
        assert ContinuityManager.continuity_path("./data/mem.json") == "data/mem.continuity.md"
        assert ContinuityManager.continuity_path("/foo/bar.json") == "/foo/bar.continuity.md"

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = os.path.join(tmpdir, "agent.json")
            mgr = ContinuityManager(llm=_make_mock_llm())

            text = "# Test — Memory\n\n## State\nTest state\n"
            saved_path = mgr.save(text, mem_path)

            assert os.path.exists(saved_path)
            assert saved_path.endswith(".continuity.md")

            loaded = ContinuityManager.load(mem_path)
            assert loaded == text

    def test_load_nonexistent(self):
        loaded = ContinuityManager.load("/tmp/nonexistent_flowscript_test.json")
        assert loaded is None

    def test_save_atomic(self):
        """Save should be atomic — no partial writes visible."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = os.path.join(tmpdir, "agent.json")
            mgr = ContinuityManager(llm=_make_mock_llm())

            # Write initial content
            mgr.save("version 1", mem_path)
            assert ContinuityManager.load(mem_path) == "version 1"

            # Overwrite
            mgr.save("version 2", mem_path)
            assert ContinuityManager.load(mem_path) == "version 2"

            # No temp files left
            files = os.listdir(tmpdir)
            assert not any(f.endswith(".tmp") for f in files)


# =============================================================================
# Tests: Session data formatting
# =============================================================================


class TestFormatSessionNodes:
    """Tests for the internal _format_session_nodes function."""

    def test_empty_nodes(self):
        result = _format_session_nodes([], [], [])
        assert "No nodes" in result

    def test_formats_node_types(self):
        mem = _make_session_memory()
        nodes = list(mem._nodes.values())
        result = _format_session_nodes(nodes, [], [])

        # Should have type headers
        assert "Question" in result or "question" in result.lower()
        assert "Thought" in result or "thought" in result.lower()

    def test_formats_relationships(self):
        mem = _make_session_memory()
        result = _format_session_nodes(
            list(mem._nodes.values()),
            list(mem._relationships),
            [],
        )
        assert "Relationship" in result

    def test_formats_states(self):
        mem = _make_session_memory()
        result = _format_session_nodes(
            list(mem._nodes.values()),
            [],
            list(mem._states),
        )
        assert "State" in result
        assert "decided" in result.lower() or "rationale" in result.lower()

    def test_includes_temporal_info(self):
        mem = _make_session_memory()
        temporal = {}
        for nid in list(mem._nodes.keys())[:1]:
            temporal[nid] = {"tier": "developing", "frequency": 3}

        result = _format_session_nodes(
            list(mem._nodes.values()),
            [],
            [],
            temporal_map=temporal,
        )
        assert "developing" in result
        assert "3x" in result


# =============================================================================
# Tests: Wrap prompt construction
# =============================================================================


class TestBuildWrapPrompt:
    """Tests for the wrap prompt builder."""

    def test_includes_session_data(self):
        prompt = _build_wrap_prompt("some session data", None, "Test", 20000)
        assert "some session data" in prompt
        assert "<session_data>" in prompt

    def test_includes_existing_continuity(self):
        prompt = _build_wrap_prompt("data", "existing stuff", "Test", 20000)
        assert "existing stuff" in prompt
        assert "<existing_continuity>" in prompt

    def test_no_existing_continuity(self):
        prompt = _build_wrap_prompt("data", None, "Test", 20000)
        assert "first session" in prompt.lower()

    def test_includes_max_chars_instruction(self):
        prompt = _build_wrap_prompt("data", None, "Test", 15000)
        assert "15000" in prompt

    def test_includes_project_name(self):
        prompt = _build_wrap_prompt("data", None, "MyAgent", 20000)
        assert "MyAgent" in prompt

    def test_includes_temporal_graduation_instructions(self):
        prompt = _build_wrap_prompt("data", None, "Test", 20000)
        assert "1x" in prompt
        assert "2x" in prompt
        assert "3x" in prompt
        assert "temporal graduation" in prompt.lower() or "graduation" in prompt.lower()

    def test_includes_decision_lifecycle_instructions(self):
        prompt = _build_wrap_prompt("data", None, "Test", 20000)
        assert "lifecycle" in prompt.lower() or "cluster" in prompt.lower()

    def test_includes_flowscript_markers(self):
        prompt = _build_wrap_prompt("data", None, "Test", 20000)
        assert "thought:" in prompt
        assert "decided" in prompt
        assert "blocked" in prompt
        assert "><[" in prompt  # tension marker


# =============================================================================
# Tests: Configuration
# =============================================================================


class TestConfiguration:
    """Tests for ContinuityManager configuration."""

    def test_default_max_chars(self):
        mgr = ContinuityManager(llm=_make_mock_llm())
        assert mgr.max_chars == 20000

    def test_custom_max_chars(self):
        mgr = ContinuityManager(llm=_make_mock_llm(), max_chars=10000)
        assert mgr.max_chars == 10000

    def test_env_var_max_chars(self, monkeypatch):
        monkeypatch.setenv("FLOWSCRIPT_CONTINUITY_MAX_CHARS", "30000")
        mgr = ContinuityManager(llm=_make_mock_llm())
        assert mgr.max_chars == 30000

    def test_default_project_name(self):
        mgr = ContinuityManager(llm=_make_mock_llm())
        assert mgr.project_name == "Agent"

    def test_custom_project_name(self):
        mgr = ContinuityManager(llm=_make_mock_llm(), project_name="FlowBot")
        assert mgr.project_name == "FlowBot"

    def test_env_var_project_name(self, monkeypatch):
        monkeypatch.setenv("FLOWSCRIPT_PROJECT_NAME", "EnvBot")
        mgr = ContinuityManager(llm=_make_mock_llm())
        assert mgr.project_name == "EnvBot"


# =============================================================================
# Tests: Truncation
# =============================================================================


class TestTruncation:
    """Tests for the section-aware truncation logic."""

    def test_no_truncation_when_under_limit(self):
        mgr = ContinuityManager(llm=_make_mock_llm(), max_chars=50000)
        mem = _make_session_memory()
        result = mgr.produce(mem)
        assert not result.truncated

    def test_truncation_preserves_structure(self):
        mgr = ContinuityManager(llm=_make_mock_llm(), max_chars=400)
        mem = _make_session_memory()
        result = mgr.produce(mem)

        assert result.truncated
        # Should still start with the title
        assert result.text.startswith("# Agent")
        # Should have at least State section
        assert "## State" in result.text
