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


class TestGraduationValidation:
    """Tests for graph-grounded graduation — anti-semantic-inbreeding defense."""

    def test_valid_citation_kept(self):
        text = (
            "## Patterns\n"
            "thought: caching helps | 2x (2026-03-30) [evidence: abc12345]\n"
            "## Decisions\n"
        )
        result_text, validated, demoted, _reuse = ContinuityManager._validate_graduations(
            text, {"abc12345", "def67890"}, today="2026-03-30"
        )
        assert "| 2x" in result_text
        assert "ungrounded" not in result_text
        assert validated == 1
        assert demoted == 0

    def test_invalid_citation_demoted(self):
        text = (
            "## Patterns\n"
            "thought: caching helps | 2x (2026-03-30) [evidence: ffffffff]\n"
            "## Decisions\n"
        )
        result_text, validated, demoted, _reuse = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30"
        )
        assert "| 1x" in result_text
        assert "| 2x" not in result_text
        assert "(ungrounded)" in result_text
        assert validated == 0
        assert demoted == 1

    def test_3x_demoted_to_2x(self):
        text = (
            "## Patterns\n"
            "thought: principle | 3x (2026-03-30) [evidence: badbadba]\n"
            "## Decisions\n"
        )
        result_text, validated, demoted, _reuse = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30"
        )
        assert "| 2x" in result_text
        assert "| 3x" not in result_text
        assert "(ungrounded)" in result_text
        assert demoted == 1

    def test_no_citations_passthrough(self):
        """Old-format patterns without [evidence:] pass through unchanged."""
        text = (
            "## Patterns\n"
            "thought: caching helps | 2x (2026-03-30)\n"
            "thought: pooling matters | 3x (2026-03-30)\n"
            "## Decisions\n"
        )
        result_text, validated, demoted, _reuse = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30"
        )
        assert result_text == text
        assert validated == 0
        assert demoted == 0

    def test_mixed_valid_and_invalid(self):
        text = (
            "## Patterns\n"
            "thought: good pattern | 2x (2026-03-30) [evidence: abc12345]\n"
            "thought: hallucinated | 2x (2026-03-30) [evidence: ffffffff]\n"
            "## Decisions\n"
        )
        result_text, validated, demoted, _reuse = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30"
        )
        assert validated == 1
        assert demoted == 1
        # First pattern kept at 2x, second demoted to 1x
        lines = result_text.split("\n")
        assert "| 2x" in lines[1]
        assert "| 1x" in lines[2]
        assert "(ungrounded)" in lines[2]

    def test_multiple_citations_one_valid_sufficient(self):
        text = (
            "## Patterns\n"
            "thought: pattern | 2x (2026-03-30) [evidence: bad00000, abc12345]\n"
            "## Decisions\n"
        )
        result_text, validated, demoted, _reuse = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30"
        )
        assert validated == 1
        assert demoted == 0
        assert "| 2x" in result_text

    def test_1x_not_affected(self):
        """1x patterns are new observations — never checked for citations."""
        text = (
            "## Patterns\n"
            "thought: new observation | 1x (2026-03-30)\n"
            "thought: also new | 1x (2026-03-30) [evidence: ffffffff]\n"
            "## Decisions\n"
        )
        result_text, validated, demoted, _reuse = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30"
        )
        # 1x lines are never matched by _GRADUATION_RE (only matches 2x/3x)
        assert validated == 0
        assert demoted == 0

    def test_outside_patterns_section_ignored(self):
        """Citations in non-Patterns sections should not be validated."""
        text = (
            "## State\n"
            "some state | 2x (2026-03-30) [evidence: ffffffff]\n"
            "## Patterns\n"
            "thought: real pattern | 2x (2026-03-30) [evidence: abc12345]\n"
            "## Decisions\n"
        )
        result_text, validated, demoted, _reuse = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30"
        )
        # Only the Patterns section line is checked
        assert validated == 1
        assert demoted == 0
        # State section line unchanged (still has ffffffff)
        assert "ffffffff" in result_text

    def test_uppercase_citation_normalized(self):
        """LLMs may uppercase hex — citations should be case-insensitive."""
        text = (
            "## Patterns\n"
            "thought: pattern | 2x (2026-03-30) [evidence: ABC12345]\n"
            "## Decisions\n"
        )
        result_text, validated, demoted, _reuse = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30"
        )
        assert validated == 1
        assert demoted == 0

    def test_space_separated_citations(self):
        """LLMs might use spaces instead of commas between IDs."""
        text = (
            "## Patterns\n"
            "thought: pattern | 2x (2026-03-30) [evidence: bad00000 abc12345]\n"
            "## Decisions\n"
        )
        result_text, validated, demoted, _reuse = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30"
        )
        assert validated == 1
        assert demoted == 0

    def test_long_id_truncated_to_8_chars(self):
        """LLM might cite full 64-char ID — should be truncated to 8 for matching."""
        text = (
            "## Patterns\n"
            "thought: pattern | 2x (2026-03-30) [evidence: abc12345ffffffffffffffff]\n"
            "## Decisions\n"
        )
        result_text, validated, demoted, _reuse = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30"
        )
        assert validated == 1
        assert demoted == 0

    def test_carried_forward_evidence_not_demoted(self):
        """Patterns from previous sessions (old dates) should pass through unchanged."""
        text = (
            "## Patterns\n"
            "thought: old pattern | 2x (2026-03-28) [evidence: abc12345]\n"
            "thought: new pattern | 2x (2026-03-30) [evidence: def67890]\n"
            "## Decisions\n"
        )
        # abc12345 is NOT in valid_ids, but its date is old → should pass through
        # def67890 IS in valid_ids and its date matches today → validated
        result_text, validated, demoted, _reuse = ContinuityManager._validate_graduations(
            text, {"def67890"}, today="2026-03-30"
        )
        assert validated == 1
        assert demoted == 0
        # Old pattern still at 2x (not demoted despite abc12345 not in current nodes)
        assert "2026-03-28" in result_text
        assert "ungrounded" not in result_text

    def test_graduation_validation_through_produce(self):
        """Integration: graduation validation works through the full produce() pipeline."""
        import datetime
        today = datetime.date.today().isoformat()

        # Node ID 50d7c6fd = "Connection pooling will be the real bottleneck"
        # from _make_session_memory(). Use today's date so validation fires.
        response_with_valid_citation = f"""# Agent — Memory (v1)

## State
Working on database selection.

## Patterns
{{database_architecture:
  thought: connection pooling is critical | 2x ({today}) [evidence: 50d7c6fd]
  thought: ACID compliance matters | 2x ({today}) [evidence: ffffffff]
}}

## Decisions
[decided(rationale: "ACID required", on: "{today}")] Use PostgreSQL

## Context
Selected PostgreSQL, investigating pooling."""

        mgr = ContinuityManager(
            llm=_make_mock_llm(response_with_valid_citation),
        )
        mem = _make_session_memory()
        result = mgr.produce(mem)

        # One citation valid (50d7c6fd exists), one invalid (ffffffff doesn't)
        assert result.graduations_validated == 1
        assert result.graduations_demoted == 1
        assert "(ungrounded)" in result.text
        # The valid graduation should still be 2x
        assert "| 2x" in result.text
        # The invalid one should be demoted to 1x
        assert "| 1x" in result.text


class TestExplanationValidation:
    """Tests for explain-your-evidence — citation relevance checking."""

    def test_explanation_with_node_content_overlap_passes(self):
        text = (
            '## Patterns\n'
            'thought: pooling matters | 2x (2026-03-30) '
            '[evidence: abc12345 "connection pooling identified as bottleneck"]\n'
            '## Decisions\n'
        )
        node_map = {"abc12345": "Connection pooling will be the real bottleneck"}
        result_text, validated, demoted, _r = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30", node_content_map=node_map
        )
        assert validated == 1
        assert demoted == 0

    def test_explanation_without_overlap_demoted(self):
        text = (
            '## Patterns\n'
            'thought: pooling matters | 2x (2026-03-30) '
            '[evidence: abc12345 "confirms the pattern"]\n'
            '## Decisions\n'
        )
        node_map = {"abc12345": "Connection pooling will be the real bottleneck"}
        result_text, validated, demoted, _r = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30", node_content_map=node_map
        )
        # "confirms the pattern" has no meaningful overlap with node content
        assert validated == 0
        assert demoted == 1
        assert "(ungrounded)" in result_text

    def test_no_explanation_still_passes_id_check(self):
        """Citations without explanations pass on ID alone (backward compat)."""
        text = (
            "## Patterns\n"
            "thought: pattern | 2x (2026-03-30) [evidence: abc12345]\n"
            "## Decisions\n"
        )
        node_map = {"abc12345": "Some node content"}
        result_text, validated, demoted, _r = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30", node_content_map=node_map
        )
        # No explanation = no overlap check, just ID validation
        assert validated == 1
        assert demoted == 0

    def test_no_node_map_skips_explanation_check(self):
        """Without node_content_map, explanation check is skipped."""
        text = (
            '## Patterns\n'
            'thought: pattern | 2x (2026-03-30) '
            '[evidence: abc12345 "totally irrelevant words"]\n'
            '## Decisions\n'
        )
        result_text, validated, demoted, _r = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30", node_content_map=None
        )
        assert validated == 1
        assert demoted == 0


class TestCitationReuse:
    """Tests for citation gaming detection."""

    def test_reuse_count_tracked(self):
        text = (
            "## Patterns\n"
            "thought: pattern A | 2x (2026-03-30) [evidence: abc12345]\n"
            "thought: pattern B | 2x (2026-03-30) [evidence: abc12345]\n"
            "thought: pattern C | 2x (2026-03-30) [evidence: abc12345]\n"
            "## Decisions\n"
        )
        _text, _v, _d, reuse_max = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30"
        )
        assert reuse_max == 3


class TestContinuityMeta:
    """Tests for continuity metadata sidecar (session tracking, fail-safe sunset)."""

    def test_meta_defaults_when_missing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = os.path.join(tmpdir, "agent.json")
            meta = ContinuityManager.load_meta(mem_path)
            assert meta["sessions_produced"] == 0
            assert meta["citations_seen"] is False
            assert meta["format_version"] == 1

    def test_meta_save_and_load_roundtrip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = os.path.join(tmpdir, "agent.json")
            meta = {"sessions_produced": 5, "citations_seen": True, "format_version": 1}
            ContinuityManager.save_meta(meta, mem_path)
            loaded = ContinuityManager.load_meta(mem_path)
            assert loaded == meta

    def test_meta_path_follows_sidecar_pattern(self):
        path = ContinuityManager.meta_path("/tmp/agent.json")
        assert path == "/tmp/agent.continuity.meta.json"

    def test_corrupt_meta_returns_defaults(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = os.path.join(tmpdir, "agent.json")
            meta_path = ContinuityManager.meta_path(mem_path)
            with open(meta_path, "w") as f:
                f.write("NOT JSON")
            meta = ContinuityManager.load_meta(mem_path)
            assert meta["sessions_produced"] == 0


class TestFailSafeSunset:
    """Tests for citation requirement enforcement after first successful citation."""

    def test_bare_graduation_passes_before_sunset(self):
        """Before citations_seen, bare graduations (no [evidence:]) pass through."""
        text = (
            "## Patterns\n"
            "thought: pattern | 2x (2026-03-30)\n"
            "## Decisions\n"
        )
        result_text, validated, demoted, _r = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30", citations_seen=False
        )
        assert demoted == 0
        assert "| 2x" in result_text
        assert "needs-evidence" not in result_text

    def test_bare_graduation_demoted_after_sunset(self):
        """After citations_seen, bare graduations are demoted with (needs-evidence)."""
        text = (
            "## Patterns\n"
            "thought: pattern | 2x (2026-03-30)\n"
            "## Decisions\n"
        )
        result_text, validated, demoted, _r = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30", citations_seen=True
        )
        assert demoted == 1
        assert "| 1x" in result_text
        assert "(needs-evidence)" in result_text

    def test_bare_3x_demoted_to_2x_after_sunset(self):
        text = (
            "## Patterns\n"
            "thought: pattern | 3x (2026-03-30)\n"
            "## Decisions\n"
        )
        result_text, validated, demoted, _r = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30", citations_seen=True
        )
        assert demoted == 1
        assert "| 2x" in result_text
        assert "(needs-evidence)" in result_text

    def test_old_date_bare_graduation_unaffected_by_sunset(self):
        """Carried-forward bare graduations from old sessions are not demoted."""
        text = (
            "## Patterns\n"
            "thought: old pattern | 2x (2026-03-28)\n"
            "## Decisions\n"
        )
        result_text, validated, demoted, _r = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30", citations_seen=True
        )
        assert demoted == 0
        assert "| 2x" in result_text

    def test_cited_graduation_still_passes_after_sunset(self):
        """Properly cited graduations pass regardless of sunset state."""
        text = (
            "## Patterns\n"
            "thought: pattern | 2x (2026-03-30) [evidence: abc12345]\n"
            "## Decisions\n"
        )
        result_text, validated, demoted, _r = ContinuityManager._validate_graduations(
            text, {"abc12345"}, today="2026-03-30", citations_seen=True
        )
        assert validated == 1
        assert demoted == 0

    def test_no_reuse(self):
        text = (
            "## Patterns\n"
            "thought: pattern A | 2x (2026-03-30) [evidence: abc12345]\n"
            "thought: pattern B | 2x (2026-03-30) [evidence: def67890]\n"
            "## Decisions\n"
        )
        _text, _v, _d, reuse_max = ContinuityManager._validate_graduations(
            text, {"abc12345", "def67890"}, today="2026-03-30"
        )
        assert reuse_max == 1

    def test_reuse_in_produce_result(self):
        """citation_reuse_max flows through to ContinuityResult."""
        import datetime
        today = datetime.date.today().isoformat()

        response = f"""# Test — Memory (v1)

## State
Testing.

## Patterns
thought: A | 2x ({today}) [evidence: 50d7c6fd]
thought: B | 2x ({today}) [evidence: 50d7c6fd]
thought: C | 2x ({today}) [evidence: 50d7c6fd]

## Decisions
None.

## Context
Testing citation reuse."""

        mgr = ContinuityManager(llm=_make_mock_llm(response))
        mem = _make_session_memory()
        result = mgr.produce(mem)
        assert result.citation_reuse_max == 3
