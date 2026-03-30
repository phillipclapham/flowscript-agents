"""Tests for Article 86 Explanation Generator."""

import pytest

from flowscript_agents import Memory
from flowscript_agents.explain import explain, explain_counterfactual, AUDIENCE_GENERAL, AUDIENCE_LEGAL, AUDIENCE_TECHNICAL
from flowscript_agents.query import CausalAncestry, MinimalWhy, CausalTree


# =============================================================================
# Fixtures — build reusable reasoning graphs
# =============================================================================


@pytest.fixture
def simple_causal_mem():
    """Single-step causal chain: one cause → one effect."""
    mem = Memory()
    cause = mem.statement("credit score below threshold")
    effect = mem.statement("loan application rejected")
    cause.causes(effect)
    return mem, cause, effect


@pytest.fixture
def chain_causal_mem():
    """Multi-step chain: income low → can't repay → rejected."""
    mem = Memory()
    root = mem.statement("applicant income below minimum requirement")
    mid = mem.statement("debt-to-income ratio exceeds policy limit")
    leaf = mem.statement("loan application rejected")
    root.causes(mid)
    mid.causes(leaf)
    return mem, root, mid, leaf


@pytest.fixture
def deep_chain_mem():
    """4-step chain for depth testing."""
    mem = Memory()
    a = mem.statement("market volatility high")
    b = mem.statement("collateral value uncertain")
    c = mem.statement("risk assessment failed")
    d = mem.statement("credit line suspended")
    a.causes(b)
    b.causes(c)
    c.causes(d)
    return mem, a, b, c, d


@pytest.fixture
def no_ancestors_mem():
    """Node with no causal ancestors (standalone)."""
    mem = Memory()
    node = mem.statement("initial policy decision")
    return mem, node


# =============================================================================
# CausalAncestry — general audience
# =============================================================================


class TestCausalAncestryGeneral:
    def test_simple_chain_contains_cause_and_effect(self, simple_causal_mem):
        mem, cause, effect = simple_causal_mem
        result = mem.query.why(effect.id)
        text = explain(result)
        assert "credit score below threshold" in text
        assert "loan application rejected" in text

    def test_simple_chain_header(self, simple_causal_mem):
        mem, cause, effect = simple_causal_mem
        result = mem.query.why(effect.id)
        text = explain(result)
        assert "Decision Explanation" in text

    def test_chain_contains_all_steps(self, chain_causal_mem):
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id)
        text = explain(result)
        assert "applicant income below minimum requirement" in text
        assert "debt-to-income ratio exceeds policy limit" in text
        assert "loan application rejected" in text

    def test_chain_mentions_starting_point(self, chain_causal_mem):
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id)
        text = explain(result)
        assert "Starting point" in text

    def test_chain_mentions_final_outcome(self, chain_causal_mem):
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id)
        text = explain(result)
        assert "Final outcome" in text or "outcome" in text.lower()

    def test_summary_mentions_root_and_target(self, chain_causal_mem):
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id)
        text = explain(result)
        assert "Summary" in text
        assert "applicant income below minimum requirement" in text

    def test_deep_chain_step_count(self, deep_chain_mem):
        mem, a, b, c, d = deep_chain_mem
        result = mem.query.why(d.id)
        assert isinstance(result, CausalAncestry)
        text = explain(result)
        assert "3 steps" in text or "3 step" in text

    def test_no_ancestors_handled(self, no_ancestors_mem):
        mem, node = no_ancestors_mem
        result = mem.query.why(node.id)
        text = explain(result)
        assert "initial policy decision" in text or "no causal history" in text.lower() or "no recorded" in text.lower()

    def test_default_audience_is_general(self, simple_causal_mem):
        mem, cause, effect = simple_causal_mem
        result = mem.query.why(effect.id)
        text_default = explain(result)
        text_general = explain(result, audience="general")
        assert text_default == text_general

    def test_deterministic_output(self, chain_causal_mem):
        """Same input always produces same output."""
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id)
        text1 = explain(result)
        text2 = explain(result)
        assert text1 == text2

    def test_returns_string(self, simple_causal_mem):
        mem, cause, effect = simple_causal_mem
        result = mem.query.why(effect.id)
        text = explain(result)
        assert isinstance(text, str)
        assert len(text) > 0


# =============================================================================
# CausalAncestry — legal audience
# =============================================================================


class TestCausalAncestryLegal:
    def test_legal_header(self, simple_causal_mem):
        mem, cause, effect = simple_causal_mem
        result = mem.query.why(effect.id)
        text = explain(result, audience="legal")
        assert "Article 86" in text
        assert "AUTOMATED DECISION EXPLANATION" in text

    def test_legal_includes_certification(self, simple_causal_mem):
        mem, cause, effect = simple_causal_mem
        result = mem.query.why(effect.id)
        text = explain(result, audience="legal")
        assert "audit trail" in text.lower() or "certification" in text.upper() or "CERTIFICATION" in text

    def test_legal_includes_subject_when_provided(self, simple_causal_mem):
        mem, cause, effect = simple_causal_mem
        result = mem.query.why(effect.id)
        text = explain(result, subject="Applicant ID #4821", audience="legal")
        assert "Applicant ID #4821" in text

    def test_legal_no_subject_when_omitted(self, simple_causal_mem):
        mem, cause, effect = simple_causal_mem
        result = mem.query.why(effect.id)
        text = explain(result, audience="legal")
        assert "Subject:" not in text

    def test_legal_includes_causal_sequence(self, chain_causal_mem):
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id)
        text = explain(result, audience="legal")
        assert "CAUSAL SEQUENCE" in text
        assert "applicant income below minimum requirement" in text
        assert "debt-to-income ratio exceeds policy limit" in text

    def test_legal_includes_foundational_factor(self, chain_causal_mem):
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id)
        text = explain(result, audience="legal")
        assert "FOUNDATIONAL FACTOR" in text
        assert "applicant income below minimum requirement" in text

    def test_legal_mentions_hash_chain(self, simple_causal_mem):
        mem, cause, effect = simple_causal_mem
        result = mem.query.why(effect.id)
        text = explain(result, audience="legal")
        assert "hash" in text.lower()

    def test_legal_depth_reported(self, chain_causal_mem):
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id)
        text = explain(result, audience="legal")
        # Chain has 2 nodes, so depth should be mentioned
        assert "step" in text.lower()

    def test_subject_not_in_general(self, simple_causal_mem):
        """subject= param is for legal mode only — shouldn't bleed into general."""
        mem, cause, effect = simple_causal_mem
        result = mem.query.why(effect.id)
        text = explain(result, subject="Applicant #999", audience="general")
        assert "Applicant #999" not in text


# =============================================================================
# CausalAncestry — technical audience
# =============================================================================


class TestCausalAncestryTechnical:
    def test_technical_returns_str_repr(self, simple_causal_mem):
        mem, cause, effect = simple_causal_mem
        result = mem.query.why(effect.id)
        text = explain(result, audience="technical")
        assert text == str(result)

    def test_technical_differs_from_general(self, chain_causal_mem):
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id)
        general = explain(result, audience="general")
        technical = explain(result, audience="technical")
        assert general != technical


# =============================================================================
# MinimalWhy
# =============================================================================


class TestMinimalWhy:
    def test_minimal_general_contains_root_cause(self, chain_causal_mem):
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id, format="minimal")
        assert isinstance(result, MinimalWhy)
        text = explain(result)
        assert "applicant income below minimum requirement" in text

    def test_minimal_general_header(self, chain_causal_mem):
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id, format="minimal")
        text = explain(result)
        assert "Decision Explanation" in text

    def test_minimal_legal_has_article86(self, chain_causal_mem):
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id, format="minimal")
        text = explain(result, audience="legal")
        assert "Article 86" in text

    def test_minimal_legal_with_subject(self, simple_causal_mem):
        mem, cause, effect = simple_causal_mem
        result = mem.query.why(effect.id, format="minimal")
        text = explain(result, subject="Case #100", audience="legal")
        assert "Case #100" in text

    def test_minimal_general_shows_path_arrow(self, chain_causal_mem):
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id, format="minimal")
        assert isinstance(result, MinimalWhy)
        text = explain(result)
        # Should show the → arrow path
        assert "→" in text

    def test_minimal_technical_returns_repr(self, simple_causal_mem):
        mem, cause, effect = simple_causal_mem
        result = mem.query.why(effect.id, format="minimal")
        text = explain(result, audience="technical")
        assert text == str(result)


# =============================================================================
# CausalTree
# =============================================================================


class TestCausalTree:
    def test_tree_general_contains_target(self, chain_causal_mem):
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id, format="tree")
        assert isinstance(result, CausalTree)
        text = explain(result)
        assert "loan application rejected" in text

    def test_tree_general_header(self, chain_causal_mem):
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id, format="tree")
        text = explain(result)
        assert "Decision Explanation" in text

    def test_tree_legal_has_article86(self, chain_causal_mem):
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id, format="tree")
        text = explain(result, audience="legal")
        assert "Article 86" in text

    def test_tree_mentions_ancestor_count(self, chain_causal_mem):
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id, format="tree")
        text = explain(result)
        # Should mention the number of contributing factors
        assert "factor" in text.lower() or "ancestor" in text.lower()

    def test_tree_technical_returns_repr(self, chain_causal_mem):
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id, format="tree")
        text = explain(result, audience="technical")
        assert text == str(result)

    def test_tree_contains_intermediate_nodes(self, chain_causal_mem):
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id, format="tree")
        text = explain(result)
        assert "debt-to-income ratio exceeds policy limit" in text or "applicant income" in text


# =============================================================================
# Input validation
# =============================================================================


class TestValidation:
    def test_invalid_audience_raises(self, simple_causal_mem):
        mem, cause, effect = simple_causal_mem
        result = mem.query.why(effect.id)
        with pytest.raises(ValueError, match="audience must be"):
            explain(result, audience="executive")

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="CausalAncestry, MinimalWhy, or CausalTree"):
            explain("not a why result")  # type: ignore

    def test_none_raises(self):
        with pytest.raises(TypeError):
            explain(None)  # type: ignore


# =============================================================================
# Article 86 compliance properties
# =============================================================================


class TestArticle86Compliance:
    """These tests verify the properties required by EU AI Act Article 86."""

    def test_general_mode_avoids_technical_jargon(self, chain_causal_mem):
        """Plain language requirement: no internal type names in general output."""
        mem, root, mid, leaf = chain_causal_mem
        result = mem.query.why(leaf.id)
        text = explain(result, audience="general")
        # Should not expose internal type names
        assert "CausalAncestry" not in text
        assert "CausalChainNode" not in text
        assert "RelationType" not in text

    def test_legal_mode_identifies_regulation(self, simple_causal_mem):
        """Legal mode must cite the specific regulatory basis."""
        mem, cause, effect = simple_causal_mem
        result = mem.query.why(effect.id)
        text = explain(result, audience="legal")
        assert "EU AI Act" in text
        assert "Article 86" in text

    def test_legal_mode_mentions_verifiability(self, simple_causal_mem):
        """Legal mode must indicate that the explanation can be verified."""
        mem, cause, effect = simple_causal_mem
        result = mem.query.why(effect.id)
        text = explain(result, audience="legal")
        assert "verif" in text.lower() or "audit" in text.lower()

    def test_explanation_is_non_empty_for_all_formats(self, simple_causal_mem):
        """All three why() formats must produce non-empty explanations."""
        mem, cause, effect = simple_causal_mem
        for fmt in ("chain", "minimal", "tree"):
            result = mem.query.why(effect.id, format=fmt)
            text = explain(result)
            assert len(text.strip()) > 0, f"Empty explanation for format={fmt}"

    def test_explanation_contains_decision_for_chain_and_tree(self, simple_causal_mem):
        """The target decision must appear in chain and tree explanations.

        Note: format="minimal" (MinimalWhy) omits the target node by design —
        it stores only the causal ancestors. Chain and tree are the
        Article 86-compliant formats.
        """
        mem, cause, effect = simple_causal_mem
        for fmt in ("chain", "tree"):
            result = mem.query.why(effect.id, format=fmt)
            text = explain(result)
            assert "loan application rejected" in text, f"Target missing for format={fmt}"

    def test_minimal_why_omits_target_by_design(self, simple_causal_mem):
        """Document: MinimalWhy chain contains ancestors only, not the target.

        This is a data structure constraint — the target is external to
        MinimalWhy.chain. For full Article 86 output, use format='chain'.
        """
        mem, cause, effect = simple_causal_mem
        result = mem.query.why(effect.id, format="minimal")
        assert isinstance(result, MinimalWhy)
        # The chain contains the root/ancestors, not the target
        assert "loan application rejected" not in result.chain

    def test_explanation_contains_root_cause_for_all_formats(self, simple_causal_mem):
        """The root cause must appear in all explanations."""
        mem, cause, effect = simple_causal_mem
        for fmt in ("chain", "minimal", "tree"):
            result = mem.query.why(effect.id, format=fmt)
            text = explain(result)
            assert "credit score below threshold" in text, f"Root cause missing for format={fmt}"


# =============================================================================
# explain_counterfactual tests
# =============================================================================


class TestExplainCounterfactual:
    """Tests for explain_counterfactual() — Article 86 counterfactual explanations."""

    @pytest.fixture
    def counterfactual_mem(self):
        """Graph with tensions for counterfactual analysis."""
        mem = Memory()
        low_cost = mem.thought("low cost database option")
        high_perf = mem.thought("high performance database option")
        mem.tension(low_cost, high_perf, "cost vs performance")
        decision = mem.thought("selected low cost database")
        low_cost.causes(decision)
        return mem, decision

    def test_general_mode(self, counterfactual_mem):
        """General mode produces readable output with factors."""
        mem, decision = counterfactual_mem
        result = mem.query.counterfactual(decision.id)
        text = explain_counterfactual(result)
        assert "pivotal factor" in text.lower()
        assert "cost vs performance" in text

    def test_legal_mode(self, counterfactual_mem):
        """Legal mode includes CJEU citation and certification."""
        mem, decision = counterfactual_mem
        result = mem.query.counterfactual(decision.id)
        text = explain_counterfactual(result, audience="legal")
        assert "CJEU" in text or "C-203/22" in text
        assert "CERTIFICATION" in text
        assert "No large language model" in text
        assert "deterministically" in text

    def test_legal_mode_with_subject(self, counterfactual_mem):
        """Subject label appears in legal output."""
        mem, decision = counterfactual_mem
        result = mem.query.counterfactual(decision.id)
        text = explain_counterfactual(result, audience="legal", subject="Applicant #42")
        assert "Applicant #42" in text

    def test_empty_factors(self):
        """No factors produces appropriate message."""
        mem = Memory()
        n = mem.thought("simple decision")
        result = mem.query.counterfactual(n.id)
        text = explain_counterfactual(result)
        assert "no" in text.lower() or "without" in text.lower()

    def test_empty_factors_legal(self):
        """No factors in legal mode produces appropriate output."""
        mem = Memory()
        n = mem.thought("simple decision")
        result = mem.query.counterfactual(n.id)
        text = explain_counterfactual(result, audience="legal")
        assert "no counterfactual factors" in text.lower()

    def test_depth_in_summary(self, counterfactual_mem):
        """Summary includes depth information."""
        mem, decision = counterfactual_mem
        result = mem.query.counterfactual(decision.id)
        text = explain_counterfactual(result)
        assert "level" in text.lower()
