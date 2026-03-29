"""
Article 86 Explanation Generator — EU AI Act compliance module.

Converts structured why() query results into plain-language explanations
for non-technical individuals, satisfying EU AI Act Article 86
(Right to Explanation for automated decision-making).

The EU AI Act requires that affected individuals receive "clear and meaningful
explanations" of decisions made by AI systems. FlowScript's why() query returns
a typed causal graph — this module converts that graph into human-readable
prose suitable for compliance submissions, affected individual notifications,
and regulatory audit packages.

Key properties:
  - Deterministic: same input always produces the same output
  - No LLM dependency: runs offline, suitable for compliance artifacts
  - Multiple audience modes: general, legal, technical
  - Supports all why() return formats: chain, minimal, tree
  - Counterfactual explanations via explain_counterfactual() (CJEU C-203/22)

Usage:
    from flowscript_agents import Memory
    from flowscript_agents.explain import explain

    mem = Memory()
    cause = mem.statement("credit score below threshold")
    effect = mem.statement("loan application rejected")
    cause.causes(effect)

    result = mem.query.why(effect.id)

    # Plain English for affected individual
    print(explain(result, audience="general"))

    # Formal compliance language for regulatory submissions
    print(explain(result, audience="legal"))
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Union

from .query import (
    CausalAncestry, CausalTree, CausalTreeNode, CounterfactualResult, MinimalWhy,
)


# Public type alias — all why() return types
WhyResult = Union[CausalAncestry, CausalTree, MinimalWhy]

# Audience modes
AUDIENCE_GENERAL = "general"
AUDIENCE_LEGAL = "legal"
AUDIENCE_TECHNICAL = "technical"


def explain(
    why_result: WhyResult,
    subject: str | None = None,
    audience: str = AUDIENCE_GENERAL,
) -> str:
    """Convert a why() query result into a human-readable explanation.

    Args:
        why_result: Output from ``Memory.query.why()``. Accepts chain
            (``CausalAncestry``), minimal (``MinimalWhy``), or tree
            (``CausalTree``) formats.
        subject: Optional label for the entity or person affected by the
            decision (e.g. "Applicant ID #4821"). Included in legal-mode
            headers. Omitted from general-mode output.
        audience: One of:
            - ``"general"`` (default): plain English for non-technical
              individuals. Satisfies Article 86 clarity requirement.
            - ``"legal"``: formal compliance language for regulatory
              submissions and audit packages.
            - ``"technical"``: structured representation, equivalent to
              str(why_result). Useful for developer debugging.

    Returns:
        A human-readable string explanation.

    Note:
        ``MinimalWhy`` (format="minimal") contains only the causal chain
        (ancestors), not the target decision. For full Article 86 compliance
        — where the affected person must know WHAT decision was made —
        use format="chain" (``CausalAncestry``) or format="tree"
        (``CausalTree``).

    Raises:
        TypeError: If why_result is not a recognised why() return type.
        ValueError: If audience is not one of the valid options.

    Examples:
        >>> mem = Memory()
        >>> cause = mem.statement("income below threshold")
        >>> effect = mem.statement("loan rejected")
        >>> cause.causes(effect)
        >>> result = mem.query.why(effect.id)
        >>> print(explain(result))
        Decision Explanation
        ...
    """
    if audience not in (AUDIENCE_GENERAL, AUDIENCE_LEGAL, AUDIENCE_TECHNICAL):
        raise ValueError(
            f"audience must be 'general', 'legal', or 'technical', got {audience!r}"
        )

    if audience == AUDIENCE_TECHNICAL:
        return str(why_result)

    if isinstance(why_result, CausalAncestry):
        return _explain_ancestry(why_result, subject=subject, audience=audience)
    if isinstance(why_result, MinimalWhy):
        return _explain_minimal(why_result, subject=subject, audience=audience)
    if isinstance(why_result, CausalTree):
        return _explain_tree(why_result, subject=subject, audience=audience)

    raise TypeError(
        f"why_result must be CausalAncestry, MinimalWhy, or CausalTree, "
        f"got {type(why_result).__name__!r}"
    )


# =============================================================================
# CausalAncestry (format="chain" — the default)
# =============================================================================


def _explain_ancestry(
    result: CausalAncestry,
    *,
    subject: str | None,
    audience: str,
) -> str:
    target = result.target["content"]
    chain = result.causal_chain  # root at index 0, highest depth
    root = result.root_cause["content"]
    depth = result.metadata.get("max_depth", len(chain))
    multiple_paths = result.metadata.get("has_multiple_paths", False)

    if audience == AUDIENCE_GENERAL:
        return _ancestry_general(target, chain, root, depth, multiple_paths)
    else:
        return _ancestry_legal(target, chain, root, depth, multiple_paths, subject)


def _ancestry_general(
    target: str,
    chain: list,
    root: str,
    depth: int,
    multiple_paths: bool,
) -> str:
    lines = ["Decision Explanation", ""]
    lines.append(f'The outcome "{target}" can be explained as follows.')
    lines.append("")

    if not chain:
        lines.append(
            f'This outcome has no recorded causal history. '
            f'It was entered directly without a traced reasoning path.'
        )
        return "\n".join(lines)

    if depth == 1:
        lines.append(f'Direct cause: {root}')
        lines.append("")
        lines.append(
            f'In plain terms: "{target}" occurred because of "{root}".'
        )
    else:
        lines.append(f"Reasoning chain ({depth} step{'s' if depth != 1 else ''}):")
        lines.append("")
        for i, node in enumerate(chain):
            step_num = i + 1
            content = node.content
            rel = _humanize_relationship(node.relationship_type)
            if i == 0:
                lines.append(f"  {step_num}. Starting point: {content}")
            elif i == len(chain) - 1:
                lines.append(f"  {step_num}. {rel}: {content}")
                lines.append(f"     → Final outcome: {target}")
            else:
                lines.append(f"  {step_num}. {rel}: {content}")

        lines.append("")
        lines.append(
            f'Summary: The fundamental starting point was "{root}", which through '
            f'{depth} step{"s" if depth != 1 else ""} of reasoning led to '
            f'"{target}".'
        )

    if multiple_paths:
        lines.append("")
        lines.append(
            "Note: Multiple causal paths contributed to this outcome. "
            "The explanation above shows the primary reasoning chain."
        )

    return "\n".join(lines)


def _ancestry_legal(
    target: str,
    chain: list,
    root: str,
    depth: int,
    multiple_paths: bool,
    subject: str | None,
) -> str:
    lines = ["AUTOMATED DECISION EXPLANATION"]
    lines.append("Issued under EU AI Act Article 86 (Right to Explanation)")
    lines.append("")

    if subject:
        lines.append(f"Subject: {subject}")
    lines.append(f"Decision: {target}")
    lines.append(f"Causal chain depth: {depth} step{'s' if depth != 1 else ''}")
    lines.append(
        f"Multiple causal paths: {'Yes' if multiple_paths else 'No'}"
    )
    lines.append("")
    lines.append("CAUSAL SEQUENCE")
    lines.append("")

    if not chain:
        lines.append("  [No causal history recorded for this decision.]")
    else:
        for i, node in enumerate(chain):
            step_num = i + 1
            content = node.content
            rel = node.relationship_type or "derives_from"
            if i == 0:
                lines.append(f"  Step {step_num} (foundational factor): {content}")
            else:
                lines.append(
                    f"  Step {step_num} ({rel.replace('_', ' ')}): {content}"
                )

        lines.append(f"  Outcome: {target}")

    lines.append("")
    lines.append("FOUNDATIONAL FACTOR")
    lines.append("")
    lines.append(f"  {root}")
    lines.append("")
    lines.append("CERTIFICATION")
    lines.append("")
    lines.append(
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )
    lines.append(
        "This explanation is generated from a deterministic causal reasoning "
        "graph maintained by FlowScript. The causal chain accurately reflects "
        "the reasoning recorded at the time of the decision. The complete "
        "hash-chained audit trail is available upon request and can be "
        "verified against the original reasoning record."
    )

    if multiple_paths:
        lines.append("")
        lines.append(
            "Note: Multiple causal paths contributed to this outcome. This "
            "explanation represents the primary reasoning chain. A complete "
            "causal tree is available in the full audit package."
        )

    return "\n".join(lines)


# =============================================================================
# MinimalWhy (format="minimal")
# =============================================================================


def _explain_minimal(
    result: MinimalWhy,
    *,
    subject: str | None,
    audience: str,
) -> str:
    root = result.root_cause
    chain = result.chain  # list of strings, root→target direction

    if audience == AUDIENCE_GENERAL:
        lines = ["Decision Explanation", ""]
        if not chain or (len(chain) == 1 and chain[0] == root):
            lines.append(f'This outcome traces back to: "{root}".')
        else:
            arrow = " \u2192 "
            lines.append("Reasoning path:")
            lines.append("")
            lines.append(f"  {arrow.join(chain)}")
            lines.append("")
            lines.append(f'Starting point: "{root}".')
        return "\n".join(lines)

    else:  # legal
        lines = ["AUTOMATED DECISION EXPLANATION"]
        lines.append("Issued under EU AI Act Article 86 (Right to Explanation)")
        lines.append("")
        if subject:
            lines.append(f"Subject: {subject}")
        lines.append(f"Foundational factor: {root}")
        lines.append("")
        if chain:
            lines.append("REASONING PATH")
            lines.append("")
            for i, step in enumerate(chain):
                lines.append(f"  Step {i + 1}: {step}")
        lines.append("")
        lines.append(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )
        lines.append(
            "This explanation is generated from a deterministic causal reasoning "
            "graph. The complete audit trail is available upon request."
        )
        return "\n".join(lines)


# =============================================================================
# CausalTree (format="tree")
# =============================================================================


def _explain_tree(
    result: CausalTree,
    *,
    subject: str | None,
    audience: str,
) -> str:
    target = result.target["content"]
    total_ancestors = result.metadata.get("total_ancestors", 0)

    if audience == AUDIENCE_GENERAL:
        lines = ["Decision Explanation", ""]
        lines.append(
            f'The outcome "{target}" has {total_ancestors} contributing '
            f'factor{"s" if total_ancestors != 1 else ""}.'
        )
        lines.append("")
        lines.append("Contributing factors (closest to most distant):")
        lines.append("")
        _render_tree_general(result.tree, lines, indent=2)
        lines.append("")
        lines.append(f'Outcome: {target}')
        return "\n".join(lines)

    else:  # legal
        lines = ["AUTOMATED DECISION EXPLANATION"]
        lines.append("Issued under EU AI Act Article 86 (Right to Explanation)")
        lines.append("")
        if subject:
            lines.append(f"Subject: {subject}")
        lines.append(f"Decision: {target}")
        lines.append(f"Total contributing factors: {total_ancestors}")
        lines.append("")
        lines.append("CAUSAL TREE")
        lines.append("")
        _render_tree_general(result.tree, lines, indent=2)
        lines.append("")
        lines.append(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )
        lines.append(
            "This explanation is generated from a deterministic causal reasoning "
            "graph. The complete audit trail is available upon request."
        )
        return "\n".join(lines)


def _render_tree_general(
    node: CausalTreeNode,
    lines: list[str],
    indent: int,
    depth: int = 0,
) -> None:
    """Recursively render a CausalTreeNode into the lines list."""
    prefix = " " * indent
    rel = (
        _humanize_relationship(node.relationship_type)
        if node.relationship_type
        else "caused by"
    )
    if depth == 0:
        lines.append(f"{prefix}• {node.content}")
    else:
        lines.append(f"{prefix}{'  ' * (depth - 1)}  ↑ {rel}: {node.content}")

    for parent in node.parents:
        _render_tree_general(parent, lines, indent=indent, depth=depth + 1)


# =============================================================================
# Helpers
# =============================================================================


def _humanize_relationship(rel_type: str | None) -> str:
    """Convert internal relationship type names to plain English."""
    mapping = {
        "derives_from": "which follows from",
        "causes": "which was caused by",
        "equivalent": "which is equivalent to",
        "alternative": "as an alternative",
        "blocks": "which was blocked by",
        "tension": "in tension with",
    }
    if rel_type is None:
        return "which follows from"
    return mapping.get(rel_type, f"({rel_type})")


# =============================================================================
# Counterfactual Explanation (CJEU C-203/22)
# =============================================================================


def explain_counterfactual(
    result: CounterfactualResult,
    subject: str | None = None,
    audience: str = AUDIENCE_GENERAL,
) -> str:
    """Convert a counterfactual() result into a human-readable explanation.

    Satisfies CJEU C-203/22: not just "why this" but "why not that" — what
    conditions, if different, would lead to a different outcome.

    Args:
        result: Output from ``Memory.query.counterfactual()``.
        subject: Optional label for the affected person (e.g. "Elena Petrova").
        audience: "general", "legal", or "technical".

    Returns:
        Human-readable counterfactual explanation.
    """
    if audience not in (AUDIENCE_GENERAL, AUDIENCE_LEGAL, AUDIENCE_TECHNICAL):
        raise ValueError(
            f"audience must be 'general', 'legal', or 'technical', got {audience!r}"
        )

    if audience == AUDIENCE_TECHNICAL:
        return str(result)

    decision = result.decision["content"]
    factors = result.factors

    if audience == AUDIENCE_LEGAL:
        return _counterfactual_legal(decision, factors, subject)
    return _counterfactual_general(decision, factors)


def _counterfactual_general(
    decision: str,
    factors: list,
) -> str:
    lines = ["Counterfactual Explanation", ""]
    lines.append(f'The decision "{decision}" was reached based on specific conditions.')
    lines.append(
        "The following factors were pivotal — if any had been different, "
        "the outcome could have changed:"
    )
    lines.append("")

    if not factors:
        lines.append(
            "No pivotal factors were identified in the reasoning chain. "
            "The decision may have been reached without competing considerations."
        )
        return "\n".join(lines)

    for i, f in enumerate(factors, 1):
        lines.append(f"  {i}. {f.factor['content']}")
        lines.append(f"     Tension: {f.tension_axis}")
        lines.append(f"     If instead: {f.counterfactual_condition['content']}")
        lines.append("")

    lines.append(
        f"In total, {len(factors)} pivotal factor{'s were' if len(factors) != 1 else ' was'} "
        f"identified across {factors[0].depth if factors else 0} levels of reasoning."
    )
    return "\n".join(lines)


def _counterfactual_legal(
    decision: str,
    factors: list,
    subject: str | None,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = ["COUNTERFACTUAL ANALYSIS — AUTOMATED DECISION", ""]

    if subject:
        lines.append(f"Subject: {subject}")
    lines.append(f"Decision under analysis: {decision}")
    lines.append(f"Generated: {now}")
    lines.append(
        "Basis: CJEU Case C-203/22 (counterfactual explanation requirement)"
    )
    lines.append("")

    lines.append("PIVOTAL FACTORS")
    lines.append("")

    if not factors:
        lines.append(
            "No counterfactual factors identified. The decision chain contains "
            "no recorded competing considerations that would alter the outcome."
        )
        return "\n".join(lines)

    lines.append(
        "The following factors were identified as pivotal to the decision. "
        "Each represents a point where different conditions would have led "
        "to a different outcome:"
    )
    lines.append("")

    for i, f in enumerate(factors, 1):
        lines.append(f"  Factor {i} (reasoning depth {f.depth}):")
        lines.append(f"    Determining condition: {f.factor['content']}")
        lines.append(f"    Dimension of competition: {f.tension_axis}")
        lines.append(f"    Counterfactual condition: {f.counterfactual_condition['content']}")
        lines.append(
            f"    Implication: Had the counterfactual condition prevailed over "
            f"the determining condition on the axis of \"{f.tension_axis}\", "
            f"the decision would likely differ."
        )
        lines.append("")

    lines.append("CERTIFICATION")
    lines.append("")
    lines.append(
        f"This counterfactual analysis was computed deterministically from the "
        f"agent's recorded reasoning chain. {len(factors)} pivotal "
        f"factor{'s were' if len(factors) != 1 else ' was'} identified across "
        f"{max(f.depth for f in factors)} levels of causal reasoning. "
        f"No large language model was used in generating this explanation."
    )

    return "\n".join(lines)
