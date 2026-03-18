"""
FlowScript Query Engine.

Inlined from flowscript-ldp v0.2.1. Five computational operations
on IR graphs: why, what_if, tensions, blocked, alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .types import IR, Node, Relationship, RelationType, State


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class CausalChainNode:
    depth: int
    id: str
    content: str
    relationship_type: str

    def __repr__(self) -> str:
        return f"CausalChainNode(depth={self.depth}, content={self.content!r})"


@dataclass
class CausalAncestry:
    target: dict[str, str]
    causal_chain: list[CausalChainNode]
    root_cause: dict[str, Any]
    metadata: dict[str, Any]

    def __repr__(self) -> str:
        chain_str = " \u2192 ".join(n.content for n in self.causal_chain)
        if chain_str:
            chain_str += f" \u2192 {self.target['content']}"
        else:
            chain_str = self.target["content"]
        return f"CausalAncestry({chain_str})"


@dataclass
class CausalTreeNode:
    """Recursive tree showing all causal paths to a node."""

    id: str
    content: str
    relationship_type: Optional[str] = None
    parents: list[CausalTreeNode] = field(default_factory=list)

    def __repr__(self) -> str:
        n = len(self.parents)
        return f"CausalTreeNode({self.content!r}, {n} parent{'s' if n != 1 else ''})"


@dataclass
class CausalTree:
    """Tree format for why() -- shows all causal paths, not just one chain."""

    target: dict[str, str]
    tree: CausalTreeNode
    metadata: dict[str, Any]

    def __repr__(self) -> str:
        return f"CausalTree(target={self.target['content']!r}, ancestors={self.metadata['total_ancestors']})"


@dataclass
class MinimalWhy:
    root_cause: str
    chain: list[str]

    def __repr__(self) -> str:
        chain_str = " \u2192 ".join(self.chain) if self.chain else self.root_cause
        return f"MinimalWhy({chain_str})"


@dataclass
class ImpactConsequence:
    id: str
    content: str
    relationship: str
    depth: int
    has_tension: bool = False
    tension_axis: Optional[str] = None

    def __repr__(self) -> str:
        t = f" \u26a1{self.tension_axis}" if self.tension_axis else ""
        return f"ImpactConsequence(d{self.depth}: {self.content!r}{t})"


@dataclass
class TensionInfo:
    axis: str
    source: dict[str, str]
    target: dict[str, str]

    def __repr__(self) -> str:
        return f"TensionInfo({self.source['content']!r} vs {self.target['content']!r} [{self.axis}])"


@dataclass
class ImpactAnalysis:
    source: dict[str, str]
    impact_tree: dict[str, list[ImpactConsequence]]
    tensions_in_impact_zone: list[TensionInfo]
    metadata: dict[str, Any]

    def __repr__(self) -> str:
        d = len(self.impact_tree.get("direct_consequences", []))
        i = len(self.impact_tree.get("indirect_consequences", []))
        t = len(self.tensions_in_impact_zone)
        return f"ImpactAnalysis({self.source['content']!r}: {d} direct, {i} indirect, {t} tensions)"


@dataclass
class ImpactList:
    """Flat list format for what_if() -- all consequences sorted by depth."""

    source: dict[str, str]
    consequences: list[ImpactConsequence]
    tensions_in_impact_zone: list[TensionInfo]
    metadata: dict[str, Any]

    def __repr__(self) -> str:
        n = len(self.consequences)
        return f"ImpactList({self.source['content']!r}: {n} consequence{'s' if n != 1 else ''})"


@dataclass
class ImpactSummary:
    impact_summary: str
    benefits: list[str]
    risks: list[str]
    key_tradeoff: Optional[str]

    def __repr__(self) -> str:
        return f"ImpactSummary({self.impact_summary})"


@dataclass
class TensionDetail:
    source: dict[str, str]
    target: dict[str, str]
    context: Optional[list[dict[str, str]]] = None

    def __repr__(self) -> str:
        return f"TensionDetail({self.source['content']!r} vs {self.target['content']!r})"


@dataclass
class TensionsResult:
    tensions_by_axis: Optional[dict[str, list[TensionDetail]]] = None
    tensions_by_node: Optional[dict[str, list[TensionDetail]]] = None
    tensions: Optional[list[TensionDetail]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        n = self.metadata.get("total_tensions", 0)
        axes = self.metadata.get("unique_axes", [])
        return f"TensionsResult({n} tension{'s' if n != 1 else ''}, axes={axes})"


@dataclass
class BlockerDetail:
    node: dict[str, str]
    blocked_state: dict[str, Any]
    impact_score: int = 0
    transitive_causes: Optional[list[dict[str, str]]] = None
    transitive_effects: Optional[list[dict[str, str]]] = None

    def __repr__(self) -> str:
        days = self.blocked_state.get("days_blocked", "?")
        return f"BlockerDetail({self.node['content']!r}, {days}d, impact={self.impact_score})"


@dataclass
class BlockedResult:
    blockers: list[BlockerDetail]
    metadata: dict[str, Any]

    def __repr__(self) -> str:
        n = self.metadata.get("total_blockers", 0)
        return f"BlockedResult({n} blocker{'s' if n != 1 else ''})"


@dataclass
class AlternativeDetail:
    id: str
    content: str
    chosen: bool
    rationale: Optional[str] = None
    decided_on: Optional[str] = None
    rejection_reasons: Optional[list[str]] = None
    consequences: Optional[list[dict[str, str]]] = None
    tensions: Optional[list[TensionInfo]] = None

    def __repr__(self) -> str:
        mark = "\u2713" if self.chosen else "\u25cb"
        return f"AlternativeDetail({mark} {self.content!r})"


@dataclass
class AlternativesResultComparison:
    format: str
    question: dict[str, str]
    alternatives: list[AlternativeDetail]
    decision_summary: dict[str, Any]

    def __repr__(self) -> str:
        chosen = self.decision_summary.get("chosen", "none")
        n = len(self.alternatives)
        return f"AlternativesResult(comparison: {n} options, chosen={chosen!r})"


@dataclass
class AlternativesResultSimple:
    format: str
    question: str
    options_considered: list[str]
    chosen: Optional[str]
    reason: Optional[str]

    def __repr__(self) -> str:
        return f"AlternativesResult(simple: chosen={self.chosen!r})"


@dataclass
class TreeAlternative:
    id: str
    content: str
    chosen: bool
    children: list[TreeAlternative] = field(default_factory=list)
    rejection_reasons: Optional[list[str]] = None

    def __repr__(self) -> str:
        mark = "\u2713" if self.chosen else "\u25cb"
        kids = f", {len(self.children)} children" if self.children else ""
        return f"TreeAlternative({mark} {self.content!r}{kids})"


@dataclass
class AlternativesResultTree:
    format: str
    question: dict[str, str]
    alternatives: list[TreeAlternative]

    def __repr__(self) -> str:
        return f"AlternativesResult(tree: {len(self.alternatives)} alternatives)"


AlternativesResult = (
    AlternativesResultComparison | AlternativesResultTree | AlternativesResultSimple
)


# Internal traversal result
@dataclass
class _TraversalNode:
    node: Node
    depth: int
    relationship_type: Optional[str] = None


# =============================================================================
# Query Engine
# =============================================================================


class QueryEngine:
    """FlowScript Query Engine -- computational operations on IR graphs."""

    def __init__(self, ir: Optional[IR] = None):
        self._ir: Optional[IR] = None
        self._node_map: dict[str, Node] = {}
        self._rels_from_source: dict[str, list[Relationship]] = {}
        self._rels_to_target: dict[str, list[Relationship]] = {}
        self._states_by_node: dict[str, list[State]] = {}

        if ir is not None:
            self.load(ir)

    def load(self, ir: IR) -> None:
        """Load IR graph and build indexes for O(1) lookups."""
        self._ir = ir
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build efficient indexes for querying."""
        assert self._ir is not None

        self._node_map.clear()
        self._rels_from_source.clear()
        self._rels_to_target.clear()
        self._states_by_node.clear()

        for node in self._ir.nodes:
            self._node_map[node.id] = node

        for rel in self._ir.relationships:
            self._rels_from_source.setdefault(rel.source, []).append(rel)
            self._rels_to_target.setdefault(rel.target, []).append(rel)

        for state in self._ir.states:
            self._states_by_node.setdefault(state.node_id, []).append(state)

    # =========================================================================
    # Query 1: why -- Causal ancestry (backward traversal)
    # =========================================================================

    def why(
        self,
        node_id: str,
        *,
        max_depth: Optional[int] = None,
        include_correlations: bool = False,
        format: str = "chain",
    ) -> CausalAncestry | CausalTree | MinimalWhy:
        rel_types = [RelationType.DERIVES_FROM, RelationType.CAUSES]
        if include_correlations:
            rel_types.append(RelationType.EQUIVALENT)

        ancestors = self._traverse_backward(node_id, rel_types, max_depth)

        target_node = self._node_map.get(node_id)
        if target_node is None:
            raise ValueError(f"Node not found: {node_id}")

        if format == "minimal":
            chain, root_cause = self._build_causal_chain(
                node_id, ancestors, rel_types
            )
            return MinimalWhy(
                root_cause=root_cause.content,
                chain=[n.node.content for n in chain],
            )

        if format == "tree":
            tree = self._build_causal_tree(node_id, rel_types, max_depth)
            return CausalTree(
                target={"id": target_node.id, "content": target_node.content},
                tree=tree,
                metadata={
                    "total_ancestors": len(ancestors),
                    "max_depth": max((a.depth for a in ancestors), default=0),
                    "has_multiple_paths": self._has_multiple_paths(
                        node_id, rel_types
                    ),
                },
            )

        chain, root_cause = self._build_causal_chain(
            node_id, ancestors, rel_types
        )
        return CausalAncestry(
            target={"id": target_node.id, "content": target_node.content},
            causal_chain=[
                CausalChainNode(
                    depth=len(chain) - i,
                    id=n.node.id,
                    content=n.node.content,
                    relationship_type=n.relationship_type or "derives_from",
                )
                for i, n in enumerate(chain)
            ],
            root_cause={
                "id": root_cause.id,
                "content": root_cause.content,
                "is_root": True,
            },
            metadata={
                "total_ancestors": len(ancestors),
                "max_depth": len(chain),
                "has_multiple_paths": self._has_multiple_paths(node_id, rel_types),
            },
        )

    # =========================================================================
    # Query 2: what_if -- Impact analysis (forward traversal)
    # =========================================================================

    def what_if(
        self,
        node_id: str,
        *,
        max_depth: Optional[int] = None,
        include_correlations: bool = False,
        include_temporal: bool = True,
        format: str = "tree",
    ) -> ImpactAnalysis | ImpactList | ImpactSummary:
        rel_types = [RelationType.CAUSES]
        if include_temporal:
            rel_types.append(RelationType.TEMPORAL)
        if include_correlations:
            rel_types.append(RelationType.EQUIVALENT)

        source_node = self._node_map.get(node_id)
        if source_node is None:
            raise ValueError(f"Node not found: {node_id}")

        descendants = self._traverse_forward(node_id, rel_types, max_depth)

        desc_ids = {d.node.id for d in descendants}
        desc_ids.add(node_id)
        tensions = self._find_tensions_in_subgraph(desc_ids)

        has_temporal = any(
            d.relationship_type == RelationType.TEMPORAL.value for d in descendants
        )

        metadata = {
            "total_descendants": len(descendants),
            "max_depth": max((d.depth for d in descendants), default=0),
            "tension_count": len(tensions),
            "has_temporal_consequences": has_temporal,
        }

        if format == "summary":
            return self._build_impact_summary(source_node, descendants, tensions)

        if format == "list":
            impact_tree = self._build_impact_tree(descendants)
            all_consequences = (
                impact_tree["direct"] + impact_tree["indirect"]
            )
            all_consequences.sort(key=lambda c: c.depth)
            return ImpactList(
                source={"id": source_node.id, "content": source_node.content},
                consequences=all_consequences,
                tensions_in_impact_zone=tensions,
                metadata=metadata,
            )

        impact_tree = self._build_impact_tree(descendants)
        return ImpactAnalysis(
            source={"id": source_node.id, "content": source_node.content},
            impact_tree={
                "direct_consequences": impact_tree["direct"],
                "indirect_consequences": impact_tree["indirect"],
            },
            tensions_in_impact_zone=tensions,
            metadata=metadata,
        )

    # =========================================================================
    # Query 3: tensions -- Tradeoff mapping
    # =========================================================================

    def tensions(
        self,
        *,
        group_by: str = "axis",
        filter_by_axis: Optional[list[str]] = None,
        include_context: bool = False,
        scope: Optional[str] = None,
    ) -> TensionsResult:
        assert self._ir is not None

        tension_rels = [
            r for r in self._ir.relationships if r.type == RelationType.TENSION
        ]

        if scope is not None:
            scope_ids: set[str] = {scope}
            descs = self._traverse_forward(
                scope,
                [RelationType.CAUSES, RelationType.TEMPORAL, RelationType.DERIVES_FROM],
            )
            scope_ids.update(d.node.id for d in descs)
            tension_rels = [
                r
                for r in tension_rels
                if r.source in scope_ids and r.target in scope_ids
            ]

        if filter_by_axis:
            tension_rels = [
                r
                for r in tension_rels
                if r.axis_label and r.axis_label in filter_by_axis
            ]

        details: list[tuple[str, TensionDetail]] = []
        for rel in tension_rels:
            src = self._node_map.get(rel.source)
            tgt = self._node_map.get(rel.target)
            if not src or not tgt:
                continue

            axis = rel.axis_label or "unlabeled"
            detail = TensionDetail(
                source={"id": src.id, "content": src.content},
                target={"id": tgt.id, "content": tgt.content},
            )

            if include_context:
                context = []
                for parent_rel in self._rels_to_target.get(rel.source, []):
                    if parent_rel.type != RelationType.TENSION:
                        parent = self._node_map.get(parent_rel.source)
                        if parent:
                            context.append(
                                {"id": parent.id, "content": parent.content}
                            )
                if context:
                    detail.context = context

            details.append((axis, detail))

        unique_axes = sorted(set(a for a, _ in details))
        axis_counts: dict[str, int] = {}
        for a, _ in details:
            axis_counts[a] = axis_counts.get(a, 0) + 1

        most_common = max(axis_counts, key=axis_counts.get, default=None) if axis_counts else None  # type: ignore[arg-type]

        metadata = {
            "total_tensions": len(details),
            "unique_axes": unique_axes,
            "most_common_axis": most_common,
        }

        if group_by == "axis":
            by_axis: dict[str, list[TensionDetail]] = {}
            for axis, detail in details:
                by_axis.setdefault(axis, []).append(detail)
            return TensionsResult(tensions_by_axis=by_axis, metadata=metadata)
        elif group_by == "node":
            by_node: dict[str, list[TensionDetail]] = {}
            for _, detail in details:
                nid = detail.source["id"]
                by_node.setdefault(nid, []).append(detail)
            return TensionsResult(tensions_by_node=by_node, metadata=metadata)
        else:
            return TensionsResult(
                tensions=[d for _, d in details], metadata=metadata
            )

    # =========================================================================
    # Query 4: blocked -- Blocker tracking
    # =========================================================================

    def blocked(
        self,
        *,
        since: Optional[str] = None,
        include_transitive_causes: bool = True,
        include_transitive_effects: bool = True,
        format: str = "detailed",
    ) -> BlockedResult:
        assert self._ir is not None

        blocked_states = [s for s in self._ir.states if s.type.value == "blocked"]

        if since:
            blocked_states = [
                s
                for s in blocked_states
                if s.fields and s.fields.since and s.fields.since >= since
            ]

        blockers: list[BlockerDetail] = []
        now = datetime.now(tz=timezone.utc)

        for state in blocked_states:
            node = self._node_map.get(state.node_id)
            if not node:
                continue

            reason = state.fields.reason if state.fields else "unknown"
            since_date = state.fields.since if state.fields else ""

            days_blocked = 0
            if since_date:
                try:
                    since_dt = datetime.fromisoformat(since_date.replace("Z", "+00:00"))
                    days_blocked = (now - since_dt).days
                except (ValueError, TypeError):
                    pass

            detail = BlockerDetail(
                node={"id": node.id, "content": node.content},
                blocked_state={
                    "reason": reason or "unknown",
                    "since": since_date or "",
                    "days_blocked": days_blocked,
                },
            )

            if include_transitive_causes:
                causes = self._traverse_backward(
                    node.id, [RelationType.DERIVES_FROM, RelationType.CAUSES]
                )
                detail.transitive_causes = [
                    {"id": c.node.id, "content": c.node.content} for c in causes
                ]

            if include_transitive_effects:
                effects = self._traverse_forward(
                    node.id, [RelationType.CAUSES, RelationType.TEMPORAL]
                )
                detail.transitive_effects = [
                    {"id": e.node.id, "content": e.node.content} for e in effects
                ]
                detail.impact_score = len(effects)

            blockers.append(detail)

        blockers.sort(
            key=lambda b: (b.impact_score, b.blocked_state["days_blocked"]),
            reverse=True,
        )

        total = len(blockers)
        high_priority = sum(
            1
            for b in blockers
            if b.impact_score > 0 or b.blocked_state["days_blocked"] > 7
        )
        avg_days = (
            sum(b.blocked_state["days_blocked"] for b in blockers) / total
            if total
            else 0
        )
        oldest = None
        if blockers:
            o = max(blockers, key=lambda b: b.blocked_state["days_blocked"])
            oldest = {"id": o.node["id"], "days": o.blocked_state["days_blocked"]}

        return BlockedResult(
            blockers=blockers,
            metadata={
                "total_blockers": total,
                "high_priority_count": high_priority,
                "average_days_blocked": round(avg_days, 1),
                "oldest_blocker": oldest,
            },
        )

    # =========================================================================
    # Query 5: alternatives -- Decision reconstruction
    # =========================================================================

    def alternatives(
        self,
        question_id: str,
        *,
        include_rationale: bool = True,
        include_consequences: bool = False,
        show_rejected_reasons: bool = False,
        format: str = "comparison",
    ) -> AlternativesResult:
        assert self._ir is not None

        question = self._node_map.get(question_id)
        if question is None:
            raise ValueError(f"Node not found: {question_id}")
        if question.type.value != "question":
            raise ValueError(
                f"Node {question_id} is not a question (type: {question.type.value})"
            )

        alt_rels = [
            r
            for r in self._rels_from_source.get(question_id, [])
            if r.type == RelationType.ALTERNATIVE
        ]

        if format == "simple":
            alts: list[dict[str, Any]] = []
            for rel in alt_rels:
                alt_node = self._node_map.get(rel.target)
                if not alt_node:
                    continue
                is_chosen, rationale, _ = self._check_decided(alt_node)
                alts.append(
                    {
                        "content": alt_node.content,
                        "chosen": is_chosen,
                        "rationale": rationale if include_rationale else None,
                    }
                )
            chosen_alt = next((a for a in alts if a["chosen"]), None)
            return AlternativesResultSimple(
                format="simple",
                question=question.content,
                options_considered=[a["content"] for a in alts],
                chosen=chosen_alt["content"] if chosen_alt else None,
                reason=chosen_alt["rationale"] if chosen_alt else None,
            )

        if format == "tree":
            tree_alts = [
                self._build_alternative_tree(
                    rel.target, set(), show_rejected_reasons
                )
                for rel in alt_rels
            ]
            return AlternativesResultTree(
                format="tree",
                question={"id": question.id, "content": question.content},
                alternatives=tree_alts,
            )

        alternatives_list: list[AlternativeDetail] = []
        chosen_detail: Optional[AlternativeDetail] = None

        for rel in alt_rels:
            alt_node = self._node_map.get(rel.target)
            if not alt_node:
                continue

            is_chosen, rationale, decided_on = self._check_decided(alt_node)

            detail = AlternativeDetail(
                id=alt_node.id,
                content=alt_node.content,
                chosen=is_chosen,
            )

            if is_chosen and rationale and include_rationale:
                detail.rationale = rationale
                detail.decided_on = decided_on

            if show_rejected_reasons and not is_chosen:
                reasons = self._extract_rejection_reasons(alt_node.id)
                if reasons:
                    detail.rejection_reasons = reasons

            if include_consequences:
                consequences = [
                    {"id": n.id, "content": n.content}
                    for r in self._rels_from_source.get(alt_node.id, [])
                    if r.type == RelationType.CAUSES
                    for n in [self._node_map.get(r.target)]
                    if n is not None
                ]
                if consequences:
                    detail.consequences = consequences

            alt_tensions = [
                TensionInfo(
                    axis=r.axis_label or "unlabeled",
                    source={"id": alt_node.id, "content": alt_node.content},
                    target={
                        "id": t.id,
                        "content": t.content,
                    },
                )
                for r in self._rels_from_source.get(alt_node.id, [])
                if r.type == RelationType.TENSION
                for t in [self._node_map.get(r.target)]
                if t is not None
            ]
            if alt_tensions:
                detail.tensions = alt_tensions

            alternatives_list.append(detail)
            if is_chosen:
                chosen_detail = detail

        rejected = [a.content for a in alternatives_list if not a.chosen]
        key_factors = list(
            set(
                t.axis
                for t in (chosen_detail.tensions if chosen_detail and chosen_detail.tensions else [])
            )
        )

        return AlternativesResultComparison(
            format="comparison",
            question={"id": question.id, "content": question.content},
            alternatives=alternatives_list,
            decision_summary={
                "chosen": chosen_detail.content if chosen_detail else None,
                "rationale": chosen_detail.rationale if chosen_detail else None,
                "rejected": rejected,
                "key_factors": key_factors,
            },
        )

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _check_decided(
        self, node: Node
    ) -> tuple[bool, Optional[str], Optional[str]]:
        for state in self._states_by_node.get(node.id, []):
            if state.type.value == "decided":
                rationale = state.fields.rationale if state.fields else None
                on = state.fields.on if state.fields else None
                return True, rationale, on
        return False, None, None

    def _extract_rejection_reasons(self, alt_node_id: str) -> list[str]:
        return [
            n.content
            for r in self._rels_from_source.get(alt_node_id, [])
            if r.type == RelationType.CAUSES
            for n in [self._node_map.get(r.target)]
            if n is not None and n.type.value == "thought"
        ]

    def _build_alternative_tree(
        self,
        node_id: str,
        visited: set[str],
        include_rejection_reasons: bool = False,
    ) -> TreeAlternative:
        if node_id in visited:
            node = self._node_map[node_id]
            return TreeAlternative(
                id=node.id,
                content=node.content + " [cycle detected]",
                chosen=False,
            )

        visited = visited | {node_id}
        node = self._node_map[node_id]

        is_chosen = any(
            s.type.value == "decided"
            for s in self._states_by_node.get(node_id, [])
        )

        tree_node = TreeAlternative(
            id=node.id,
            content=node.content,
            chosen=is_chosen,
        )

        if include_rejection_reasons and not is_chosen:
            reasons = self._extract_rejection_reasons(node_id)
            if reasons:
                tree_node.rejection_reasons = reasons

        child_rels = [
            r
            for r in self._rels_from_source.get(node_id, [])
            if r.type == RelationType.CAUSES
        ]
        for rel in child_rels:
            tree_node.children.append(
                self._build_alternative_tree(
                    rel.target, visited, include_rejection_reasons
                )
            )

        return tree_node

    def _traverse_backward(
        self,
        node_id: str,
        rel_types: list[RelationType],
        max_depth: Optional[int] = None,
        _visited: Optional[set[str]] = None,
        _depth: int = 0,
    ) -> list[_TraversalNode]:
        if max_depth is not None and _depth >= max_depth:
            return []

        visited = _visited or set()
        if node_id in visited:
            return []
        visited = visited | {node_id}

        result: list[_TraversalNode] = []
        for rel in self._rels_to_target.get(node_id, []):
            if rel.type not in rel_types:
                continue
            parent = self._node_map.get(rel.source)
            if parent:
                result.append(
                    _TraversalNode(
                        node=parent,
                        depth=_depth + 1,
                        relationship_type=rel.type.value,
                    )
                )
                result.extend(
                    self._traverse_backward(
                        rel.source, rel_types, max_depth, visited, _depth + 1
                    )
                )
        return result

    def _traverse_forward(
        self,
        node_id: str,
        rel_types: list[RelationType],
        max_depth: Optional[int] = None,
        _visited: Optional[set[str]] = None,
        _depth: int = 0,
    ) -> list[_TraversalNode]:
        if max_depth is not None and _depth >= max_depth:
            return []

        visited = _visited or set()
        if node_id in visited:
            return []
        visited = visited | {node_id}

        result: list[_TraversalNode] = []
        for rel in self._rels_from_source.get(node_id, []):
            if rel.type not in rel_types:
                continue
            child = self._node_map.get(rel.target)
            if child:
                result.append(
                    _TraversalNode(
                        node=child,
                        depth=_depth + 1,
                        relationship_type=rel.type.value,
                    )
                )
                result.extend(
                    self._traverse_forward(
                        rel.target, rel_types, max_depth, visited, _depth + 1
                    )
                )
        return result

    def _build_causal_chain(
        self,
        target_id: str,
        ancestors: list[_TraversalNode],
        rel_types: list[RelationType],
    ) -> tuple[list[_TraversalNode], Node]:
        if not ancestors:
            target_node = self._node_map[target_id]
            return [], target_node

        max_depth = max(a.depth for a in ancestors)
        roots = [a for a in ancestors if a.depth == max_depth]
        root = roots[0]

        chain: list[_TraversalNode] = [root]
        current_id = root.node.id

        ancestor_ids = {a.node.id for a in ancestors} | {target_id}

        while current_id != target_id:
            outgoing = self._rels_from_source.get(current_id, [])
            next_rel = next(
                (
                    r
                    for r in outgoing
                    if r.type in rel_types and r.target in ancestor_ids
                ),
                None,
            )
            if not next_rel:
                break

            if next_rel.target == target_id:
                break

            next_node = self._node_map.get(next_rel.target)
            if not next_node:
                break

            ancestor_match = next(
                (a for a in ancestors if a.node.id == next_rel.target), None
            )
            if not ancestor_match:
                break

            chain.append(ancestor_match)
            current_id = next_rel.target

        return chain, root.node

    def _build_causal_tree(
        self,
        node_id: str,
        rel_types: list[RelationType],
        max_depth: Optional[int] = None,
        _visited: Optional[set[str]] = None,
        _depth: int = 0,
    ) -> CausalTreeNode:
        node = self._node_map.get(node_id)
        if node is None:
            return CausalTreeNode(id=node_id, content=f"[unknown:{node_id[:8]}]")

        visited = _visited or set()
        if node_id in visited:
            return CausalTreeNode(
                id=node.id, content=node.content + " [cycle]"
            )

        tree_node = CausalTreeNode(id=node.id, content=node.content)

        if max_depth is not None and _depth >= max_depth:
            return tree_node

        new_visited = visited | {node_id}

        for rel in self._rels_to_target.get(node_id, []):
            if rel.type not in rel_types:
                continue
            parent_tree = self._build_causal_tree(
                rel.source, rel_types, max_depth, new_visited, _depth + 1
            )
            parent_tree.relationship_type = rel.type.value
            tree_node.parents.append(parent_tree)

        return tree_node

    def _has_multiple_paths(
        self, node_id: str, rel_types: list[RelationType]
    ) -> bool:
        incoming = self._rels_to_target.get(node_id, [])
        relevant = [r for r in incoming if r.type in rel_types]
        return len(relevant) > 1

    def _build_impact_tree(
        self, descendants: list[_TraversalNode]
    ) -> dict[str, list[ImpactConsequence]]:
        assert self._ir is not None

        tension_node_ids: set[str] = set()
        for rel in self._ir.relationships:
            if rel.type == RelationType.TENSION:
                tension_node_ids.add(rel.source)
                tension_node_ids.add(rel.target)

        direct = [d for d in descendants if d.depth == 1]
        indirect = [d for d in descendants if d.depth > 1]

        def to_consequence(d: _TraversalNode) -> ImpactConsequence:
            c = ImpactConsequence(
                id=d.node.id,
                content=d.node.content,
                relationship=d.relationship_type or "causes",
                depth=d.depth,
                has_tension=d.node.id in tension_node_ids,
            )
            if d.node.id in tension_node_ids:
                tension_rel = next(
                    (
                        r
                        for r in self._ir.relationships
                        if r.type == RelationType.TENSION
                        and (r.source == d.node.id or r.target == d.node.id)
                    ),
                    None,
                )
                if tension_rel and tension_rel.axis_label:
                    c.tension_axis = tension_rel.axis_label
            return c

        return {
            "direct": [to_consequence(d) for d in direct],
            "indirect": [to_consequence(d) for d in indirect],
        }

    def _find_tensions_in_subgraph(self, node_ids: set[str]) -> list[TensionInfo]:
        assert self._ir is not None
        tensions: list[TensionInfo] = []
        for rel in self._ir.relationships:
            if (
                rel.type == RelationType.TENSION
                and rel.source in node_ids
                and rel.target in node_ids
            ):
                src = self._node_map.get(rel.source)
                tgt = self._node_map.get(rel.target)
                if src and tgt:
                    tensions.append(
                        TensionInfo(
                            axis=rel.axis_label or "unlabeled",
                            source={"id": src.id, "content": src.content},
                            target={"id": tgt.id, "content": tgt.content},
                        )
                    )
        return tensions

    def _build_impact_summary(
        self,
        source_node: Node,
        descendants: list[_TraversalNode],
        tensions: list[TensionInfo],
    ) -> ImpactSummary:
        benefits: list[str] = []
        risks: list[str] = []

        risk_keywords = {"risk", "problem", "issue", "error", "fail"}
        for d in descendants:
            lower = d.node.content.lower()
            if any(kw in lower for kw in risk_keywords):
                risks.append(d.node.content)
            else:
                benefits.append(d.node.content)

        n = len(descendants)
        key_tradeoff = None
        if tensions:
            t = tensions[0]
            key_tradeoff = (
                f"{t.axis} ({t.source['content']} vs {t.target['content']})"
            )

        return ImpactSummary(
            impact_summary=f"{source_node.content} affects {n} downstream consideration{'s' if n != 1 else ''}",
            benefits=benefits,
            risks=risks,
            key_tradeoff=key_tradeoff,
        )
