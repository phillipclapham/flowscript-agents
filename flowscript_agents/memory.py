"""
FlowScript Memory — Python implementation.

Programmatic builder for FlowScript IR graphs with full temporal intelligence.
The Python equivalent of flowscript-core's Memory class, with inlined IR types
and query engine.

Design:
- IR is the internal representation (same schema as TypeScript)
- Content-hash deduplication drives frequency tracking
- Temporal metadata tracks engagement (frequency, tier, garden status)
- Session-scoped touch dedup prevents within-session frequency inflation
- Graduation promotes nodes through tiers based on cross-session engagement
- Query engine refreshes when IR changes
- JSON is canonical persistence (IR + temporal + config)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from .audit import AuditConfig, AuditQueryResult, AuditVerifyResult, AuditWriter
from .query import QueryEngine
from .types import (
    IR,
    GraphInvariants,
    IRMetadata,
    Node,
    NodeModifier,
    NodeType,
    Provenance,
    Relationship,
    RelationType,
    State,
    StateFields,
    StateType,
)

# =============================================================================
# Temporal Types
# =============================================================================

TemporalTier = Literal["current", "developing", "proven", "foundation"]

TIER_ORDER: list[TemporalTier] = ["current", "developing", "proven", "foundation"]


@dataclass
class TemporalMeta:
    """Per-node temporal metadata — drives graduation and garden classification."""

    created_at: str  # ISO-8601
    last_touched: str  # ISO-8601
    frequency: int  # Touch count (engagement signal)
    tier: TemporalTier  # current → developing → proven → foundation


@dataclass
class TemporalTierConfig:
    max_age: Optional[str] = None  # e.g., '24h', '7d', null = permanent
    graduation_threshold: Optional[int] = None  # frequency needed to promote


@dataclass
class DormancyConfig:
    resting: str = "3d"  # untouched this long = resting
    dormant: str = "7d"  # untouched this long = dormant
    archive: str = "30d"  # dormant this long = auto-archive


@dataclass
class TemporalConfig:
    tiers: Optional[dict[str, TemporalTierConfig]] = None
    dormancy: Optional[DormancyConfig] = None


@dataclass
class MemoryOptions:
    temporal: Optional[TemporalConfig] = None
    source_file: Optional[str] = None
    author: Optional[dict[str, str]] = None  # {agent, role}
    touch_on_query: bool = True
    audit: Optional[AuditConfig] = None  # Audit trail config (active when file_path set)


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class GardenReport:
    growing: list[str]  # node IDs
    resting: list[str]
    dormant: list[str]
    stats: dict[str, int]


@dataclass
class PruneReport:
    archived: list[str]  # node IDs
    count: int


@dataclass
class UpdateResult:
    """Result of update_node() — includes metadata for sidecar coordination."""
    ref: Any  # NodeRef (forward reference)
    old_id: str  # the original node ID before update
    merged: bool  # True if update content matched an existing node (collision merge)


@dataclass
class SessionStartResult:
    summary: str
    garden: GardenReport
    tier_counts: dict[str, int]
    total_nodes: int


@dataclass
class SessionEndResult:
    pruned: PruneReport
    garden: GardenReport
    saved: bool
    path: Optional[str]


@dataclass
class SessionWrapResult:
    nodes_before: int
    tiers_before: dict[str, int]
    pruned: PruneReport
    garden_after: GardenReport
    nodes_after: int
    tiers_after: dict[str, int]
    saved: bool
    path: Optional[str]


# =============================================================================
# Utilities
# =============================================================================


def _hash_content(content: str, node_type: str) -> str:
    """Generate SHA-256 content hash matching TypeScript hashContent."""
    raw = f"{node_type}:{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_provenance(source: str = "memory-api") -> Provenance:
    return Provenance(
        source_file=source,
        line_number=1,
        timestamp=_now_iso(),
    )


def _parse_duration(duration: str) -> float:
    """Parse duration string (e.g., '3d', '7d', '24h') to milliseconds."""
    match = re.match(r"^(\d+)(ms|s|m|h|d|w)$", duration)
    if not match:
        raise ValueError(f"Invalid duration: {duration}")
    value = int(match.group(1))
    unit = match.group(2)
    multipliers = {
        "ms": 1,
        "s": 1000,
        "m": 60 * 1000,
        "h": 60 * 60 * 1000,
        "d": 24 * 60 * 60 * 1000,
        "w": 7 * 24 * 60 * 60 * 1000,
    }
    return value * multipliers[unit]


# =============================================================================
# NodeRef
# =============================================================================


class NodeRef:
    """Fluent reference handle for a node in the memory graph."""

    def __init__(self, memory: Memory, node: Node) -> None:
        self._memory = memory
        self._node = node

    @property
    def id(self) -> str:
        return self._node.id

    @property
    def node(self) -> Node:
        return self._node

    @property
    def type(self) -> NodeType:
        return self._node.type

    @property
    def content(self) -> str:
        return self._node.content

    # -- Relationship builders (return self for chaining) --

    def causes(self, target: NodeRef | str) -> NodeRef:
        """This node causes target."""
        t = self._memory._resolve_ref(target)
        self._memory._add_relationship(self.id, t.id, RelationType.CAUSES)
        return self

    def then(self, target: NodeRef | str) -> NodeRef:
        """Temporal: this node leads to target."""
        t = self._memory._resolve_ref(target)
        self._memory._add_relationship(self.id, t.id, RelationType.TEMPORAL)
        return self

    def derives_from(self, source: NodeRef | str) -> NodeRef:
        """This node derives from source."""
        s = self._memory._resolve_ref(source)
        self._memory._add_relationship(s.id, self.id, RelationType.DERIVES_FROM)
        return self

    def tension_with(self, target: NodeRef | str, axis: str) -> NodeRef:
        """Tension between this and target on named axis."""
        t = self._memory._resolve_ref(target)
        self._memory._add_relationship(
            self.id, t.id, RelationType.TENSION, axis_label=axis
        )
        return self

    def bidirectional(self, target: NodeRef | str) -> NodeRef:
        """Bidirectional relationship."""
        t = self._memory._resolve_ref(target)
        self._memory._add_relationship(self.id, t.id, RelationType.BIDIRECTIONAL)
        return self

    # -- State builders (return self for chaining) --

    def decide(self, rationale: str, on: Optional[str] = None) -> NodeRef:
        """Mark as decided."""
        fields = StateFields(rationale=rationale, on=on or _now_iso())
        self._memory._add_state(self.id, StateType.DECIDED, fields)
        return self

    def block(self, reason: str, since: Optional[str] = None) -> NodeRef:
        """Mark as blocked."""
        fields = StateFields(reason=reason, since=since or _now_iso())
        self._memory._add_state(self.id, StateType.BLOCKED, fields)
        return self

    def park(self, why: str, until: Optional[str] = None) -> NodeRef:
        """Mark as parked."""
        fields = StateFields(why=why, until=until)
        self._memory._add_state(self.id, StateType.PARKING, fields)
        return self

    def explore(self) -> NodeRef:
        """Mark as exploring."""
        self._memory._add_state(self.id, StateType.EXPLORING)
        return self

    def unblock(self) -> NodeRef:
        """Remove blocked states."""
        self._memory._remove_states(self.id, StateType.BLOCKED)
        return self

    def __repr__(self) -> str:
        return f"NodeRef({self.type.value}: {self.content!r})"


# =============================================================================
# Memory
# =============================================================================


class Memory:
    """
    Programmatic builder for FlowScript reasoning graphs with temporal intelligence.

    Features:
    - Fluent node/relationship/state API
    - Temporal metadata: frequency tracking, tier graduation, garden classification
    - Session lifecycle: session_start/session_end/session_wrap
    - Touch-on-query with session-scoped deduplication
    - Prune with append-only audit trail (.audit.jsonl)
    - Config persistence across save/load cycles
    """

    def __init__(self, options: Optional[MemoryOptions] = None, source_file: str = "memory-api") -> None:
        self._config = options or MemoryOptions()
        self._nodes: dict[str, Node] = {}
        self._relationships: list[Relationship] = []
        self._states: list[State] = []
        self._temporal_map: dict[str, TemporalMeta] = {}
        self._source_file = self._config.source_file or source_file
        self._file_path: Optional[str] = None
        self._query_engine = QueryEngine()
        self._dirty = True
        self._session_touch_set: set[str] = set()
        self._audit_writer: Optional[AuditWriter] = None
        self._session_id: Optional[str] = None
        self._adapter_context: Optional[dict[str, str]] = None

        # Resolve dormancy config
        dormancy = self._config.temporal.dormancy if self._config.temporal else None
        self._dormancy = DormancyConfig(
            resting=dormancy.resting if dormancy else "3d",
            dormant=dormancy.dormant if dormancy else "7d",
            archive=dormancy.archive if dormancy else "30d",
        )

    # -- Properties --

    @property
    def query(self) -> _QueryProxy:
        """Query proxy with touch-on-query support."""
        return _QueryProxy(self)

    @property
    def size(self) -> int:
        return len(self._nodes)

    @property
    def nodes(self) -> list[NodeRef]:
        return [NodeRef(self, n) for n in self._nodes.values()]

    @property
    def file_path(self) -> Optional[str]:
        return self._file_path

    @property
    def temporal_map(self) -> dict[str, TemporalMeta]:
        return self._temporal_map

    @property
    def audit_path(self) -> Optional[str]:
        """Path to audit log file (.audit.jsonl next to memory file)."""
        if not self._file_path:
            return None
        p = Path(self._file_path)
        return str(p.parent / (p.stem + ".audit.jsonl"))

    def _ensure_audit_writer(self) -> Optional[AuditWriter]:
        """Lazily create AuditWriter when file_path is set."""
        if self._audit_writer is not None:
            return self._audit_writer
        if self._file_path:
            self._audit_writer = AuditWriter(
                Path(self._file_path),
                config=self._config.audit,
            )
        return self._audit_writer

    def set_adapter_context(self, framework: str, adapter_class: str, operation: str) -> None:
        """Set adapter attribution for subsequent audit events.

        Call this from adapters before operations so audit entries include
        framework context. Call clear_adapter_context() when done.
        """
        self._adapter_context = {
            "framework": framework,
            "adapter_class": adapter_class,
            "operation": operation,
        }

    def clear_adapter_context(self) -> None:
        """Clear adapter attribution."""
        self._adapter_context = None

    # -- Static constructors --

    @staticmethod
    def from_ir(ir: IR) -> Memory:
        """Create Memory from an existing IR graph.

        Initializes temporal metadata for all nodes (defaults: frequency=1,
        tier='current'). This ensures nodes loaded from legacy IR-only format
        get temporal tracking and aren't immediately classified as dormant.
        """
        mem = Memory()
        now = _now_iso()
        for node in ir.nodes:
            mem._nodes[node.id] = node
            # Initialize temporal metadata for loaded nodes
            mem._temporal_map[node.id] = TemporalMeta(
                created_at=now,
                last_touched=now,
                frequency=1,
                tier="current",
            )
        mem._relationships = list(ir.relationships)
        mem._states = list(ir.states)
        mem._dirty = True
        return mem

    @staticmethod
    def load(file_path: str) -> Memory:
        """Load Memory from a JSON file. Restores temporal data + config if present."""
        path = Path(file_path)
        data = json.loads(path.read_text("utf-8"))

        # Check for MemoryJSON format (has flowscript_memory key)
        if "flowscript_memory" in data:
            mem = Memory._from_memory_json(data)
        else:
            # Legacy IR-only format
            ir = IR.model_validate(data)
            mem = Memory.from_ir(ir)

        mem._file_path = str(path.resolve())
        return mem

    @staticmethod
    def load_or_create(file_path: str, options: Optional[MemoryOptions] = None) -> Memory:
        """Load existing memory or create empty. Zero-friction entry point.

        Note: AuditConfig (on_event, rotation, verbosity) is applied from options
        even on load, since callbacks and runtime config cannot be serialized.
        """
        path = Path(file_path)
        if path.exists():
            mem = Memory.load(str(path))
            # Apply audit config from caller — AuditConfig can't be serialized
            # (on_event callbacks, runtime settings) so it must come from caller
            if options and options.audit:
                mem._config.audit = options.audit
            return mem
        mem = Memory(options=options)
        mem._file_path = str(path.resolve())
        return mem

    @staticmethod
    def from_json(data: dict[str, Any] | str) -> Memory:
        """Create Memory from JSON dict or string."""
        if isinstance(data, str):
            data = json.loads(data)
        if "flowscript_memory" in data:
            return Memory._from_memory_json(data)
        ir = IR.model_validate(data)
        return Memory.from_ir(ir)

    @staticmethod
    def _from_memory_json(data: dict[str, Any]) -> Memory:
        """Restore Memory from MemoryJSON format (IR + temporal + config)."""
        ir_data = data.get("ir", {})
        ir = IR.model_validate(ir_data)

        # Restore config
        config_data = data.get("config", {})
        options = MemoryOptions(
            touch_on_query=config_data.get("touch_on_query", True),
            source_file=config_data.get("source_file"),
            author=config_data.get("author"),
        )
        if "temporal" in config_data and config_data["temporal"]:
            tc = config_data["temporal"]
            dormancy = None
            if "dormancy" in tc and tc["dormancy"]:
                d = tc["dormancy"]
                dormancy = DormancyConfig(
                    resting=d.get("resting", "3d"),
                    dormant=d.get("dormant", "7d"),
                    archive=d.get("archive", "30d"),
                )
            tiers = None
            if "tiers" in tc and tc["tiers"]:
                tiers = {}
                for tier_name, tier_data in tc["tiers"].items():
                    tiers[tier_name] = TemporalTierConfig(
                        max_age=tier_data.get("max_age"),
                        graduation_threshold=tier_data.get("graduation_threshold"),
                    )
            options.temporal = TemporalConfig(dormancy=dormancy, tiers=tiers)

        mem = Memory(options=options)
        for node in ir.nodes:
            mem._nodes[node.id] = node
        mem._relationships = list(ir.relationships)
        mem._states = list(ir.states)

        # Restore temporal metadata
        temporal_data = data.get("temporal", {})
        for node_id, meta in temporal_data.items():
            mem._temporal_map[node_id] = TemporalMeta(
                created_at=meta["created_at"],
                last_touched=meta["last_touched"],
                frequency=meta["frequency"],
                tier=meta["tier"],
            )

        mem._dirty = True
        return mem

    # -- Node creation --

    def _add_node(self, content: str, node_type: NodeType, source: str = "api") -> NodeRef:
        node_id = _hash_content(content, node_type.value)
        if node_id in self._nodes:
            return NodeRef(self, self._nodes[node_id])

        node = Node(
            id=node_id,
            type=node_type,
            content=content,
            provenance=_make_provenance(self._source_file),
        )
        self._nodes[node_id] = node

        # Initialize temporal metadata
        now = _now_iso()
        self._temporal_map[node_id] = TemporalMeta(
            created_at=now,
            last_touched=now,
            frequency=1,
            tier="current",
        )

        self._dirty = True

        # Audit: node creation
        self._write_audit("node_create", {
            "node_id": node_id,
            "node_type": node_type.value,
            "content": content,
            "source": source,
        })

        return NodeRef(self, node)

    def thought(self, content: str) -> NodeRef:
        return self._add_node(content, NodeType.THOUGHT)

    def statement(self, content: str) -> NodeRef:
        return self._add_node(content, NodeType.STATEMENT)

    def question(self, content: str) -> NodeRef:
        return self._add_node(content, NodeType.QUESTION)

    def action(self, content: str) -> NodeRef:
        return self._add_node(content, NodeType.ACTION)

    def insight(self, content: str) -> NodeRef:
        return self._add_node(content, NodeType.INSIGHT)

    def completion(self, content: str) -> NodeRef:
        return self._add_node(content, NodeType.COMPLETION)

    def group(self, content: str) -> NodeRef:
        """Create a structural group container (block node)."""
        return self._add_node(content, NodeType.BLOCK)

    def alternative(self, question: NodeRef | str, content: str) -> NodeRef:
        """Create an alternative linked to a question node."""
        q = self._resolve_ref(question)
        alt = self._add_node(content, NodeType.ALTERNATIVE)
        self._add_relationship(q.id, alt.id, RelationType.ALTERNATIVE)
        return alt

    # -- Relationship creation --

    def _add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: RelationType,
        axis_label: Optional[str] = None,
    ) -> None:
        # Validate both nodes exist — prevents orphaned relationships from stale NodeRefs
        if source_id not in self._nodes:
            raise KeyError(f"Source node {source_id!r} does not exist (was it pruned?)")
        if target_id not in self._nodes:
            raise KeyError(f"Target node {target_id!r} does not exist (was it pruned?)")
        raw = f"{rel_type.value}:{source_id}:{target_id}"
        if axis_label:
            raw += f":{axis_label}"
        rel_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()

        # Deduplicate
        for existing in self._relationships:
            if existing.id == rel_id:
                return

        rel = Relationship(
            id=rel_id,
            type=rel_type,
            source=source_id,
            target=target_id,
            axis_label=axis_label,
            provenance=_make_provenance(self._source_file),
        )
        self._relationships.append(rel)
        self._dirty = True

        # Audit: relationship creation
        self._write_audit("relationship_create", {
            "relationship_id": rel_id,
            "type": rel_type.value,
            "source": source_id,
            "target": target_id,
            "axis_label": axis_label,
        })

    def tension(
        self, a: NodeRef | str, b: NodeRef | str, axis: str
    ) -> None:
        """Create a tension between two nodes on a named axis."""
        a_ref = self._resolve_ref(a)
        b_ref = self._resolve_ref(b)
        self._add_relationship(a_ref.id, b_ref.id, RelationType.TENSION, axis_label=axis)

    def relate(
        self,
        source: NodeRef | str,
        target: NodeRef | str,
        rel_type: RelationType,
        axis_label: Optional[str] = None,
    ) -> None:
        """Create an arbitrary relationship."""
        s = self._resolve_ref(source)
        t = self._resolve_ref(target)
        self._add_relationship(s.id, t.id, rel_type, axis_label=axis_label)

    # -- State management --

    def _add_state(
        self,
        node_id: str,
        state_type: StateType,
        fields: Optional[StateFields] = None,
    ) -> None:
        # Validate node exists — prevents orphaned states from stale NodeRefs
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id!r} does not exist (was it pruned?)")
        raw = f"{state_type.value}:{node_id}"
        state_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()

        # Capture previous state for audit BEFORE removal
        previous_state_data = None
        for s in self._states:
            if s.node_id == node_id and s.type == state_type:
                previous_state_data = s.model_dump(mode="json", exclude_none=True)
                break

        # Remove existing state of same type on same node
        self._states = [
            s for s in self._states
            if not (s.node_id == node_id and s.type == state_type)
        ]

        state = State(
            id=state_id,
            type=state_type,
            node_id=node_id,
            fields=fields,
            provenance=_make_provenance(self._source_file),
        )
        self._states.append(state)
        self._dirty = True

        # Audit: state change
        self._write_audit("state_change", {
            "state_id": state_id,
            "state_type": state_type.value,
            "node_id": node_id,
            "fields": fields.model_dump(mode="json", exclude_none=True) if fields else None,
            "previous_state": previous_state_data,
        })

    def _remove_states(
        self, node_id: str, state_type: Optional[StateType] = None
    ) -> int:
        before = len(self._states)
        if state_type:
            self._states = [
                s for s in self._states
                if not (s.node_id == node_id and s.type == state_type)
            ]
        else:
            self._states = [s for s in self._states if s.node_id != node_id]
        removed = before - len(self._states)
        if removed > 0:
            self._dirty = True
        return removed

    # -- Touch system --

    def touch_nodes(self, ids: list[str]) -> None:
        """Touch nodes explicitly. NOT session-scoped — always increments frequency."""
        for node_id in ids:
            self._touch_node(node_id, session_scoped=False)

    def _touch_node(self, node_id: str, session_scoped: bool = True) -> None:
        """Touch a single node: update lastTouched, conditionally increment frequency."""
        meta = self._temporal_map.get(node_id)
        if not meta:
            return

        # Always update lastTouched (recency is always relevant)
        meta.last_touched = _now_iso()

        # Session-scoped dedup: query touches max +1 per session per node
        if session_scoped:
            if node_id in self._session_touch_set:
                return  # Already touched this session
            self._session_touch_set.add(node_id)

        meta.frequency += 1

        # Check graduation threshold
        threshold = self._get_graduation_threshold(meta.tier)
        if meta.frequency >= threshold and meta.tier != "foundation":
            self._graduate(node_id, meta)

    def touch_nodes_session_scoped(self, ids: list[str]) -> None:
        """Touch nodes with session-scoped dedup (max +1 frequency per session).

        Call this when retrieving nodes to track engagement. Drives temporal
        graduation (current → developing → proven → foundation).

        Session-scoped means the same node is only touched once per session,
        preventing within-session frequency inflation from repeated queries.
        Use this in adapters for query-driven touches. Use touch_nodes() for
        explicit touches that should always increment.
        """
        for node_id in ids:
            self._touch_node(node_id, session_scoped=True)

    # Backward compat alias
    _touch_nodes_session_scoped = touch_nodes_session_scoped

    # -- Graduation --

    def _get_graduation_threshold(self, tier: TemporalTier) -> int:
        """Get frequency threshold for promoting from current tier."""
        config_tiers = (
            self._config.temporal.tiers
            if self._config.temporal and self._config.temporal.tiers
            else {}
        )
        if tier == "current":
            cfg = config_tiers.get("developing")
            return cfg.graduation_threshold if cfg and cfg.graduation_threshold else 2
        elif tier == "developing":
            cfg = config_tiers.get("proven")
            return cfg.graduation_threshold if cfg and cfg.graduation_threshold else 3
        elif tier == "proven":
            cfg = config_tiers.get("foundation")
            return cfg.graduation_threshold if cfg and cfg.graduation_threshold else 5
        return 999999  # foundation can't promote

    def _next_tier(self, tier: TemporalTier) -> TemporalTier:
        if tier == "current":
            return "developing"
        elif tier == "developing":
            return "proven"
        elif tier == "proven":
            return "foundation"
        return "foundation"

    def _graduate(self, node_id: str, meta: TemporalMeta) -> None:
        """Auto-promote node to next tier."""
        old_tier = meta.tier
        meta.tier = self._next_tier(meta.tier)

        # Audit: graduation
        self._write_audit("graduation", {
            "node_id": node_id,
            "old_tier": old_tier,
            "new_tier": meta.tier,
            "frequency": meta.frequency,
        })

    # -- Garden classification --

    def garden(self) -> GardenReport:
        """Classify nodes into growing/resting/dormant based on lastTouched."""
        now_ms = datetime.now(timezone.utc).timestamp() * 1000
        resting_ms = _parse_duration(self._dormancy.resting)
        dormant_ms = _parse_duration(self._dormancy.dormant)

        growing: list[str] = []
        resting: list[str] = []
        dormant: list[str] = []

        for node in self._nodes.values():
            if node.type == NodeType.BLOCK:
                continue  # skip structural blocks

            meta = self._temporal_map.get(node.id)
            if not meta:
                dormant.append(node.id)
                continue

            touched_ms = datetime.fromisoformat(meta.last_touched).timestamp() * 1000
            age_ms = now_ms - touched_ms

            if age_ms > dormant_ms:
                dormant.append(node.id)
            elif age_ms > resting_ms:
                resting.append(node.id)
            else:
                growing.append(node.id)

        total = len(growing) + len(resting) + len(dormant)
        return GardenReport(
            growing=growing,
            resting=resting,
            dormant=dormant,
            stats={
                "total": total,
                "growing": len(growing),
                "resting": len(resting),
                "dormant": len(dormant),
            },
        )

    # -- Session lifecycle --

    def session_start(self, max_tokens: int = 4000) -> SessionStartResult:
        """Orient at session beginning. Resets touch dedup, returns memory summary."""
        self._session_touch_set = set()
        # Generate session ID for audit correlation (timestamp + random bytes for uniqueness)
        import os as _os
        self._session_id = "ses_" + hashlib.sha256(
            _now_iso().encode("utf-8") + _os.urandom(8)
        ).hexdigest()[:12]

        garden_report = self.garden()
        result = SessionStartResult(
            summary=f"Memory: {self.size} nodes",
            garden=garden_report,
            tier_counts=self._count_tiers(),
            total_nodes=self.size,
        )

        # Audit: session start
        self._write_audit("session_start", {
            "total_nodes": self.size,
            "tier_counts": self._count_tiers(),
            "garden": garden_report.stats,
        })

        return result

    def session_end(self) -> SessionEndResult:
        """Cleanup at session end. Prunes dormant nodes, saves if path set."""
        pruned = self.prune()
        garden_report = self.garden()

        saved = False
        save_path: Optional[str] = None
        if self._file_path:
            self.save()
            saved = True
            save_path = self._file_path

        # Audit: session end
        self._write_audit("session_end", {
            "nodes_pruned": pruned.count,
            "pruned_ids": pruned.archived,
            "garden": garden_report.stats,
        })

        # Clear session ID after audit
        self._session_id = None

        return SessionEndResult(
            pruned=pruned,
            garden=garden_report,
            saved=saved,
            path=save_path,
        )

    def session_wrap(self) -> SessionWrapResult:
        """Complete lifecycle — captures before/after snapshot around session_end."""
        nodes_before = self.size
        tiers_before = self._count_tiers()

        # Audit: session wrap (before end, which will log session_end)
        self._write_audit("session_wrap", {
            "nodes_before": nodes_before,
            "tiers_before": tiers_before,
        })

        end_result = self.session_end()

        return SessionWrapResult(
            nodes_before=nodes_before,
            tiers_before=tiers_before,
            pruned=end_result.pruned,
            garden_after=end_result.garden,
            nodes_after=self.size,
            tiers_after=self._count_tiers(),
            saved=end_result.saved,
            path=end_result.path,
        )

    # -- Prune --

    def prune(self) -> PruneReport:
        """Remove dormant nodes. Writes audit trail BEFORE removal (fail-safe)."""
        garden_report = self.garden()
        dormant_ids = set(garden_report.dormant)

        if not dormant_ids:
            return PruneReport(archived=[], count=0)

        # Capture data BEFORE removal for audit
        pruned_nodes = [n for n in self._nodes.values() if n.id in dormant_ids]
        pruned_rels = [
            r for r in self._relationships
            if r.source in dormant_ids or r.target in dormant_ids
        ]
        pruned_states = [s for s in self._states if s.node_id in dormant_ids]
        pruned_temporal = {
            nid: asdict(self._temporal_map[nid])
            for nid in dormant_ids
            if nid in self._temporal_map
        }

        # Write audit log BEFORE removal (write-first = crash-safe)
        self._write_audit("prune", {
            "nodes": [n.model_dump(mode="json", exclude_none=True) for n in pruned_nodes],
            "relationships": [r.model_dump(mode="json", exclude_none=True) for r in pruned_rels],
            "states": [s.model_dump(mode="json", exclude_none=True) for s in pruned_states],
            "temporal": pruned_temporal,
            "reason": f"pruned {len(dormant_ids)} dormant node(s)",
        })

        # Remove from active graph
        for nid in dormant_ids:
            del self._nodes[nid]
            self._temporal_map.pop(nid, None)
        self._relationships = [
            r for r in self._relationships
            if r.source not in dormant_ids and r.target not in dormant_ids
        ]
        self._states = [s for s in self._states if s.node_id not in dormant_ids]

        self._dirty = True
        return PruneReport(archived=list(dormant_ids), count=len(dormant_ids))

    @staticmethod
    def read_audit_log(audit_path: str) -> list[dict[str, Any]]:
        """Read audit log entries from .audit.jsonl file."""
        path = Path(audit_path)
        if not path.exists():
            return []
        entries = []
        for line in path.read_text("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # Skip malformed lines
        return entries

    @staticmethod
    def query_audit(
        audit_path: str,
        after: Optional[str] = None,
        before: Optional[str] = None,
        events: Optional[list[str]] = None,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        adapter: Optional[str] = None,
        limit: int = 100,
        verify_chain: bool = False,
    ) -> AuditQueryResult:
        """Query audit trail with filters across active + rotated files.

        Args:
            audit_path: Path to active .audit.jsonl or .audit.manifest.json
            after: Only entries after this ISO timestamp
            before: Only entries before this ISO timestamp
            events: Filter by event types (e.g., ["consolidation", "prune"])
            node_id: Filter by node involvement (searches data recursively)
            session_id: Filter by session ID
            adapter: Filter by adapter framework name
            limit: Maximum entries to return
            verify_chain: Also verify hash chain integrity
        """
        return AuditWriter.query(
            audit_path,
            after=after,
            before=before,
            events=events,
            node_id=node_id,
            session_id=session_id,
            adapter=adapter,
            limit=limit,
            verify_chain=verify_chain,
        )

    @staticmethod
    def verify_audit(audit_path: str) -> AuditVerifyResult:
        """Verify hash chain integrity of the audit trail.

        Args:
            audit_path: Path to active .audit.jsonl or .audit.manifest.json

        Returns:
            AuditVerifyResult with chain integrity status
        """
        return AuditWriter.verify(audit_path)

    # -- Lookup --

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def get_temporal(self, node_id: str) -> Optional[TemporalMeta]:
        """Get temporal metadata for a node."""
        return self._temporal_map.get(node_id)

    def ref(self, node_id: str) -> NodeRef:
        """Get a NodeRef by ID. Raises KeyError if not found."""
        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"No node with id {node_id!r}")
        return NodeRef(self, node)

    def find_nodes(self, content_match: str) -> list[NodeRef]:
        """Find nodes whose content contains the search string."""
        lower = content_match.lower()
        return [
            NodeRef(self, n)
            for n in self._nodes.values()
            if lower in n.content.lower()
        ]

    # -- Serialization --

    def to_ir(self) -> IR:
        """Export as FlowScript IR."""
        return IR(
            version="1.0.0",
            nodes=list(self._nodes.values()),
            relationships=list(self._relationships),
            states=list(self._states),
            invariants=GraphInvariants(),
            metadata=IRMetadata(
                source_files=[self._source_file],
                parsed_at=_now_iso(),
                parser="flowscript-agents",
            ),
        )

    def to_json(self) -> dict[str, Any]:
        """Export as MemoryJSON (IR + temporal + config). Full-fidelity persistence."""
        ir = self.to_ir()
        return {
            "flowscript_memory": "1.0.0",
            "ir": ir.model_dump(mode="json", exclude_none=True),
            "temporal": {
                nid: asdict(meta)
                for nid, meta in self._temporal_map.items()
            },
            "config": {
                "touch_on_query": self._config.touch_on_query,
                "source_file": self._config.source_file,
                "author": self._config.author,
                "temporal": {
                    "dormancy": asdict(self._dormancy),
                    "tiers": {
                        tier_name: asdict(tier_cfg)
                        for tier_name, tier_cfg in self._config.temporal.tiers.items()
                    } if self._config.temporal and self._config.temporal.tiers else None,
                } if self._config.temporal else None,
            },
        }

    def to_json_string(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_json(), indent=indent)

    def save(self, file_path: Optional[str] = None) -> None:
        """Save memory to JSON file. Uses atomic write (temp + rename)."""
        import os
        import tempfile

        target = file_path or self._file_path
        if target is None:
            raise ValueError("No file path provided and no stored path")
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".flowscript-"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(self.to_json_string())
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        self._file_path = str(path.resolve())

    # -- Public accessors (used by MCP server and external tools) --

    @property
    def relationship_count(self) -> int:
        """Number of relationships in the graph."""
        return len(self._relationships)

    @property
    def state_count(self) -> int:
        """Number of states in the graph."""
        return len(self._states)

    def count_tiers(self) -> dict[str, int]:
        """Count nodes per temporal tier."""
        counts: dict[str, int] = {
            "current": 0, "developing": 0, "proven": 0, "foundation": 0
        }
        for node_id, meta in self._temporal_map.items():
            if node_id in self._nodes:  # only count existing nodes
                counts[meta.tier] = counts.get(meta.tier, 0) + 1
        return counts

    # -- Internal helpers --

    # Keep private alias for backward compat with __repr__ and tests
    _count_tiers = count_tiers

    def _get_query_engine(self) -> QueryEngine:
        """Get query engine, rebuilding if dirty."""
        if self._dirty:
            self._query_engine.load(self.to_ir())
            self._dirty = False
        return self._query_engine

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and its associated relationships, states, and temporal data."""
        if node_id not in self._nodes:
            return False

        # Capture full state BEFORE removal for audit (write-first)
        node = self._nodes[node_id]
        removed_rels = [r for r in self._relationships if r.source == node_id or r.target == node_id]
        removed_states = [s for s in self._states if s.node_id == node_id]
        removed_temporal = self._temporal_map.get(node_id)
        self._write_audit("node_remove", {
            "node": node.model_dump(mode="json", exclude_none=True),
            "relationships": [r.model_dump(mode="json", exclude_none=True) for r in removed_rels],
            "states": [s.model_dump(mode="json", exclude_none=True) for s in removed_states],
            "temporal": asdict(removed_temporal) if removed_temporal else None,
        })

        del self._nodes[node_id]
        self._temporal_map.pop(node_id, None)
        self._relationships = [
            r for r in self._relationships
            if r.source != node_id and r.target != node_id
        ]
        self._states = [s for s in self._states if s.node_id != node_id]
        self._dirty = True
        return True

    def update_node(
        self,
        node_id: str,
        new_content: str,
        *,
        reason: str = "",
    ) -> "UpdateResult":
        """Update a node's content while preserving relationships, states, and temporal data.

        The node's ID is a content hash, so changing content changes the ID.
        All relationships, states, and temporal metadata are re-pointed to the new ID.
        An audit trail entry is written with old/new content and the reason.

        Args:
            node_id: ID of the existing node to update.
            new_content: The new content for the node.
            reason: Why this update happened (for audit trail).

        Returns:
            UpdateResult with ref (NodeRef), old_id (str), and merged (bool).
            Use result.ref for the updated node. Check result.merged to detect
            content-hash collision merges (important for sidecar coordination).

        Raises:
            KeyError: If node_id doesn't exist.

        Note:
            Node ``children`` and ``alias_of`` fields may contain node IDs
            that are NOT re-pointed by this method. These fields are only
            populated by parsed FlowScript IR (block/group nodes), not by
            programmatic Memory API usage. The consolidation engine operates
            on programmatic graphs, so this is safe for its use case.
        """
        old_node = self._nodes.get(node_id)
        if old_node is None:
            raise KeyError(f"Node {node_id!r} does not exist")

        # Generate new ID from new content
        new_id = _hash_content(new_content, old_node.type.value)

        # If content hash is unchanged, just return existing ref
        if new_id == node_id:
            return UpdateResult(ref=NodeRef(self, old_node), old_id=node_id, merged=False)

        # If new ID already exists (content-hash collision with another node),
        # merge into the existing node instead of creating a duplicate
        if new_id in self._nodes:
            existing_target = self._nodes[new_id]
            # Write audit trail BEFORE mutation (crash-safe)
            self._write_audit("update_node_merge", {
                "old_id": node_id,
                "merged_into": new_id,
                "old_content": old_node.content,
                "target_content": existing_target.content,
                "node_type": old_node.type.value,
                "reason": reason,
            })
            # Re-point old relationships to the existing target
            self._repoint_references(node_id, new_id)
            # Merge temporal metadata — preserve the richer history
            self._merge_temporal(node_id, new_id)
            # Remove the old node (relationships already re-pointed)
            del self._nodes[node_id]
            self._temporal_map.pop(node_id, None)
            self._dirty = True
            return UpdateResult(ref=NodeRef(self, existing_target), old_id=node_id, merged=True)

        # Write audit trail BEFORE mutation (crash-safe — matches merge path)
        self._write_audit("update_node", {
            "old_id": node_id,
            "new_id": new_id,
            "old_content": old_node.content,
            "new_content": new_content,
            "node_type": old_node.type.value,
            "reason": reason,
        })

        # Create new node preserving type, provenance, children, etc.
        new_node = Node(
            id=new_id,
            type=old_node.type,
            content=new_content,
            provenance=old_node.provenance,
            children=old_node.children,
            source_span=old_node.source_span,
            alias_of=old_node.alias_of,
            modifiers=old_node.modifiers,
            ext=old_node.ext,
        )

        # Add new node, remove old
        self._nodes[new_id] = new_node
        del self._nodes[node_id]

        # Re-point all relationships and states
        self._repoint_references(node_id, new_id)

        # Transfer temporal metadata — preserve created_at, update last_touched
        old_temporal = self._temporal_map.pop(node_id, None)
        if old_temporal:
            self._temporal_map[new_id] = TemporalMeta(
                created_at=old_temporal.created_at,
                last_touched=_now_iso(),
                frequency=old_temporal.frequency,
                tier=old_temporal.tier,
            )
        else:
            self._temporal_map[new_id] = TemporalMeta(
                created_at=_now_iso(),
                last_touched=_now_iso(),
                frequency=1,
                tier="current",
            )

        self._dirty = True
        return UpdateResult(ref=NodeRef(self, new_node), old_id=node_id, merged=False)

    def _repoint_references(self, old_id: str, new_id: str) -> None:
        """Re-point all relationships and states from old_id to new_id."""
        # Re-point relationships — need to rebuild since Relationship is a Pydantic model
        updated_rels = []
        for rel in self._relationships:
            source = new_id if rel.source == old_id else rel.source
            target = new_id if rel.target == old_id else rel.target
            if source != rel.source or target != rel.target:
                # Re-hash the relationship ID with new node IDs
                raw = f"{rel.type.value}:{source}:{target}"
                if rel.axis_label:
                    raw += f":{rel.axis_label}"
                new_rel_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                updated_rels.append(Relationship(
                    id=new_rel_id,
                    type=rel.type,
                    source=source,
                    target=target,
                    axis_label=rel.axis_label,
                    provenance=rel.provenance,
                    feedback=rel.feedback,
                ))
            else:
                updated_rels.append(rel)
        # Filter self-referential relationships (can arise when A→B and A merges into B)
        updated_rels = [r for r in updated_rels if r.source != r.target]
        # Deduplicate by relationship ID — merging nodes can create duplicate rels
        # (e.g., both old and target node had causes→C, re-pointing creates two causes→C)
        seen_ids: set[str] = set()
        deduped_rels = []
        for rel in updated_rels:
            if rel.id not in seen_ids:
                seen_ids.add(rel.id)
                deduped_rels.append(rel)
        self._relationships = deduped_rels

        # Re-point states
        updated_states = []
        for state in self._states:
            if state.node_id == old_id:
                # Re-hash state ID with new node ID
                raw = f"{state.type.value}:{new_id}"
                new_state_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                updated_states.append(State(
                    id=new_state_id,
                    type=state.type,
                    node_id=new_id,
                    fields=state.fields,
                    provenance=state.provenance,
                ))
            else:
                updated_states.append(state)
        # Deduplicate states by (type, node_id) — merging nodes with same state type
        # would create duplicates. Keep the first occurrence (typically the target's).
        seen_state_keys: set[tuple[str, str]] = set()
        deduped_states = []
        for state in updated_states:
            key = (state.type.value, state.node_id)
            if key not in seen_state_keys:
                seen_state_keys.add(key)
                deduped_states.append(state)
        self._states = deduped_states

    def _write_audit(self, event: str, data: dict[str, Any]) -> None:
        """Write an audit trail entry via AuditWriter (hash-chained, rotatable)."""
        writer = self._ensure_audit_writer()
        if writer is None:
            return
        writer.write(
            event=event,
            data=data,
            session_id=self._session_id,
            adapter=self._adapter_context,
        )

    def _merge_temporal(self, old_id: str, target_id: str) -> None:
        """Merge temporal metadata from old node into target — preserve the richer history.

        Design decisions:
        - created_at: min(old, target) — knowledge existed since the earliest
        - frequency: old + target — total engagement count across both
        - tier: highest of old/target — don't demote established knowledge
        - last_touched: now — the merge itself is an engagement event
        """
        old_temporal = self._temporal_map.get(old_id)
        target_temporal = self._temporal_map.get(target_id)
        if not old_temporal or not target_temporal:
            # If either is missing, just touch the target
            self._touch_node(target_id)
            return

        tier_order = {"current": 0, "developing": 1, "proven": 2, "foundation": 3}
        best_tier = (
            old_temporal.tier
            if tier_order.get(old_temporal.tier, 0) > tier_order.get(target_temporal.tier, 0)
            else target_temporal.tier
        )
        self._temporal_map[target_id] = TemporalMeta(
            created_at=min(old_temporal.created_at, target_temporal.created_at),
            last_touched=_now_iso(),
            frequency=old_temporal.frequency + target_temporal.frequency,
            tier=best_tier,
        )

    def _resolve_ref(self, ref: NodeRef | str) -> NodeRef:
        """Resolve a NodeRef or content string to a NodeRef."""
        if isinstance(ref, NodeRef):
            return ref
        if ref in self._nodes:
            return NodeRef(self, self._nodes[ref])
        return self.thought(ref)

    def __repr__(self) -> str:
        tiers = self._count_tiers()
        return (
            f"Memory(nodes={len(self._nodes)}, "
            f"relationships={len(self._relationships)}, "
            f"states={len(self._states)}, "
            f"tiers={tiers})"
        )


# =============================================================================
# Query Proxy (touch-on-query support)
# =============================================================================


class _QueryProxy:
    """Lazy query engine wrapper with touch-on-query support."""

    def __init__(self, memory: Memory) -> None:
        self._memory = memory

    def _get_engine(self) -> QueryEngine:
        return self._memory._get_query_engine()

    def _touch_result_nodes(self, result: Any, query_type: str) -> None:
        """Extract node IDs from query results and touch them (session-scoped)."""
        if self._memory._config.touch_on_query is False:
            return

        ids: list[str] = []
        if query_type == "why":
            if hasattr(result, "target"):
                ids.append(result.target.get("id", ""))
            if hasattr(result, "root_cause") and isinstance(result.root_cause, dict):
                ids.append(result.root_cause.get("id", ""))
            if hasattr(result, "causal_chain"):
                ids.extend(n.id for n in result.causal_chain)
        elif query_type == "what_if":
            if hasattr(result, "source"):
                ids.append(result.source.get("id", ""))
            if hasattr(result, "impact_tree") and isinstance(result.impact_tree, dict):
                for consequences in result.impact_tree.values():
                    if isinstance(consequences, list):
                        ids.extend(c.id for c in consequences if hasattr(c, "id"))
        elif query_type == "tensions":
            if hasattr(result, "tensions_by_axis") and result.tensions_by_axis:
                for details in result.tensions_by_axis.values():
                    for d in details:
                        ids.append(d.source.get("id", ""))
                        ids.append(d.target.get("id", ""))
            elif hasattr(result, "tensions") and result.tensions:
                for d in result.tensions:
                    ids.append(d.source.get("id", ""))
                    ids.append(d.target.get("id", ""))
        elif query_type == "blocked":
            if hasattr(result, "blockers"):
                for b in result.blockers:
                    ids.append(b.node.get("id", ""))
                    if b.transitive_causes:
                        ids.extend(c.get("id", "") for c in b.transitive_causes)
                    if b.transitive_effects:
                        ids.extend(e.get("id", "") for e in b.transitive_effects)
        elif query_type == "alternatives":
            if hasattr(result, "question") and isinstance(result.question, dict):
                ids.append(result.question.get("id", ""))
            if hasattr(result, "alternatives"):
                for a in result.alternatives:
                    if hasattr(a, "id"):
                        ids.append(a.id)

        # Filter empty strings and touch
        valid_ids = [i for i in ids if i]
        if valid_ids:
            self._memory.touch_nodes_session_scoped(valid_ids)

    def why(self, node_id: str, **kwargs: Any) -> Any:
        result = self._get_engine().why(node_id, **kwargs)
        self._touch_result_nodes(result, "why")
        return result

    def what_if(self, node_id: str, **kwargs: Any) -> Any:
        result = self._get_engine().what_if(node_id, **kwargs)
        self._touch_result_nodes(result, "what_if")
        return result

    def tensions(self, **kwargs: Any) -> Any:
        result = self._get_engine().tensions(**kwargs)
        self._touch_result_nodes(result, "tensions")
        return result

    def blocked(self, **kwargs: Any) -> Any:
        result = self._get_engine().blocked(**kwargs)
        self._touch_result_nodes(result, "blocked")
        return result

    def alternatives(self, question_id: str, **kwargs: Any) -> Any:
        result = self._get_engine().alternatives(question_id, **kwargs)
        self._touch_result_nodes(result, "alternatives")
        return result
