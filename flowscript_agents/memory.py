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
        """Load existing memory or create empty. Zero-friction entry point."""
        path = Path(file_path)
        if path.exists():
            return Memory.load(str(path))
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

    def _add_node(self, content: str, node_type: NodeType) -> NodeRef:
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
        raw = f"{state_type.value}:{node_id}"
        state_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()

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
        meta.tier = self._next_tier(meta.tier)

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

        garden_report = self.garden()

        return SessionStartResult(
            summary=f"Memory: {self.size} nodes",
            garden=garden_report,
            tier_counts=self._count_tiers(),
            total_nodes=self.size,
        )

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
        if self.audit_path:
            entry = {
                "timestamp": _now_iso(),
                "event": "prune",
                "nodes": [n.model_dump(mode="json", exclude_none=True) for n in pruned_nodes],
                "relationships": [r.model_dump(mode="json", exclude_none=True) for r in pruned_rels],
                "states": [s.model_dump(mode="json", exclude_none=True) for s in pruned_states],
                "temporal": pruned_temporal,
                "reason": f"pruned {len(dormant_ids)} dormant node(s)",
            }
            audit_path = Path(self.audit_path)
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

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

    # -- Internal helpers --

    def _count_tiers(self) -> dict[str, int]:
        counts: dict[str, int] = {
            "current": 0, "developing": 0, "proven": 0, "foundation": 0
        }
        for node_id, meta in self._temporal_map.items():
            if node_id in self._nodes:  # only count existing nodes
                counts[meta.tier] = counts.get(meta.tier, 0) + 1
        return counts

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
        del self._nodes[node_id]
        self._temporal_map.pop(node_id, None)
        self._relationships = [
            r for r in self._relationships
            if r.source != node_id and r.target != node_id
        ]
        self._states = [s for s in self._states if s.node_id != node_id]
        self._dirty = True
        return True

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
