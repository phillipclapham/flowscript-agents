"""
FlowScript Memory — Python implementation.

Programmatic builder for FlowScript IR graphs. The Python equivalent of
flowscript-core's Memory class, built on top of flowscript-ldp's IR types
and query engine.

Design:
- IR is the internal representation (same schema as TypeScript)
- Content-hash deduplication drives frequency tracking
- Query engine refreshes when IR changes
- JSON is canonical persistence
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from flowscript_ldp.ir import (
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
from flowscript_ldp.query import QueryEngine


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


class _QueryProxy:
    """Lazy query engine that refreshes when IR changes."""

    def __init__(self, memory: Memory) -> None:
        self._memory = memory
        self._engine: Optional[QueryEngine] = None
        self._dirty = True

    def _mark_dirty(self) -> None:
        self._dirty = True

    def _get_engine(self) -> QueryEngine:
        if self._dirty or self._engine is None:
            self._engine = QueryEngine(self._memory.to_ir())
            self._dirty = False
        return self._engine

    def why(self, node_id: str, **kwargs: Any) -> Any:
        return self._get_engine().why(node_id, **kwargs)

    def what_if(self, node_id: str, **kwargs: Any) -> Any:
        return self._get_engine().what_if(node_id, **kwargs)

    def tensions(self, **kwargs: Any) -> Any:
        return self._get_engine().tensions(**kwargs)

    def blocked(self, **kwargs: Any) -> Any:
        return self._get_engine().blocked(**kwargs)

    def alternatives(self, question_id: str, **kwargs: Any) -> Any:
        return self._get_engine().alternatives(question_id, **kwargs)


class Memory:
    """
    Programmatic builder for FlowScript reasoning graphs.

    Mirrors the TypeScript Memory class from flowscript-core.
    Built on flowscript-ldp's IR types and query engine.
    """

    def __init__(self, source_file: str = "memory-api") -> None:
        self._nodes: dict[str, Node] = {}
        self._relationships: list[Relationship] = []
        self._states: list[State] = []
        self._source_file = source_file
        self._file_path: Optional[str] = None
        self._query = _QueryProxy(self)

    @property
    def query(self) -> _QueryProxy:
        return self._query

    @property
    def size(self) -> int:
        return len(self._nodes)

    @property
    def nodes(self) -> list[NodeRef]:
        return [NodeRef(self, n) for n in self._nodes.values()]

    @property
    def file_path(self) -> Optional[str]:
        return self._file_path

    # -- Static constructors --

    @staticmethod
    def from_ir(ir: IR) -> Memory:
        """Create Memory from an existing IR graph."""
        mem = Memory()
        for node in ir.nodes:
            mem._nodes[node.id] = node
        mem._relationships = list(ir.relationships)
        mem._states = list(ir.states)
        mem._query._mark_dirty()
        return mem

    @staticmethod
    def load(file_path: str) -> Memory:
        """Load Memory from a JSON file."""
        path = Path(file_path)
        data = json.loads(path.read_text("utf-8"))
        ir = IR.model_validate(data)
        mem = Memory.from_ir(ir)
        mem._file_path = str(path.resolve())
        return mem

    @staticmethod
    def load_or_create(file_path: str) -> Memory:
        """Load existing memory or create empty. Zero-friction entry point."""
        path = Path(file_path)
        if path.exists():
            return Memory.load(str(path))
        mem = Memory()
        mem._file_path = str(path.resolve())
        return mem

    @staticmethod
    def from_json(data: dict[str, Any] | str) -> Memory:
        """Create Memory from JSON dict or string."""
        if isinstance(data, str):
            data = json.loads(data)
        ir = IR.model_validate(data)
        return Memory.from_ir(ir)

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
        self._query._mark_dirty()
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
        self._query._mark_dirty()

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
        self._query._mark_dirty()

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
            self._query._mark_dirty()
        return removed

    # -- Lookup --

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

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
        """Export as JSON-serializable dict."""
        return self.to_ir().model_dump(mode="json", exclude_none=True)

    def to_json_string(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_json(), indent=indent)

    def save(self, file_path: Optional[str] = None) -> None:
        """Save memory to JSON file. Uses atomic write (temp + rename)."""
        import tempfile
        import os

        target = file_path or self._file_path
        if target is None:
            raise ValueError("No file path provided and no stored path")
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file, then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".flowscript-"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(self.to_json_string())
            os.replace(tmp_path, str(path))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        self._file_path = str(path.resolve())

    # -- Internal helpers --

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and its associated relationships and states.

        Returns True if the node was found and removed, False otherwise.
        """
        if node_id not in self._nodes:
            return False
        del self._nodes[node_id]
        self._relationships = [
            r for r in self._relationships
            if r.source != node_id and r.target != node_id
        ]
        self._states = [s for s in self._states if s.node_id != node_id]
        self._query._mark_dirty()
        return True

    def _resolve_ref(self, ref: NodeRef | str) -> NodeRef:
        """Resolve a NodeRef or content string to a NodeRef.

        Resolution order:
        1. If NodeRef, return as-is
        2. If string matching an existing node ID, return that node
        3. Otherwise, create a new thought node with the string as content
        """
        if isinstance(ref, NodeRef):
            return ref
        # Try as node ID first
        if ref in self._nodes:
            return NodeRef(self, self._nodes[ref])
        # Create thought from content string
        return self.thought(ref)

    def __repr__(self) -> str:
        return (
            f"Memory(nodes={len(self._nodes)}, "
            f"relationships={len(self._relationships)}, "
            f"states={len(self._states)})"
        )
