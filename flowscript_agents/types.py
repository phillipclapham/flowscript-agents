"""
FlowScript IR Pydantic Models.

Inlined from flowscript-ldp v0.2.1. These are the canonical IR types
for the flowscript-agents package — no external dependency needed.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================


class NodeType(str, Enum):
    STATEMENT = "statement"
    QUESTION = "question"
    THOUGHT = "thought"
    DECISION = "decision"
    BLOCKER = "blocker"
    INSIGHT = "insight"
    ACTION = "action"
    COMPLETION = "completion"
    ALTERNATIVE = "alternative"
    EXPLORING = "exploring"
    PARKING = "parking"
    BLOCK = "block"


class RelationType(str, Enum):
    CAUSES = "causes"
    TEMPORAL = "temporal"
    DERIVES_FROM = "derives_from"
    BIDIRECTIONAL = "bidirectional"
    TENSION = "tension"
    EQUIVALENT = "equivalent"
    DIFFERENT = "different"
    ALTERNATIVE = "alternative"
    ALTERNATIVE_WORSE = "alternative_worse"
    ALTERNATIVE_BETTER = "alternative_better"


class StateType(str, Enum):
    BLOCKED = "blocked"
    DECIDED = "decided"
    EXPLORING = "exploring"
    PARKING = "parking"


class NodeModifier(str, Enum):
    URGENT = "urgent"
    STRONG_POSITIVE = "strong_positive"
    HIGH_CONFIDENCE = "high_confidence"
    LOW_CONFIDENCE = "low_confidence"


# =============================================================================
# Component Models
# =============================================================================


class Author(BaseModel):
    agent: str
    role: str  # "human" | "ai"


class Provenance(BaseModel):
    source_file: str
    line_number: int = Field(ge=1)
    timestamp: str  # ISO-8601
    author: Optional[Author] = None
    parser_version: Optional[str] = None
    hash: Optional[str] = None

    @field_validator("hash")
    @classmethod
    def validate_hash(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            import re

            if not re.fullmatch(r"[a-f0-9]{64}", v):
                raise ValueError(
                    "Content hash must be 64 lowercase hex characters (SHA-256)"
                )
        return v


class SourceSpan(BaseModel):
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    start_col: Optional[int] = Field(default=None, ge=1)
    end_col: Optional[int] = Field(default=None, ge=1)


class StateFields(BaseModel):
    """State-specific fields. Required fields depend on state type."""

    reason: Optional[str] = None  # Required for blocked
    since: Optional[str] = None  # Required for blocked
    rationale: Optional[str] = None  # Required for decided
    on: Optional[str] = None  # Required for decided
    why: Optional[str] = None  # Recommended for parking
    until: Optional[str] = None  # Recommended for parking
    hypothesis: Optional[str] = None  # Optional for exploring

    model_config = {"extra": "allow"}


class GraphInvariants(BaseModel):
    causal_acyclic: bool = True
    all_nodes_reachable: bool = True
    tension_axes_labeled: bool = True
    state_fields_present: bool = True


class IRMetadata(BaseModel):
    source_files: Optional[list[str]] = None
    parsed_at: Optional[str] = None
    parser: Optional[str] = None

    model_config = {"extra": "allow"}


# =============================================================================
# Core Graph Elements
# =============================================================================


class Node(BaseModel):
    id: str  # SHA-256 content hash
    type: NodeType
    content: str
    provenance: Provenance
    children: Optional[list[str]] = None
    source_span: Optional[SourceSpan] = None
    alias_of: Optional[str] = None
    modifiers: Optional[list[NodeModifier]] = None
    ext: Optional[dict[str, Any]] = None


class Relationship(BaseModel):
    id: str  # SHA-256 content hash
    type: RelationType
    source: str  # Node ID
    target: str  # Node ID
    axis_label: Optional[str] = None  # Required when type=tension
    provenance: Provenance
    feedback: bool = False
    ext: Optional[dict[str, Any]] = None


class State(BaseModel):
    id: str  # SHA-256 content hash
    type: StateType
    node_id: str  # ID of node this state applies to
    fields: Optional[StateFields] = None
    provenance: Provenance


# =============================================================================
# Top-Level IR Model
# =============================================================================


class IR(BaseModel):
    """Complete FlowScript IR graph. This is what parsers output."""

    version: str = "1.0.0"
    nodes: list[Node]
    relationships: list[Relationship]
    states: list[State]
    invariants: GraphInvariants = Field(default_factory=GraphInvariants)
    metadata: Optional[IRMetadata] = None
