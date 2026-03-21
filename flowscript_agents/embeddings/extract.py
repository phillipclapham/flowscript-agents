"""
AutoExtract — Typed reasoning extraction from agent conversations.

The key differentiator from Mem0: instead of extracting flat facts
("user likes pizza"), we extract typed reasoning structure — decisions
with rationale, tensions with axes, causal chains, blockers with reasons.

Pipeline:
1. Text → LLM extraction → typed JSON (nodes + relationships + states)
2. Validate + parse extraction results defensively
3. Semantic dedup: check embedding similarity against existing nodes
4. Content-hash dedup: exact match detection (built into Memory)
5. Create nodes/relationships/states in Memory

Design:
- LLM-agnostic via ExtractFn callback (same pattern as fromTranscript in TS)
- Defensive parsing: gracefully handles malformed LLM output
- Semantic dedup prevents near-duplicates (configurable threshold)
- Optional VectorIndex integration for similarity-based dedup
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Awaitable, Optional, Union

if TYPE_CHECKING:
    from ..memory import Memory, NodeRef

from ..types import NodeType, RelationType, StateType
from .index import VectorIndex
from ._utils import strip_llm_wrapping

# Type for the LLM extraction function
# Takes a prompt string, returns the LLM's response string
ExtractFn = Callable[[str], str]


# =============================================================================
# Result types
# =============================================================================


@dataclass
class ExtractedNode:
    """A node extracted from text before insertion."""

    type: str
    content: str
    state_type: str | None = None
    state_reason: str | None = None
    state_rationale: str | None = None


@dataclass
class ExtractedRelationship:
    """A relationship extracted from text."""

    type: str
    source_idx: int  # index into nodes list
    target_idx: int
    axis: str | None = None


@dataclass
class ExtractionResult:
    """Raw extraction before dedup and insertion."""

    nodes: list[ExtractedNode]
    relationships: list[ExtractedRelationship]


@dataclass
class IngestResult:
    """Result of ingesting text into memory."""

    nodes_created: int
    nodes_deduplicated: int  # matched existing via content-hash or similarity
    relationships_created: int
    states_created: int
    node_ids: list[str]  # IDs of all nodes (new + existing matches)

    # Consolidation-specific fields (populated when consolidation engine is active)
    nodes_updated: int = 0  # existing nodes updated with richer content
    nodes_related: int = 0  # new nodes connected to existing via typed relationships
    nodes_resolved: int = 0  # existing states resolved (unblocked, decided)
    nodes_novel: int = 0  # bypassed consolidation (no similar existing)
    consolidation_used: bool = False  # whether consolidation engine was active
    fallback_count: int = 0  # ADD actions from consolidation errors (>10% = degradation)


# =============================================================================
# Extraction prompt
# =============================================================================

EXTRACTION_PROMPT_BASE = """You are a reasoning structure extractor for FlowScript, a decision intelligence system.
Given text from an AI agent conversation or human input, extract the reasoning structure as typed nodes and relationships.

## Node Types
- thought: An observation, fact, preference, or idea
- decision: A choice that was made (include rationale in state)
- question: Something being explored or evaluated
- insight: A deeper understanding, pattern, or lesson learned
- action: Something to be done or that was done
- alternative: An option being considered for a question

## Relationship Types
- causes: A causes or leads to B
- tension: A and B are in tension on some axis (MUST include axis label)
- derives_from: B is derived from or based on A
- alternative: B is an alternative answer to question A

## State Types (optional, only when clearly indicated)
- blocked: Something is preventing progress (include reason)
- decided: A decision has been made (include rationale)
- exploring: Actively being investigated
- parking: Deferred for later (include why)

## Rules
- Extract ONLY what is explicitly stated or strongly implied
- Prefer fewer, higher-quality extractions over many weak ones
- Each node should capture ONE discrete piece of reasoning
- Relationships must connect nodes that have a clear logical connection
- Do NOT infer speculative relationships
- Keep node content concise but complete (one sentence typically)

## Output Format
Return ONLY valid JSON (no markdown fences, no explanation):
{
  "nodes": [
    {"type": "thought", "content": "..."}
  ],
  "relationships": [
    {"type": "causes", "source": 0, "target": 1}
  ],
  "states": [
    {"type": "decided", "node": 0, "rationale": "..."}
  ]
}

Relationship source/target and state node are zero-based indices into the nodes array.
For tension relationships, include "axis": "..." describing what they're in tension about.
For blocked states, include "reason": "..." instead of "rationale".
If no meaningful structure exists, return {"nodes": [], "relationships": [], "states": []}.
"""

# Actor-specific suffixes that tune extraction for different input sources.
# Human input: prioritize decisions, preferences, explicit statements.
# Agent output: prioritize observations, analysis, status updates.
_ACTOR_SUFFIX = {
    "user": (
        "\n## Actor Context\n"
        "This text is from a HUMAN USER. Prioritize extracting:\n"
        "- Decisions and preferences (these are authoritative)\n"
        "- Explicit goals and requirements\n"
        "- Constraints and blockers they've identified\n"
        "Weight user-stated decisions and preferences as high-confidence.\n\n"
    ),
    "agent": (
        "\n## Actor Context\n"
        "This text is from an AI AGENT. Prioritize extracting:\n"
        "- Analysis and observations (these are tentative)\n"
        "- Proposed alternatives and their trade-offs\n"
        "- Status updates and progress markers\n"
        "- Identified risks and tensions\n"
        "Weight agent observations as medium-confidence (may be revised).\n\n"
    ),
}

# Default prompt (no actor context) — backward compatible
EXTRACTION_PROMPT = EXTRACTION_PROMPT_BASE + "\n## Text to extract from:\n"


# =============================================================================
# Parsing helpers
# =============================================================================

# Valid node types (matching FlowScript NodeType enum values)
_VALID_NODE_TYPES = {
    "thought", "statement", "question", "decision", "insight",
    "action", "completion", "alternative", "blocker",
}

# Valid relationship types
_VALID_REL_TYPES = {
    "causes", "tension", "derives_from", "alternative",
    "temporal", "bidirectional", "equivalent", "different",
}

# Valid state types
_VALID_STATE_TYPES = {"blocked", "decided", "exploring", "parking"}

# Map from extraction type names to NodeType enum
_NODE_TYPE_MAP: dict[str, NodeType] = {
    "thought": NodeType.THOUGHT,
    "statement": NodeType.STATEMENT,
    "question": NodeType.QUESTION,
    "decision": NodeType.DECISION,
    "insight": NodeType.INSIGHT,
    "action": NodeType.ACTION,
    "completion": NodeType.COMPLETION,
    "alternative": NodeType.ALTERNATIVE,
    "blocker": NodeType.BLOCKER,
}

_REL_TYPE_MAP: dict[str, RelationType] = {
    "causes": RelationType.CAUSES,
    "tension": RelationType.TENSION,
    "derives_from": RelationType.DERIVES_FROM,
    "alternative": RelationType.ALTERNATIVE,
    "temporal": RelationType.TEMPORAL,
    "bidirectional": RelationType.BIDIRECTIONAL,
    "equivalent": RelationType.EQUIVALENT,
    "different": RelationType.DIFFERENT,
}

_STATE_TYPE_MAP: dict[str, StateType] = {
    "blocked": StateType.BLOCKED,
    "decided": StateType.DECIDED,
    "exploring": StateType.EXPLORING,
    "parking": StateType.PARKING,
}


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response, handling markdown fences and preamble.

    Uses bracket-counting to find the outermost JSON object, which is more
    reliable than greedy regex (which can match from first { to last } across
    unrelated content).
    """
    _EMPTY: dict[str, Any] = {"nodes": [], "relationships": [], "states": []}

    text = strip_llm_wrapping(text)

    # Try direct parse first (handles clean responses)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Bracket-counting: find each top-level { and its matching }
    # Try json.loads on each candidate substring
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_string = False
        escape_next = False
        for j in range(i, len(text)):
            c = text[j]
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[i : j + 1]
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, dict):
                            return result
                    except json.JSONDecodeError:
                        break  # this { didn't lead to valid JSON, try next
                    break

    return _EMPTY


def _parse_extraction(raw: dict[str, Any]) -> ExtractionResult:
    """Parse and validate raw extraction JSON into typed structures."""
    nodes: list[ExtractedNode] = []
    relationships: list[ExtractedRelationship] = []

    # Parse nodes
    for item in raw.get("nodes", []):
        if not isinstance(item, dict):
            continue
        node_type = str(item.get("type", "thought")).lower()
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        # Normalize type
        if node_type not in _VALID_NODE_TYPES:
            node_type = "thought"  # safe fallback
        nodes.append(ExtractedNode(type=node_type, content=content))

    # Parse relationships
    for item in raw.get("relationships", []):
        if not isinstance(item, dict):
            continue
        rel_type = str(item.get("type", "")).lower()
        if rel_type not in _VALID_REL_TYPES:
            continue
        source = item.get("source")
        target = item.get("target")
        if not isinstance(source, int) or not isinstance(target, int):
            continue
        if source < 0 or source >= len(nodes) or target < 0 or target >= len(nodes):
            continue
        if source == target:
            continue  # no self-loops
        axis = item.get("axis")
        if rel_type == "tension" and not axis:
            continue  # tensions require axis
        relationships.append(
            ExtractedRelationship(
                type=rel_type,
                source_idx=source,
                target_idx=target,
                axis=str(axis) if axis else None,
            )
        )

    # Parse states and attach to nodes
    for item in raw.get("states", []):
        if not isinstance(item, dict):
            continue
        state_type = str(item.get("type", "")).lower()
        if state_type not in _VALID_STATE_TYPES:
            continue
        node_idx = item.get("node")
        if not isinstance(node_idx, int) or node_idx < 0 or node_idx >= len(nodes):
            continue
        node = nodes[node_idx]
        node.state_type = state_type
        node.state_reason = item.get("reason")
        node.state_rationale = item.get("rationale")

    return ExtractionResult(nodes=nodes, relationships=relationships)


# =============================================================================
# AutoExtract
# =============================================================================


class AutoExtract:
    """Auto-extract typed reasoning structure from text into Memory.

    Usage:
        extractor = AutoExtract(memory, llm=my_llm_fn, embedder=embedder)
        result = extractor.ingest("User chose PostgreSQL for ACID compliance")

    With consolidation (type-aware memory management):
        from flowscript_agents.embeddings.consolidate import ConsolidationEngine
        engine = ConsolidationEngine(memory, provider=my_tool_llm, vector_index=index)
        extractor = AutoExtract(memory, llm=my_llm_fn, vector_index=index,
                                consolidation_engine=engine)

    The LLM function signature: (prompt: str) -> str
    It receives the extraction prompt and should return the LLM's response.
    """

    DEFAULT_MAX_RETRIES: int = 3

    def __init__(
        self,
        memory: Memory,
        llm: ExtractFn,
        vector_index: VectorIndex | None = None,
        dedup_threshold: float = 0.80,
        consolidation_engine: Any | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._memory = memory
        self._llm = llm
        self._vector_index = vector_index
        self._dedup_threshold = dedup_threshold
        self._consolidation_engine = consolidation_engine
        self._max_retries = max_retries

    @property
    def memory(self) -> Memory:
        return self._memory

    def ingest(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        actor: str | None = None,
    ) -> IngestResult:
        """Extract reasoning structure from text and add to memory.

        When a consolidation engine is configured, uses type-aware consolidation
        instead of simple semantic dedup. This is the core differentiator from Mem0:
        contradictions become tensions, causes become chains, blockers get resolved.

        Args:
            text: Raw text to extract from (conversation, notes, etc.)
            metadata: Optional metadata to store in node.ext
            actor: Source of the text — "user" or "agent". When provided, tunes
                extraction priorities: user input weights decisions/preferences
                higher, agent output weights observations/analysis. None = default
                (no actor-specific tuning).

        Returns:
            IngestResult with counts and node IDs
        """
        # Step 1: LLM extraction with retry
        # Build actor-aware prompt
        actor_suffix = _ACTOR_SUFFIX.get(actor, "") if actor else ""
        prompt = EXTRACTION_PROMPT_BASE + actor_suffix + "\n## Text to extract from:\n" + text

        # Non-retryable errors: programming bugs, auth failures, validation.
        _NO_RETRY = (TypeError, ValueError, KeyError, AttributeError)

        response: str | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._llm(prompt)
                break
            except _NO_RETRY:
                # Non-transient — don't retry, return empty
                print(f"AutoExtract: non-retryable error: {sys.exc_info()[1]}", file=sys.stderr)
                return IngestResult(
                    nodes_created=0, nodes_deduplicated=0,
                    relationships_created=0, states_created=0, node_ids=[],
                )
            except Exception as e:
                if attempt < self._max_retries - 1:
                    delay = (2 ** attempt) + random.uniform(0, 0.5)  # jitter prevents thundering herd
                    print(
                        f"AutoExtract: LLM call attempt {attempt + 1}/{self._max_retries} "
                        f"failed: {e}. Retrying in {delay:.1f}s...",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                else:
                    # All retries exhausted — return empty result rather than crash
                    print(f"AutoExtract: LLM call failed after {self._max_retries} attempts: {e}", file=sys.stderr)
                    return IngestResult(
                        nodes_created=0, nodes_deduplicated=0,
                        relationships_created=0, states_created=0, node_ids=[],
                    )

        if response is None:
            return IngestResult(
                nodes_created=0, nodes_deduplicated=0,
                relationships_created=0, states_created=0, node_ids=[],
            )

        # Step 2: Parse response
        raw = _extract_json(response)
        extraction = _parse_extraction(raw)

        if not extraction.nodes:
            # Warn if input was non-trivial but extraction found nothing
            if len(text.strip()) > 20:
                print(
                    f"AutoExtract: LLM returned 0 nodes from {len(text)}-char input. "
                    "The extraction prompt may need tuning for this LLM, or the input "
                    "may not contain extractable reasoning structure.",
                    file=sys.stderr,
                )
            return IngestResult(
                nodes_created=0, nodes_deduplicated=0,
                relationships_created=0, states_created=0, node_ids=[],
            )

        # Step 3: Create nodes — route through consolidation or simple dedup
        if self._consolidation_engine is not None:
            return self._ingest_with_consolidation(extraction, metadata)
        else:
            return self._ingest_with_simple_dedup(extraction, metadata)

    def _ingest_with_simple_dedup(
        self,
        extraction: ExtractionResult,
        metadata: dict[str, Any] | None,
    ) -> IngestResult:
        """Original ingest path: semantic dedup at fixed threshold.

        Used when no consolidation engine is configured.
        """
        node_refs: list[NodeRef] = []
        created = 0
        deduped = 0

        for extracted_node in extraction.nodes:
            # Check semantic dedup via vector index
            if self._vector_index is not None and self._vector_index.indexed_count > 0:
                similar = self._vector_index.search(
                    extracted_node.content, top_k=1, threshold=self._dedup_threshold
                )
                if similar:
                    # Found a semantic match — touch existing node instead
                    existing_id = similar[0].node_id
                    self._memory.touch_nodes([existing_id])
                    node = self._memory.get_node(existing_id)
                    if node is not None:
                        node_refs.append(self._memory.ref(existing_id))
                        deduped += 1
                        continue

            # No semantic match — create the node
            ref = self._create_node(extracted_node, metadata)
            node_refs.append(ref)
            created += 1

        # Create relationships and states
        rels_created = self._create_extraction_relationships(extraction, node_refs)
        states_created = self._apply_extraction_states(extraction, node_refs)

        return IngestResult(
            nodes_created=created,
            nodes_deduplicated=deduped,
            relationships_created=rels_created,
            states_created=states_created,
            node_ids=[ref.id for ref in node_refs],
        )

    def _ingest_with_consolidation(
        self,
        extraction: ExtractionResult,
        metadata: dict[str, Any] | None,
    ) -> IngestResult:
        """Consolidation-aware ingest: type-aware memory management.

        Flow:
        1. Create ALL extracted nodes + embed them (no dedup yet)
        2. Run consolidation engine (searches existing, triages, LLM decides)
        3. Consolidation may remove nodes (NONE/UPDATE) or add relationships (RELATE/RESOLVE)
        4. Create extraction relationships for surviving nodes
        5. Apply extraction states for surviving nodes
        """
        # Create all nodes and embed them
        node_refs: list[NodeRef | None] = []
        extracted_node_dicts: list[dict[str, Any]] = []

        for i, extracted_node in enumerate(extraction.nodes):
            ref = self._create_node(extracted_node, metadata)
            node_refs.append(ref)
            extracted_node_dicts.append({
                "index": i,
                "type": extracted_node.type,
                "content": extracted_node.content,
            })

        # Run consolidation
        consolidation_result = self._consolidation_engine.consolidate(
            extracted_node_dicts, node_refs
        )

        # Create extraction relationships for surviving nodes only
        # (consolidation may have removed some via UPDATE/NONE)
        rels_created = self._create_extraction_relationships(extraction, node_refs)

        # Apply extraction states for surviving nodes only
        states_created = self._apply_extraction_states(extraction, node_refs)

        # Count surviving nodes
        surviving_ids = [ref.id for ref in node_refs if ref is not None]
        # Also include target IDs from NONE actions (existing nodes that were touched)
        for action in consolidation_result.actions:
            if action.action == "NONE" and action.target_node_id:
                if action.target_node_id not in surviving_ids:
                    surviving_ids.append(action.target_node_id)

        return IngestResult(
            nodes_created=consolidation_result.nodes_added,
            nodes_deduplicated=consolidation_result.nodes_skipped,
            relationships_created=rels_created + consolidation_result.nodes_related + consolidation_result.nodes_resolved,
            states_created=states_created,
            node_ids=surviving_ids,
            # Consolidation-specific
            nodes_updated=consolidation_result.nodes_updated,
            nodes_related=consolidation_result.nodes_related,
            nodes_resolved=consolidation_result.nodes_resolved,
            nodes_novel=consolidation_result.nodes_novel,
            consolidation_used=True,
            fallback_count=consolidation_result.fallback_count,
        )

    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------

    def _create_node(
        self,
        extracted_node: ExtractedNode,
        metadata: dict[str, Any] | None,
    ) -> NodeRef:
        """Create a node in Memory and index it in VectorIndex."""
        creator = self._get_node_creator(extracted_node.type)
        ref = creator(extracted_node.content)

        # Store metadata in ext
        if metadata:
            if ref.node.ext is None:
                ref.node.ext = {}
            ref.node.ext.update(metadata)

        # Index the new node for future dedup/consolidation
        if self._vector_index is not None:
            self._vector_index.index_node(ref.id)

        return ref

    def _create_extraction_relationships(
        self,
        extraction: ExtractionResult,
        node_refs: list[NodeRef | None],
    ) -> int:
        """Create relationships from extraction results. Skips if either node was removed."""
        rels_created = 0
        for rel in extraction.relationships:
            if rel.source_idx >= len(node_refs) or rel.target_idx >= len(node_refs):
                continue
            source_ref = node_refs[rel.source_idx]
            target_ref = node_refs[rel.target_idx]
            # Skip if either node was removed by consolidation
            if source_ref is None or target_ref is None:
                continue
            rel_type = _REL_TYPE_MAP.get(rel.type)
            if rel_type is None:
                continue
            try:
                self._memory.relate(
                    source_ref, target_ref, rel_type,
                    axis_label=rel.axis,
                )
                rels_created += 1
            except (KeyError, ValueError):
                continue  # defensive — skip invalid relationships
        return rels_created

    def _apply_extraction_states(
        self,
        extraction: ExtractionResult,
        node_refs: list[NodeRef | None],
    ) -> int:
        """Apply states from extraction results. Skips if node was removed."""
        states_created = 0
        for i, extracted_node in enumerate(extraction.nodes):
            if i >= len(node_refs):
                break
            if extracted_node.state_type is None:
                continue
            ref = node_refs[i]
            # Skip if node was removed by consolidation
            if ref is None:
                continue
            try:
                if extracted_node.state_type == "decided":
                    ref.decide(
                        rationale=extracted_node.state_rationale or "extracted from conversation"
                    )
                    states_created += 1
                elif extracted_node.state_type == "blocked":
                    ref.block(
                        reason=extracted_node.state_reason or "extracted from conversation"
                    )
                    states_created += 1
                elif extracted_node.state_type == "exploring":
                    ref.explore()
                    states_created += 1
                elif extracted_node.state_type == "parking":
                    ref.park(
                        why=extracted_node.state_reason or "extracted from conversation"
                    )
                    states_created += 1
            except (KeyError, ValueError):
                continue
        return states_created

    def ingest_conversation(
        self,
        messages: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
        actor: str | None = None,
    ) -> IngestResult:
        """Extract reasoning from a conversation (list of role/content dicts).

        Formats messages into a readable conversation transcript, then extracts.
        If actor is not specified, auto-detects based on the majority of message roles:
        - "user"/"human" roles → actor="user"
        - "assistant"/"agent"/"system" roles → actor="agent"
        """
        lines = []
        user_count = 0
        agent_count = 0
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
            if role in ("user", "human"):
                user_count += 1
            elif role in ("assistant", "agent", "system"):
                agent_count += 1
        transcript = "\n".join(lines)

        # Auto-detect actor from message roles if not specified
        if actor is None and (user_count > 0 or agent_count > 0):
            actor = "user" if user_count > agent_count else "agent"

        return self.ingest(transcript, metadata=metadata, actor=actor)

    def _get_node_creator(self, type_str: str) -> Callable[[str], NodeRef]:
        """Get the Memory node creation method for a type string.

        Note: 'alternative' maps to _add_node with ALTERNATIVE type directly
        (not memory.alternative() which requires a question ref). The ALTERNATIVE
        relationship is created separately in Step 4 via memory.relate().
        'decision' and 'blocker' use thought() because they're semantically
        thoughts with decided/blocked states applied in Step 5.
        """
        creators: dict[str, Callable[[str], NodeRef]] = {
            "thought": self._memory.thought,
            "statement": self._memory.statement,
            "question": self._memory.question,
            "decision": self._memory.thought,  # decisions are thoughts with decided state
            "insight": self._memory.insight,
            "action": self._memory.action,
            "completion": self._memory.completion,
            "blocker": self._memory.thought,  # blockers are thoughts with blocked state
            "alternative": lambda content: self._memory._add_node(
                content, NodeType.ALTERNATIVE
            ),
        }
        return creators.get(type_str, self._memory.thought)

    def __repr__(self) -> str:
        has_index = self._vector_index is not None
        return (
            f"AutoExtract(dedup_threshold={self._dedup_threshold}, "
            f"vector_index={has_index})"
        )
