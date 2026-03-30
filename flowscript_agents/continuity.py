"""
ContinuityManager — LLM-driven session boundary compression for Layer 1.

Produces a compressed continuity file from session memory, implementing
temporal graduation (1x→2x→3x→principle) and decision lifecycle management.
This is compression-as-cognition: the act of compressing forces abstraction,
making the system LEARN, not just STORE.

The continuity file is a 4-section markdown document:
  - State: Current focus, replaced each session
  - Patterns: Temporal graduation with FlowScript markers
  - Decisions: Committed decisions with lifecycle (active→clustered→archived)
  - Context: Compressed narrative, rewritten each session

Usage:
    from flowscript_agents.continuity import ContinuityManager

    mgr = ContinuityManager(llm=my_llm, max_chars=20000)
    continuity_text = mgr.produce(session_nodes, existing_continuity)
    mgr.save(continuity_text, "./agent.json")  # saves as ./agent.continuity.md
    loaded = mgr.load("./agent.json")  # reads the sidecar
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .embeddings.extract import ExtractFn


def _log(msg: str) -> None:
    """Log to stderr."""
    sys.stderr.write(f"[flowscript-continuity] {msg}\n")
    sys.stderr.flush()


# Matches graduated patterns (2x or 3x) with [evidence: <id> "explanation"] citations.
# Captures: (1) level, (2) date, (3) cited IDs, (4) optional explanation in quotes.
# Used by _validate_graduations to verify citations against actual session nodes.
_GRADUATION_RE = re.compile(
    r'\|\s*([23])x\s*\((\d{4}-\d{2}-\d{2})\)\s*\[evidence:\s*'
    r'([a-fA-F0-9][a-fA-F0-9, ]*)'  # one or more hex IDs
    r'(?:\s+"([^"]*)")?\s*\]'  # optional quoted explanation
)


# =============================================================================
# Result types
# =============================================================================


@dataclass
class ContinuityResult:
    """Result of producing a continuity file."""

    text: str
    char_count: int
    section_sizes: dict[str, int]  # section name → char count
    truncated: bool  # whether LLM output exceeded max_chars
    session_nodes_count: int  # how many nodes were in this session
    patterns_extracted: int  # estimated from output (best-effort)
    graduations_validated: int = 0  # citations that checked out
    graduations_demoted: int = 0  # citations that failed → demoted
    citation_reuse_max: int = 0  # max times any single node was cited (>2 = suspicious)


# =============================================================================
# Wrap prompt
# =============================================================================


def _build_wrap_prompt(
    session_summary: str,
    existing_continuity: str | None,
    project_name: str,
    max_chars: int,
    today: str | None = None,
) -> str:
    """Build the LLM prompt for continuity compression.

    This prompt is the most important piece of code in the system.
    It teaches the LLM temporal graduation, pattern extraction,
    and decision lifecycle — all without requiring FlowScript knowledge.

    Args:
        today: Override for current date string (for testing). Defaults to today's date.
    """
    if today is None:
        import datetime
        today = datetime.date.today().isoformat()

    existing_section = ""
    if existing_continuity and existing_continuity.strip():
        existing_section = f"""
## Existing Continuity File
(This is the current continuity file from previous sessions. Update it with the new session data.)

<existing_continuity>
{existing_continuity}
</existing_continuity>
"""
    else:
        existing_section = """
## Existing Continuity File
(No existing continuity — this is the first session. Create a fresh continuity file.)
"""

    return f"""You are a memory compression engine for an AI agent's reasoning memory.
Your job is to produce a compressed continuity file that captures the PRINCIPLES and
PATTERNS from this session, not just a summary. The act of compression is itself a
form of thinking — extract what matters, discard what doesn't.

## Session Data
(These are the typed reasoning nodes from this session's work.)

<session_data>
{session_summary}
</session_data>
{existing_section}
## Output Format

Produce a markdown file with EXACTLY these 4 section headers (use these headers VERBATIM):
`## State`, `## Patterns`, `## Decisions`, `## Context`
Stay within {max_chars} characters total.

### ## State
Replace completely with this session's current focus, active work, and status.
What is the agent working on RIGHT NOW? What's blocked? What's next?
Write in plain prose, 2-5 lines.

### ## Patterns
This is where learning happens. Use these markers for density:
- `? question` — open question needing decision
- `thought: insight` — observation or principle worth preserving
- `✓ item` — completed/resolved
- `A -> B` — A causes or leads to B
- `A ><[axis] B` — tension between A and B on the named axis
- `[decided(rationale: "why", on: "date")] choice` — committed decision
- `[blocked(reason: "what", since: "date")] item` — waiting on dependency

**Temporal graduation (CRITICAL — this is what makes the system learn):**
- Mark each pattern with `| Nx (date)` where N = validation count, date = last validated
- New observation from THIS session not in existing patterns → add at `| 1x ({today})`
- Observation that VALIDATES an existing 1x pattern → increment to:
  `| 2x ({today}) [evidence: <node_id> "brief explanation of how node validates pattern"]`
  where `<node_id>` is the 8-char ID prefix (e.g., `abc12345`) from the session data above.
  The explanation MUST reference specific content from the cited node.
- Observation that VALIDATES an existing 2x pattern → graduate to:
  `| 3x ({today}) [evidence: <node_id> "explanation"]`
- Evidence citations with explanations are REQUIRED for all graduations (2x and 3x).
  Cite the specific session node AND explain how it validates the pattern.
  Without a valid citation, the graduation will be rejected.
- For patterns you are NOT graduating (carrying forward at the same level), drop the
  `[evidence:]` tag — evidence only appears on the graduation that created it.
- Patterns marked `(ungrounded)` were demoted in a previous session due to invalid evidence.
  They need FRESH validating evidence from THIS session to be re-graduated. Do not re-graduate
  without new evidence — remove the `(ungrounded)` marker only when providing a valid citation.
  Patterns marked `(ungrounded)` that you cannot provide fresh evidence for should be removed.
- Patterns at 3x: extract the PRINCIPLE underneath, not the surface observations.
  Multiple related observations → single meta-pattern. This is compression-as-cognition.
- Patterns with dates older than 7 days and no new validation → remove (they're stale)

Group related patterns in FlowScript blocks: `{{topic: ... }}`

**Example Patterns section:**
```
{{database_architecture:
  thought: ACID compliance outweighs raw speed | 2x (2026-03-30) [evidence: 4931b6a8 "PostgreSQL chosen for ACID compliance"]
  thought: connection pooling is the real performance bottleneck | 1x (2026-03-30)
  ? horizontal scaling strategy ><[single-writer vs multi-writer] | 1x (2026-03-29)
}}
```

### ## Decisions
Committed decisions with rationale. Use `[decided()]` markers (which include dates).

**Decision lifecycle (prevents unbounded growth):**
- New decisions from this session → add with rationale and date
- Existing decisions still referenced by active State or Patterns → keep
- Cluster of 3+ related decisions pointing same direction → extract principle
  to Patterns section, archive the individual decisions
- Decisions referencing nothing in active State/Patterns AND with dates older than 30 days → remove
- Reversed decisions → mark as reversed with rationale, remove next session

### ## Context
Compressed narrative of recent work history. Plain prose, NOT markers.
Rewrite incorporating this session — compress previous narrative into momentum.
Shape of the work, not transcript. 5-15 lines.

## Quality Rules
- PRINCIPLES over facts. "We keep hitting X because Y" > "X happened again"
- JUDGMENT over mechanical rules. You decide what matters.
- DENSITY over length. One insightful line > three vague ones.
- When uncertain whether to keep something, ask: "Would the agent make a worse
  decision next session without this?" If no, cut it.
- Do NOT narrate FlowScript in prose ("we noted a thought about..."). Use markers directly.
- Do NOT include the session data verbatim. Compress it.

## Output
Return ONLY the markdown continuity file. No explanation, no preamble, no code fences.
Start directly with `# {project_name} — Memory (v1)`.
Do NOT add extra headers like `## Summary` or `## Overview`.
Do NOT wrap the output in markdown code fences.
Do NOT number the sections ("## Section 1: State" is WRONG, "## State" is correct)."""


# =============================================================================
# Session data formatting
# =============================================================================


def _format_session_nodes(
    nodes: list[Any],
    relationships: list[Any],
    states: list[Any],
    temporal_map: dict[str, Any] | None = None,
) -> str:
    """Format session memory nodes into a readable summary for the LLM.

    Takes raw IR objects from the Memory class and produces a structured
    text representation the wrap prompt can work with.
    """
    lines: list[str] = []

    if not nodes:
        return "(No nodes in this session)"

    # Group nodes by type for clearer presentation
    by_type: dict[str, list[Any]] = {}
    for node in nodes:
        type_name = node.type.value if hasattr(node.type, "value") else str(node.type)
        by_type.setdefault(type_name, []).append(node)

    for type_name, type_nodes in sorted(by_type.items()):
        lines.append(f"\n### {type_name.title()}s ({len(type_nodes)})")
        for node in type_nodes:
            tier_info = ""
            if temporal_map and node.id in temporal_map:
                t = temporal_map[node.id]
                tier = t.tier if hasattr(t, "tier") else t.get("tier", "current")
                freq = t.frequency if hasattr(t, "frequency") else t.get("frequency", 1)
                tier_info = f" [{tier}|{freq}x]"
            lines.append(f"- ({node.id[:8]}){tier_info} {node.content}")

    # Relationships
    if relationships:
        lines.append(f"\n### Relationships ({len(relationships)})")
        node_map = {n.id: n.content[:60] for n in nodes}
        for rel in relationships:
            rel_type = rel.type.value if hasattr(rel.type, "value") else str(rel.type)
            src = node_map.get(rel.source, rel.source[:8])
            tgt = node_map.get(rel.target, rel.target[:8])
            axis = f" [{rel.axis_label}]" if getattr(rel, "axis_label", None) else ""
            lines.append(f"- {src} --{rel_type}{axis}--> {tgt}")

    # States
    if states:
        lines.append(f"\n### States ({len(states)})")
        node_map = {n.id: n.content[:60] for n in nodes}
        for state in states:
            state_type = state.type.value if hasattr(state.type, "value") else str(state.type)
            node_content = node_map.get(state.node_id, state.node_id[:8])
            fields_str = ""
            if state.fields:
                field_parts = []
                for attr in ("rationale", "reason", "since", "on", "why", "until"):
                    val = getattr(state.fields, attr, None)
                    if val:
                        field_parts.append(f"{attr}: {val}")
                if field_parts:
                    fields_str = f" ({', '.join(field_parts)})"
            lines.append(f"- [{state_type}]{fields_str} {node_content}")

    return "\n".join(lines)


# =============================================================================
# ContinuityManager
# =============================================================================


class ContinuityManager:
    """Manages the Layer 1 continuity file — LLM-driven session compression.

    The continuity file is a lossy, principled compression of the agent's
    reasoning history. It implements temporal graduation (observations →
    validated patterns → proven principles) and decision lifecycle management
    (active → clustered → archived).

    Args:
        llm: LLM function for compression. Same signature as AutoExtract:
             (prompt: str) -> str.
        max_chars: Maximum size of the continuity file in characters.
                   Default 20000 (~5k tokens). Configurable via
                   FLOWSCRIPT_CONTINUITY_MAX_CHARS env var.
        project_name: Name for the continuity file header.
                      Default "Agent" or FLOWSCRIPT_PROJECT_NAME env var.
    """

    def __init__(
        self,
        llm: ExtractFn,
        max_chars: int | None = None,
        project_name: str | None = None,
    ) -> None:
        self._llm = llm
        self._max_chars = max_chars or int(
            os.environ.get("FLOWSCRIPT_CONTINUITY_MAX_CHARS", "20000")
        )
        self._project_name = project_name or os.environ.get(
            "FLOWSCRIPT_PROJECT_NAME", "Agent"
        )

    @property
    def max_chars(self) -> int:
        return self._max_chars

    @property
    def project_name(self) -> str:
        return self._project_name

    # -- Core API --

    def produce(
        self,
        memory: Any,
        existing_continuity: str | None = None,
        citations_seen: bool = False,
    ) -> ContinuityResult:
        """Produce a compressed continuity file from session memory.

        Args:
            memory: A Memory instance containing the session's nodes.
            existing_continuity: The current continuity file text (if any).
                                 Pass None for first session.
            citations_seen: If True, enforces citation requirement (fail-safe sunset).

        Returns:
            ContinuityResult with the compressed continuity text and metadata.
        """
        # Extract session data from Memory via internal attributes.
        # TODO: Memory should expose a public snapshot() method to avoid
        # coupling to private attributes. For now, this is the only way
        # to get the full graph data needed for compression.
        nodes = list(memory._nodes.values())
        relationships = list(memory._relationships)
        states = list(memory._states)
        temporal_map = dict(memory._temporal_map)

        return self.produce_from_nodes(
            nodes, relationships, states, existing_continuity, temporal_map,
            citations_seen=citations_seen,
        )

    def produce_from_nodes(
        self,
        nodes: list[Any],
        relationships: list[Any],
        states: list[Any],
        existing_continuity: str | None = None,
        temporal_map: dict[str, Any] | None = None,
        citations_seen: bool = False,
    ) -> ContinuityResult:
        """Produce continuity from raw node lists (alternative to Memory instance).

        Useful when you have nodes but not a full Memory object, e.g.,
        from a filtered set or from deserialized data.

        Args:
            citations_seen: If True, enforces citation requirement on all today's
                           graduations. Set from metadata after first successful citation.
        """
        import datetime
        today = datetime.date.today().isoformat()

        session_summary = _format_session_nodes(
            nodes, relationships, states, temporal_map
        )

        prompt = _build_wrap_prompt(
            session_summary=session_summary,
            existing_continuity=existing_continuity,
            project_name=self._project_name,
            max_chars=self._max_chars,
            today=today,
        )

        _log(f"Producing continuity ({len(nodes)} nodes, max {self._max_chars} chars)")

        raw_output = self._llm(prompt)
        text = raw_output.strip()

        # Validate structural integrity — the LLM must produce all 4 sections.
        # If validation fails, return existing continuity unchanged (fail-safe).
        if not self._validate_structure(text):
            _log("WARNING: LLM output missing required sections — keeping existing continuity")
            if existing_continuity:
                return ContinuityResult(
                    text=existing_continuity,
                    char_count=len(existing_continuity),
                    section_sizes=self._measure_sections(existing_continuity),
                    truncated=False,
                    session_nodes_count=len(nodes),
                    patterns_extracted=len(re.findall(r"\|\s*\d+x", existing_continuity)),
                )
            # No existing continuity and LLM failed — return the output anyway
            # (first session, something is better than nothing)
            _log("WARNING: No existing continuity to fall back to — using LLM output as-is")

        # Validate graduation citations against actual session nodes.
        # Only checks citations from today (carried-forward patterns are trusted).
        valid_ids = {n.id[:8].lower() for n in nodes}
        node_content_map = {n.id[:8].lower(): n.content for n in nodes}
        text, grad_validated, grad_demoted, reuse_max = self._validate_graduations(
            text, valid_ids, today=today, node_content_map=node_content_map,
            citations_seen=citations_seen,
        )
        if grad_demoted:
            _log(
                f"Graduation validation: {grad_validated} validated, "
                f"{grad_demoted} demoted (ungrounded)"
            )
        if reuse_max > 2:
            _log(
                f"Graduation warning: single node cited {reuse_max} times "
                f"(possible citation gaming)"
            )

        truncated = False
        if len(text) > self._max_chars:
            truncated = True
            text = self._truncate_to_sections(text)

        section_sizes = self._measure_sections(text)

        patterns_extracted = len(re.findall(r"\|\s*\d+x", text))

        return ContinuityResult(
            text=text,
            char_count=len(text),
            section_sizes=section_sizes,
            truncated=truncated,
            session_nodes_count=len(nodes),
            patterns_extracted=patterns_extracted,
            graduations_validated=grad_validated,
            graduations_demoted=grad_demoted,
            citation_reuse_max=reuse_max,
        )

    # -- File I/O --

    @staticmethod
    def continuity_path(memory_path: str) -> str:
        """Get the sidecar continuity file path for a memory file.

        Pattern: ./agent.json → ./agent.continuity.md
        Matches the VectorIndex sidecar pattern (.embeddings.json).
        """
        p = Path(memory_path)
        return str(p.parent / f"{p.stem}.continuity.md")

    @staticmethod
    def meta_path(memory_path: str) -> str:
        """Get the metadata sidecar path. ./agent.json → ./agent.continuity.meta.json"""
        p = Path(memory_path)
        return str(p.parent / f"{p.stem}.continuity.meta.json")

    @staticmethod
    def load_meta(memory_path: str) -> dict:
        """Load continuity metadata from the JSON sidecar.

        Returns a dict with keys: sessions_produced, citations_seen, format_version.
        Returns defaults if the file doesn't exist.
        """
        import json
        path = ContinuityManager.meta_path(memory_path)
        defaults = {"sessions_produced": 0, "citations_seen": False, "format_version": 1}
        if not os.path.exists(path):
            return defaults
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Merge with defaults for forward compatibility
            return {**defaults, **data}
        except (json.JSONDecodeError, OSError):
            _log(f"WARNING: corrupt continuity meta at {path} — using defaults")
            return defaults

    @staticmethod
    def save_meta(meta: dict, memory_path: str) -> str:
        """Save continuity metadata to the JSON sidecar. Atomic write."""
        import json
        path = ContinuityManager.meta_path(memory_path)
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return path

    # -- Validation --

    _REQUIRED_SECTIONS = {"state", "patterns", "decisions", "context"}

    @classmethod
    def _validate_structure(cls, text: str) -> bool:
        """Validate that LLM output contains all 4 required sections.

        Checks case-insensitively for ## headers containing each section name.
        Returns True if all sections found, False otherwise.
        """
        text_lower = text.lower()
        found = set()
        for line in text_lower.split("\n"):
            if line.startswith("##"):
                for section in cls._REQUIRED_SECTIONS:
                    if section in line:
                        found.add(section)
        return found == cls._REQUIRED_SECTIONS

    # Minimum meaningful words for explanation-to-node content overlap check.
    # Short/common words are excluded to avoid false positives.
    _STOP_WORDS = frozenset(
        "a an the is are was were be been being have has had do does did "
        "will would shall should may might can could this that these those "
        "it its he she they we you i me my our his her their in on at to "
        "for of by with from and or but not no nor so if as".split()
    )

    # Matches bare graduations (2x or 3x) WITHOUT [evidence:] tags.
    # Used to enforce citation requirement after fail-safe sunset.
    _BARE_GRADUATION_RE = re.compile(
        r"\|\s*([23])x\s*\((\d{4}-\d{2}-\d{2})\)\s*(?!\[evidence:)"
    )

    @staticmethod
    def _validate_graduations(
        text: str,
        valid_ids: set[str],
        today: str | None = None,
        node_content_map: dict[str, str] | None = None,
        citations_seen: bool = False,
    ) -> tuple[str, int, int, int]:
        """Validate evidence citations on graduated patterns.

        Scans the ## Patterns section for 2x/3x lines with [evidence: <id> "explanation"].
        Only validates citations whose date matches today (newly graduated this
        session). Carried-forward patterns from previous sessions pass through
        unchanged — their evidence was valid when originally graduated.

        Validation checks (all must pass for a citation to be accepted):
        1. At least one cited ID exists in the current session's node set
        2. If an explanation is provided and node_content_map is available,
           the explanation must reference actual content from the cited node
           (word overlap check — prevents citation of irrelevant nodes)

        If validation fails, demotes the graduation (3x→2x, 2x→1x).

        Fail-safe sunset: when citations_seen=True, today's graduations WITHOUT
        [evidence:] tags are also demoted. Before citations_seen, they pass through
        (migration grace period). Once the LLM demonstrates citation ability, it
        must always cite.

        Returns:
            (possibly_modified_text, validated_count, demoted_count, citation_reuse_max)
        """
        if today is None:
            import datetime
            today = datetime.date.today().isoformat()

        lines = text.split("\n")
        in_patterns = False
        validated = 0
        demoted = 0
        citation_counts: dict[str, int] = {}  # track per-node citation frequency

        for i, line in enumerate(lines):
            # Track section boundaries (substring match, consistent with _validate_structure)
            if line.startswith("## "):
                in_patterns = "pattern" in line.lower()
                continue
            if not in_patterns:
                continue

            match = _GRADUATION_RE.search(line)
            if match:
                level = int(match.group(1))  # 2 or 3
                date_str = match.group(2)    # YYYY-MM-DD
                cited_raw = match.group(3)
                explanation = match.group(4)  # may be None if no quotes

                # Only validate citations from THIS session (today's date).
                # Carried-forward patterns retain their evidence unchecked.
                if date_str != today:
                    continue

                # Normalize cited IDs: lowercase, truncate to 8 chars, filter empties
                cited_ids = {
                    cid.strip().lower()[:8]
                    for cid in re.split(r"[,\s]+", cited_raw)
                    if cid.strip()
                }

                # Track citation frequency
                for cid in cited_ids & valid_ids:
                    citation_counts[cid] = citation_counts.get(cid, 0) + 1

                # Check 1: at least one cited ID exists in session nodes
                ids_valid = bool(cited_ids & valid_ids)

                # Check 2: explanation references cited node content (if available)
                explanation_valid = True
                if ids_valid and explanation and node_content_map:
                    matched_id = next(iter(cited_ids & valid_ids))
                    node_content = node_content_map.get(matched_id, "")
                    if node_content:
                        explanation_valid = ContinuityManager._check_explanation_overlap(
                            explanation, node_content
                        )

                if ids_valid and explanation_valid:
                    validated += 1
                else:
                    demoted += 1
                    demoted_level = level - 1
                    old_marker = match.group(0)
                    new_marker = old_marker.replace(
                        f"| {level}x", f"| {demoted_level}x"
                    )
                    new_marker = re.sub(
                        r'\[evidence:\s*[a-fA-F0-9][a-fA-F0-9, ]*(?:\s+"[^"]*")?\s*\]',
                        "(ungrounded)", new_marker
                    )
                    lines[i] = line.replace(old_marker, new_marker)
                continue

            # Fail-safe sunset: once the LLM has demonstrated citation ability,
            # today's graduations WITHOUT [evidence:] are demoted.
            if not citations_seen:
                continue

            bare_match = ContinuityManager._BARE_GRADUATION_RE.search(line)
            if not bare_match:
                continue

            bare_level = int(bare_match.group(1))
            bare_date = bare_match.group(2)
            if bare_date != today:
                continue

            demoted += 1
            demoted_level = bare_level - 1
            old_marker = bare_match.group(0)
            new_marker = old_marker.replace(
                f"| {bare_level}x", f"| {demoted_level}x"
            )
            lines[i] = line.replace(old_marker, new_marker + " (needs-evidence)")

        reuse_max = max(citation_counts.values()) if citation_counts else 0
        return "\n".join(lines), validated, demoted, reuse_max

    @classmethod
    def _check_explanation_overlap(cls, explanation: str, node_content: str) -> bool:
        """Check if an explanation references actual content from the cited node.

        Uses word overlap (excluding stop words). At least one meaningful word
        from the explanation must appear in the node content. This prevents
        generic explanations like "confirms pattern" while allowing legitimate
        paraphrasing.
        """
        def meaningful_words(text: str) -> set[str]:
            return {
                w for w in re.split(r"[^a-zA-Z0-9]+", text.lower())
                if len(w) > 2 and w not in cls._STOP_WORDS
            }

        explanation_words = meaningful_words(explanation)
        node_words = meaningful_words(node_content)
        return bool(explanation_words & node_words)

    # -- File I/O --

    def save(self, text: str, memory_path: str) -> str:
        """Save continuity text to the sidecar file.

        Args:
            text: The continuity file content.
            memory_path: Path to the memory JSON file (sidecar derived from this).

        Returns:
            The path where the continuity file was saved.
        """
        path = self.continuity_path(memory_path)
        # Atomic write: temp file + rename (crash-safe)
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        _log(f"Saved continuity to {path} ({len(text)} chars)")
        return path

    @staticmethod
    def load(memory_path: str) -> str | None:
        """Load continuity text from the sidecar file.

        Args:
            memory_path: Path to the memory JSON file.

        Returns:
            The continuity text, or None if no continuity file exists.
        """
        path = ContinuityManager.continuity_path(memory_path)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # -- Internal helpers --

    def _truncate_to_sections(self, text: str) -> str:
        """Truncate to fit max_chars while preserving complete sections.

        Cuts from the bottom of the last section that exceeds the limit,
        ensuring we never have a partial section.
        """
        if len(text) <= self._max_chars:
            return text

        # Find section boundaries (## headers)
        lines = text.split("\n")
        section_starts: list[int] = []
        for i, line in enumerate(lines):
            if line.startswith("## "):
                section_starts.append(i)

        if not section_starts:
            # No sections found — hard truncate
            return text[: self._max_chars]

        # Build text incrementally by section, stop when we'd exceed limit
        result_lines: list[str] = []

        # Include everything before first section (title line)
        for i in range(section_starts[0]):
            result_lines.append(lines[i])

        # Add sections until we'd exceed limit
        for idx, start in enumerate(section_starts):
            end = section_starts[idx + 1] if idx + 1 < len(section_starts) else len(lines)
            section_lines = lines[start:end]
            candidate = "\n".join(result_lines + section_lines)
            if len(candidate) > self._max_chars and result_lines:
                # This section would exceed limit — truncate within it
                remaining = self._max_chars - len("\n".join(result_lines)) - 1
                partial: list[str] = []
                char_count = 0
                for line in section_lines:
                    if char_count + len(line) + 1 > remaining:
                        break
                    partial.append(line)
                    char_count += len(line) + 1
                result_lines.extend(partial)
                break
            result_lines.extend(section_lines)

        return "\n".join(result_lines)

    @staticmethod
    def _measure_sections(text: str) -> dict[str, int]:
        """Measure character count per section."""
        sections: dict[str, int] = {}
        current_section = "_header"
        current_chars = 0

        for line in text.split("\n"):
            if line.startswith("## "):
                # Save previous section
                if current_chars > 0:
                    sections[current_section] = current_chars
                current_section = line[3:].strip()
                current_chars = len(line) + 1
            else:
                current_chars += len(line) + 1

        # Save last section
        if current_chars > 0:
            sections[current_section] = current_chars

        return sections
