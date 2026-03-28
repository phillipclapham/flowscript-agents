"""
FixpointContext — Convergence tracking and attestation for fixpoint computations.

Every fixpoint computation in FlowScript (consolidation, garden/prune, future custom @fix)
runs within a FixpointContext. The context:
- Emits fixpoint_start/fixpoint_iteration/fixpoint_end audit events
- Tracks delta sequences (new content-hashes per iteration)
- Computes graph state hashes (content-aware, not just IDs)
- Produces FixpointResult with convergence attestation

Design: FixpointContext wraps computations — it doesn't replace them.
ConsolidationEngine, GardenEngine, and future FixpointEngine all run INSIDE
a FixpointContext. The context is the audit + attestation layer; the engines
are the computation layer.

Attestation vs. verification (important distinction):
The certificate attests that "a computation ran, the graph changed from state A
to state B, N changes were recorded per iteration, and the computation terminated."
Full replay-based verification requires additional information not currently in the
certificate (LLM model version, exact prompts, non-deterministic choices). The
attestation is sufficient for EU AI Act Article 86 (proof that the decision process
was logged and terminated correctly). Future versions may add replay information
for L1 computations where deterministic replay is possible.

Usage:
    from flowscript_agents.fixpoint import FixpointContext

    # Run computation first, then wrap with certificate
    result = engine.consolidate(nodes, refs)
    ctx = FixpointContext.attest(memory, name="consolidation", constraint="L1",
                                 delta_sequence=[delta, 0])
    certificate = ctx.result
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .memory import Memory


# Sentinel for graph hash failures — distinct from any valid SHA-256
_SENTINEL_HASH = hashlib.sha256(b"__fixpoint_graph_hash_unavailable__").hexdigest()


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class FixpointResult:
    """Result of a fixpoint computation — the convergence attestation.

    This is the data that flows to FlowScript Cloud for compliance dashboards.
    The attestation proves the operation happened, terminated, and the graph
    transitioned between known states. It is a compliance artifact, not a
    replay-based proof (see module docstring for distinction).
    """

    name: str
    constraint: str  # "L1" or "L2"
    status: str  # "converged", "bounded", or "error"
    iterations: int
    delta_sequence: list[int]
    initial_graph_hash: str
    final_graph_hash: str
    timestamp: str = ""  # ISO 8601 UTC — when computation occurred
    elapsed_ms: float = 0.0
    audited: bool = True  # False if audit trail writes failed
    graph_hash_valid: bool = True  # False if graph hash fell to sentinel
    # L2-specific (populated only for L2 computations)
    bound_type: Optional[str] = None  # "max_iterations", "timeout", "measure"
    bound_value: Optional[int] = None

    @property
    def converged(self) -> bool:
        """Whether the computation reached a natural fixpoint."""
        return self.status == "converged"

    @property
    def certificate_hash(self) -> str:
        """Hash of the complete certificate for tamper detection.

        Includes all semantically meaningful fields — two certificates with
        different constraints, bounds, graph states, timestamps, or audit
        status produce different hashes. Excludes elapsed_ms (variable).
        Certificate hash proves WHAT happened and WHEN. The audit trail's
        hash chain proves the temporal ordering within the event sequence.
        """
        cert_data = json.dumps({
            "name": self.name,
            "constraint": self.constraint,
            "status": self.status,
            "iterations": self.iterations,
            "delta_sequence": self.delta_sequence,
            "initial_graph_hash": self.initial_graph_hash,
            "final_graph_hash": self.final_graph_hash,
            "timestamp": self.timestamp,
            "audited": self.audited,
            "graph_hash_valid": self.graph_hash_valid,
            "bound_type": self.bound_type,
            "bound_value": self.bound_value,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(cert_data.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for audit trail / Cloud ingestion."""
        d: dict[str, Any] = {
            "name": self.name,
            "constraint": self.constraint,
            "status": self.status,
            "iterations": self.iterations,
            "delta_sequence": self.delta_sequence,
            "initial_graph_hash": self.initial_graph_hash,
            "final_graph_hash": self.final_graph_hash,
            "timestamp": self.timestamp,
            "elapsed_ms": self.elapsed_ms,
            "audited": self.audited,
            "graph_hash_valid": self.graph_hash_valid,
            "certificate_hash": self.certificate_hash,
        }
        if self.bound_type is not None:
            d["bound_type"] = self.bound_type
            d["bound_value"] = self.bound_value
        return d


# =============================================================================
# Context Manager
# =============================================================================


class FixpointContext:
    """Context manager for fixpoint computations.

    Wraps any computation with:
    - fixpoint_start audit event (entry, initial state)
    - fixpoint_iteration audit events (per iteration, delta tracking)
    - fixpoint_end audit event (exit, convergence attestation)

    The context doesn't run the computation — it wraps it. The engine
    (ConsolidationEngine, future FixpointEngine) runs inside the context.

    Args:
        memory: The Memory instance (for audit trail + graph hashing).
        name: Human-readable name for this computation.
        constraint: "L1" (bounded, always terminates) or "L2" (general, bounded).
        bound_type: For L2, the bound type ("max_iterations", "timeout", "measure").
        bound_value: For L2, the bound value.
    """

    def __init__(
        self,
        memory: "Memory",
        *,
        name: str,
        constraint: str = "L1",
        bound_type: Optional[str] = None,
        bound_value: Optional[int] = None,
        _pre_hash: Optional[str] = None,
    ) -> None:
        self._memory = memory
        self._name = name
        self._constraint = constraint
        self._bound_type = bound_type
        self._bound_value = bound_value
        self._pre_hash = _pre_hash  # pre-computed initial hash (for post-hoc wrapping)
        self._iterations = 0
        self._delta_sequence: list[int] = []
        self._initial_hash: str = ""
        self._final_hash: str = ""
        self._status = "declared"
        self._start_time: float = 0.0
        self._elapsed_ms: float = 0.0
        self._timestamp: str = ""
        self._audited: bool = True  # tracks whether ALL audit writes succeeded
        self._result: Optional[FixpointResult] = None

    def __enter__(self) -> "FixpointContext":
        self._start_time = time.monotonic()
        self._timestamp = datetime.now(timezone.utc).isoformat()
        # Use pre-computed hash if provided (for post-hoc wrapping where
        # the computation already ran before the context opened)
        self._initial_hash = self._pre_hash or self._compute_graph_hash()

        try:
            self._memory.write_audit("fixpoint_start", {
                "name": self._name,
                "constraint": self._constraint,
                "initial_graph_hash": self._initial_hash,
            })
        except Exception:
            self._audited = False
            print("FixpointContext: audit write failed (fixpoint_start)", file=sys.stderr)

        return self

    def record_iteration(self, delta_size: int) -> None:
        """Record a single iteration's delta (number of new content-hashes produced).

        Call this after each iteration of the computation. For degenerate @fix
        (consolidation), call twice: once with the actual delta, once with 0
        (convergence marker).

        Args:
            delta_size: Number of new content-hashes produced in this iteration.
                0 indicates convergence (no new structure derived).
        """
        iteration_num = len(self._delta_sequence)
        self._delta_sequence.append(delta_size)
        self._iterations = len(self._delta_sequence)

        try:
            self._memory.write_audit("fixpoint_iteration", {
                "name": self._name,
                "iteration": iteration_num,
                "delta_size": delta_size,
                "elapsed_ms": (time.monotonic() - self._start_time) * 1000,
            })
        except Exception:
            self._audited = False
            print(f"FixpointContext: audit write failed (fixpoint_iteration {iteration_num})", file=sys.stderr)

    @property
    def converged(self) -> bool:
        """Whether the computation has converged (last delta was 0)."""
        return len(self._delta_sequence) > 0 and self._delta_sequence[-1] == 0

    @property
    def result(self) -> Optional[FixpointResult]:
        """The computation result. Available after context exit."""
        return self._result

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self._elapsed_ms = (time.monotonic() - self._start_time) * 1000
        self._final_hash = self._compute_graph_hash()

        # Determine status — evidence-based, not assumption-based.
        # All constraint levels require the same evidence: last delta == 0.
        # Knaster-Tarski guarantees L1 WILL converge, but the certificate
        # must SHOW it converged via recorded evidence. Callers must
        # record_iteration(0) explicitly as the convergence marker.
        if exc_type is not None:
            self._status = "error"
        elif self.converged:
            self._status = "converged"
        elif self._iterations == 0:
            self._status = "converged"  # zero-match case: trivially converged (spec section 1.1)
        else:
            self._status = "bounded"

        # Check graph hash integrity — sentinel means hash computation failed
        graph_hash_valid = (
            self._initial_hash != _SENTINEL_HASH
            and self._final_hash != _SENTINEL_HASH
        )

        # Build result
        self._result = FixpointResult(
            name=self._name,
            constraint=self._constraint,
            status=self._status,
            iterations=self._iterations,
            delta_sequence=self._delta_sequence,
            initial_graph_hash=self._initial_hash,
            final_graph_hash=self._final_hash,
            timestamp=self._timestamp,
            elapsed_ms=self._elapsed_ms,
            audited=self._audited,
            graph_hash_valid=graph_hash_valid,
            bound_type=self._bound_type,
            bound_value=self._bound_value,
        )

        # Emit fixpoint_end with full attestation
        try:
            self._memory.write_audit("fixpoint_end", self._result.to_dict())
        except Exception:
            self._audited = False
            self._result.audited = False
            print("FixpointContext: audit write failed (fixpoint_end)", file=sys.stderr)

        return False  # Don't suppress exceptions

    @staticmethod
    def _compute_graph_hash_static(memory: "Memory") -> str:
        """Static version for pre-computing graph hash before context opens."""
        ctx = FixpointContext.__new__(FixpointContext)
        ctx._memory = memory
        return ctx._compute_graph_hash()

    def _compute_graph_hash(self) -> str:
        """Compute a deterministic content-aware hash of the current graph state.

        Hashes node content (via content-hash IDs which ARE content hashes),
        relationship structure (type + source + target), and state values.
        This captures content changes — an UPDATE that modifies node content
        produces a different node ID (content-hash), which changes the graph hash.

        Note: _nodes is dict[str, Node] where keys ARE content-hash IDs.
        _relationships and _states are lists.
        """
        try:
            # Node IDs are content-hashes — they change when content changes
            node_ids = sorted(self._memory._nodes.keys())

            # Relationships: hash type + endpoints (captures structural changes)
            rel_data = sorted(
                f"{r.type.value}:{r.source}:{r.target}"
                for r in self._memory._relationships
            )

            # States: hash type + node + fields (captures state changes)
            state_data = sorted(
                f"{s.type.value}:{s.node_id}:{json.dumps(s.fields.__dict__ if hasattr(s.fields, '__dict__') else s.fields, sort_keys=True)}"
                for s in self._memory._states
            )

            graph_data = json.dumps(
                {"nodes": node_ids, "rels": rel_data, "states": state_data},
                sort_keys=True,
                separators=(",", ":"),
            )
            return hashlib.sha256(graph_data.encode("utf-8")).hexdigest()
        except Exception as e:
            # Log the failure — silent sentinel is a compliance risk
            print(f"FixpointContext: graph hash computation failed: {e}", file=sys.stderr)
            return _SENTINEL_HASH
