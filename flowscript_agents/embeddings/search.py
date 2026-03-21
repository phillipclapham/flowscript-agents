"""
UnifiedSearch — Merged ranking across vector similarity, keyword match, and temporal signals.

Combines three independent scoring signals:
1. Vector similarity (semantic match via embeddings)
2. Keyword match (word-level matching, existing find_nodes logic)
3. Temporal score (tier weight + recency + frequency)

Results are deduplicated by node_id and ranked by weighted combination.

Design:
- Works with or without VectorIndex (degrades gracefully)
- All scores normalized to [0, 1] before combining
- Configurable weights (defaults: vector=0.4, keyword=0.4, temporal=0.2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..memory import Memory

from .index import VectorIndex


# =============================================================================
# Result type
# =============================================================================


@dataclass
class UnifiedSearchResult:
    """A single unified search result with scores from each signal."""

    node_id: str
    content: str
    node_type: str
    combined_score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    temporal_score: float = 0.0
    tier: str | None = None
    frequency: int | None = None
    sources: list[str] = field(default_factory=list)  # which signals contributed


# =============================================================================
# Scoring helpers
# =============================================================================

# Tier weights: proven knowledge is more valuable in search
_TIER_WEIGHTS: dict[str, float] = {
    "foundation": 1.0,
    "proven": 0.8,
    "developing": 0.5,
    "current": 0.3,
}


def _keyword_score(query: str, content: str) -> float:
    """Word-level keyword matching. Returns 0-1 based on proportion of
    query words found in content."""
    query_words = set(query.lower().split())
    if not query_words:
        return 0.0
    content_lower = content.lower()
    matched = sum(1 for w in query_words if w in content_lower)
    return matched / len(query_words)


def _temporal_score(
    tier: str | None,
    frequency: int | None,
    last_touched: str | None,
) -> float:
    """Temporal score combining tier, frequency, and recency. Returns 0-1."""
    # Tier component (0-1)
    tier_score = _TIER_WEIGHTS.get(tier or "current", 0.3)

    # Frequency component (0-1, logarithmic scaling, capped)
    freq = frequency or 1
    freq_score = min(1.0, math.log2(freq + 1) / 5.0)  # log2(32+1)/5 ≈ 1.0

    # Recency component (0-1, exponential decay over 30 days)
    recency_score = 0.5  # default if no timestamp
    if last_touched:
        try:
            touched = datetime.fromisoformat(last_touched)
            now = datetime.now(timezone.utc)
            age_hours = max(0, (now - touched).total_seconds() / 3600)
            # Half-life of ~72 hours (3 days)
            recency_score = math.exp(-0.693 * age_hours / 72)
        except (ValueError, TypeError):
            pass

    # Weight: tier most important for search relevance
    return 0.5 * tier_score + 0.25 * freq_score + 0.25 * recency_score


# =============================================================================
# UnifiedSearch
# =============================================================================


class UnifiedSearch:
    """Combines vector similarity, keyword matching, and temporal signals.

    Usage:
        search = UnifiedSearch(memory, vector_index=index)
        results = search.search("database decisions", top_k=10)

    Without a VectorIndex, falls back to keyword + temporal only.
    """

    def __init__(
        self,
        memory: Memory,
        vector_index: VectorIndex | None = None,
        vector_weight: float = 0.4,
        keyword_weight: float = 0.4,
        temporal_weight: float = 0.2,
    ) -> None:
        self._memory = memory
        self._vector_index = vector_index
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight
        self._temporal_weight = temporal_weight

    def search(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float | None = None,
        keyword_weight: float | None = None,
        temporal_weight: float | None = None,
    ) -> list[UnifiedSearchResult]:
        """Search across all signals. Returns merged, deduplicated results.

        Args:
            query: Search query text
            top_k: Maximum results to return
            vector_weight: Override instance vector weight (0-1)
            keyword_weight: Override instance keyword weight (0-1)
            temporal_weight: Override instance temporal weight (0-1)
        """
        vw = vector_weight if vector_weight is not None else self._vector_weight
        kw = keyword_weight if keyword_weight is not None else self._keyword_weight
        tw = temporal_weight if temporal_weight is not None else self._temporal_weight

        # If no vector index, redistribute weight to keyword
        if self._vector_index is None or self._vector_index.indexed_count == 0:
            kw += vw
            vw = 0.0

        # Collect scores per node_id
        scores: dict[str, dict[str, float]] = {}  # node_id → {signal: score}
        sources: dict[str, list[str]] = {}  # node_id → [signal names]

        # -- Vector similarity --
        if vw > 0 and self._vector_index is not None:
            vector_results = self._vector_index.search(query, top_k=top_k * 2)
            for vr in vector_results:
                scores.setdefault(vr.node_id, {})["vector"] = vr.score
                sources.setdefault(vr.node_id, []).append("vector")

        # -- Keyword matching --
        if kw > 0:
            for node_ref in self._memory.nodes:
                ks = _keyword_score(query, node_ref.content)
                if ks > 0:
                    scores.setdefault(node_ref.id, {})["keyword"] = ks
                    sources.setdefault(node_ref.id, []).append("keyword")

        # -- Temporal scoring (applied to all candidates) --
        all_candidates = set(scores.keys())
        for node_id in all_candidates:
            temporal = self._memory.get_temporal(node_id)
            if temporal:
                ts = _temporal_score(
                    temporal.tier,
                    temporal.frequency,
                    temporal.last_touched,
                )
                scores[node_id]["temporal"] = ts
                sources.setdefault(node_id, []).append("temporal")

        # -- Combine scores --
        results: list[UnifiedSearchResult] = []
        for node_id, signal_scores in scores.items():
            node = self._memory.get_node(node_id)
            if node is None:
                continue

            vs = signal_scores.get("vector", 0.0)
            ks_val = signal_scores.get("keyword", 0.0)
            ts = signal_scores.get("temporal", 0.0)

            combined = vw * vs + kw * ks_val + tw * ts

            temporal = self._memory.get_temporal(node_id)
            results.append(
                UnifiedSearchResult(
                    node_id=node_id,
                    content=node.content,
                    node_type=node.type.value,
                    combined_score=combined,
                    vector_score=vs,
                    keyword_score=ks_val,
                    temporal_score=ts,
                    tier=temporal.tier if temporal else None,
                    frequency=temporal.frequency if temporal else None,
                    sources=sources.get(node_id, []),
                )
            )

        # Sort by combined score descending
        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:top_k]

    def __repr__(self) -> str:
        has_vector = self._vector_index is not None
        return (
            f"UnifiedSearch(vector={has_vector}, "
            f"weights=v{self._vector_weight}/k{self._keyword_weight}/t{self._temporal_weight})"
        )
