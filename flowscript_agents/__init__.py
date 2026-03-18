"""
FlowScript Agents — Decision intelligence memory for AI agent frameworks.

Drop-in memory provider for LangGraph, CrewAI, Google ADK, and OpenAI Agents SDK.
Replaces vector retrieval with queryable reasoning: why(), tensions(), blocked(),
alternatives(), whatIf().

Usage:
    from flowscript_agents import Memory

    mem = Memory()
    q = mem.question("Which database?")
    mem.alternative(q, "Redis").decide(rationale="speed critical")
    mem.alternative(q, "SQLite").block(reason="no concurrent writes")
    print(mem.query.tensions())
"""

from .memory import (
    Memory,
    MemoryOptions,
    NodeRef,
    TemporalConfig,
    TemporalMeta,
    TemporalTierConfig,
    DormancyConfig,
    GardenReport,
    PruneReport,
    SessionStartResult,
    SessionEndResult,
    SessionWrapResult,
)

__version__ = "0.1.1"
__all__ = [
    "Memory",
    "MemoryOptions",
    "NodeRef",
    "TemporalConfig",
    "TemporalMeta",
    "TemporalTierConfig",
    "DormancyConfig",
    "GardenReport",
    "PruneReport",
    "SessionStartResult",
    "SessionEndResult",
    "SessionWrapResult",
]
