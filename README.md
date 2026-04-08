<p align="center">
  <img src="docs/brand/logo-512.png" alt="FlowScript" width="120" />
</p>

<h1 align="center">flowscript-agents</h1>

<p align="center"><strong>Python SDK for FlowScript reasoning memory</strong></p>

<p align="center"><em>MCP server, nine framework adapters, typed reasoning queries, hash-chained audit trail.</em></p>

<p align="center">
  <a href="https://pypi.org/project/flowscript-agents/"><img src="https://img.shields.io/pypi/v/flowscript-agents" alt="PyPI"></a>
  <a href="https://github.com/phillipclapham/flowscript-agents"><img src="https://img.shields.io/badge/tests-717%20passing-brightgreen" alt="Tests"></a>
  <a href="https://pypi.org/project/flowscript-agents/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
</p>

---

> **Evolution note:** The architecture explored here — transport-layer interception, consolidation engine, anti-inbreeding defense, hash-chained audit trail, temporal graduation — directly informed [**anneal-memory**](https://github.com/phillipclapham/anneal-memory), a two-layer memory system for AI agents that delivers these concepts as a zero-dependency MCP server. This repo is maintained as a reference implementation.

---

## What This Is

flowscript-agents is the Python SDK for FlowScript reasoning memory. It provides:

- **20 MCP tools** for reasoning memory in Claude Code, Cursor, and other MCP-compatible editors
- **Nine framework adapters** (LangGraph, CrewAI, Google ADK, OpenAI Agents, Pydantic AI, smolagents, LlamaIndex, Haystack, CAMEL-AI)
- **Six typed reasoning queries** — `why()`, `tensions()`, `blocked()`, `alternatives()`, `whatIf()`, `counterfactual()` — graph traversals, not text search
- **Consolidation engine** — typed operations (ADD/UPDATE/RELATE/RESOLVE/NONE) where contradictions become queryable tensions instead of silent deletions
- **Hash-chained audit trail** — SHA-256 linked, append-only, crash-safe, framework attribution, SIEM callbacks
- **SDK wrapper** — transport-layer interception for OpenAI/Anthropic clients (agent never knows)

717 tests. Published on PyPI. MIT licensed.

---

## Six Reasoning Queries

These are graph traversals — sub-millisecond, deterministic, no LLM calls.

| Query | Returns | Ask when |
|:------|:--------|:---------|
| `query_why(node_id)` | Causal chain backward from any decision | "Why did we choose this?" |
| `query_tensions()` | Tradeoffs with named axes | "What tradeoffs are we navigating?" |
| `query_blocked()` | Blockers + downstream impact | "What's stuck and what does it affect?" |
| `query_alternatives(node_id)` | Options considered + outcome | "What else did we consider?" |
| `query_what_if(node_id)` | Forward impact analysis | "What breaks if we change this?" |
| `query_counterfactual(node_id)` | What would need to change | "What would it take to reverse this?" |

No vector store can answer these. Embedding similarity tells you what *looks like* your query. These queries tell you what *caused*, *blocked*, *traded off against*, and *follows from* your agent's decisions.

---

## When Memories Contradict

Most memory systems handle contradictions by deleting. Mem0's consolidation uses ADD/UPDATE/DELETE — when facts contradict, the old one is replaced.

FlowScript doesn't delete. It **relates**.

When consolidation detects a contradiction, it creates a tension with a named axis. Both perspectives survive. The disagreement itself becomes queryable knowledge. This approach satisfies [AGM belief revision postulates](https://arxiv.org/abs/2603.17244) — the formal framework proving deletion is mathematically irrational for a reasoning agent.

---

## Framework Adapters

Each adapter implements your framework's native memory interface. You don't learn a new API — you get reasoning queries on top of the one you already use.

| Framework | Adapter | Install |
|:----------|:--------|:--------|
| **LangGraph** | `FlowScriptStore` → `BaseStore` | `pip install flowscript-agents[langgraph]` |
| **CrewAI** | `FlowScriptStorage` → `StorageBackend` | `pip install flowscript-agents[crewai]` |
| **Google ADK** | `FlowScriptMemoryService` → `BaseMemoryService` | `pip install flowscript-agents[google-adk]` |
| **OpenAI Agents** | `FlowScriptSession` → `Session` | `pip install flowscript-agents[openai-agents]` |
| **Pydantic AI** | `FlowScriptDeps` → Deps + tools | `pip install flowscript-agents[pydantic-ai]` |
| **smolagents** | `FlowScriptMemory` → Tool protocol | `pip install flowscript-agents[smolagents]` |
| **LlamaIndex** | `FlowScriptMemoryBlock` → `BaseMemoryBlock` | `pip install flowscript-agents[llamaindex]` |
| **Haystack** | `FlowScriptMemoryStore` → `MemoryStore` | `pip install flowscript-agents[haystack]` |
| **CAMEL-AI** | `FlowScriptCamelMemory` → `AgentMemory` | `pip install flowscript-agents[camel-ai]` |

All adapters expose `.memory` for direct query access and support `with` blocks for automatic session lifecycle.

---

## What Evolved into anneal-memory

| flowscript-agents explored | anneal-memory delivers |
|:---------------------------|:----------------------|
| Consolidation engine (ADD/UPDATE/RELATE/RESOLVE) | Compression-as-cognition through session wraps |
| Anti-inbreeding defense (citation validation) | Graduation gate + principle demotion |
| Hash-chained audit trail | Tamper-evident compliance layer |
| Transport-layer SDK wrapper | Compliance proxy vision (MCP interception) |
| 9-marker FlowScript subset for compression | Same subset, zero-dependency MCP |
| Temporal graduation (current → developing → proven → foundation) | Citation-validated graduation with immune system |
| Continuity file compression | Two-layer memory: episodes compress into identity |

anneal-memory delivers these concepts without requiring notation learning, API key configuration, or framework-specific adapters. Zero dependencies. Install and go.

**[anneal-memory on GitHub](https://github.com/phillipclapham/anneal-memory)** | **[anneal-memory on PyPI](https://pypi.org/project/anneal-memory/)**

---

## Documentation

- **[API Reference](docs/api-reference.md)** — Full Python SDK documentation
- **[Adapter Guide](docs/adapters.md)** — Per-framework integration examples
- **[Session Lifecycle](docs/lifecycle.md)** — Temporal tiers and session wraps
- **[Audit Trail](docs/audit-trail.md)** — Hash-chain architecture and compliance
- **[CLAUDE.md snippet](examples/CLAUDE.md.example)** — Instructions for agents using FlowScript tools

---

## Related

- **[FlowScript (TypeScript)](https://github.com/phillipclapham/flowscript)** — TypeScript SDK, notation spec, web editor, 779 tests
- **[anneal-memory](https://github.com/phillipclapham/anneal-memory)** — Where the core concepts live now
- **[flowscript.org](https://flowscript.org)** — Notation reference and interactive playground

---

<p align="center">
  Built by <a href="https://phillipclapham.com">Phill Clapham</a> · <a href="https://claphamdigital.com">Clapham Digital LLC</a>
</p>
