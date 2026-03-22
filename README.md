# flowscript-agents

**Agent memory that tracks why you decided, what conflicts, and what's blocked. Not just what was said.**

[![Tests](https://img.shields.io/badge/tests-581%20passing-brightgreen)](https://github.com/phillipclapham/flowscript-agents) [![PyPI](https://img.shields.io/pypi/v/flowscript-agents)](https://pypi.org/project/flowscript-agents/) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/flowscript-agents/)

---

Plain text in. Typed reasoning queries out:

```python
from openai import OpenAI
from flowscript_agents import UnifiedMemory
from flowscript_agents.embeddings import OpenAIEmbeddings

client = OpenAI()
llm = lambda prompt: (client.chat.completions.create(
    model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
).choices[0].message.content or "")

with UnifiedMemory("agent-memory.json", embedder=OpenAIEmbeddings(), llm=llm) as mem:
    mem.add("We chose Redis for session storage — sub-ms reads are critical for UX")
    mem.add("Redis cluster costs are killing us at $200/mo for 3 nodes")
    mem.add("Decided: switch to PostgreSQL — handles our scale at $15/mo")

    print(mem.memory.query.tensions())
    # → Tension: "sub-ms reads critical for UX" vs "cluster costs $200/mo"
    #   axis: "performance vs cost"

    print(mem.memory.query.blocked())
    # → what's stuck and why, with downstream impact

    print(mem.memory.query.why(node_id))
    # → full causal chain backward from any decision
```

Five queries that no vector store can answer — `why()`, `tensions()`, `blocked()`, `alternatives()`, `whatIf()` — over a typed semantic graph. Drop-in adapters for [9 agent frameworks](#works-with-your-stack). Hash-chained audit trail. And when memories contradict, we don't delete the old one — we create a queryable *tension*.

<p align="center">
  <img src="docs/flowscript-demo.png" alt="FlowScript — editor with .fs syntax, D3 reasoning graph, and tensions query results" width="800">
</p>

---

## Get Started

### MCP Server (Claude Code / Cursor — zero code)

```json
{
  "mcpServers": {
    "flowscript": {
      "command": "flowscript-mcp",
      "args": ["--memory", "./project-memory.json"],
      "env": {
        "OPENAI_API_KEY": "your-key"
      }
    }
  }
}
```

```bash
pip install flowscript-agents
```

Auto-detects your API key and configures the full stack — vector search, typed extraction, and contradiction handling. Also supports `ANTHROPIC_API_KEY`. 11 tools, zero additional setup.

### Python SDK

```bash
pip install flowscript-agents                       # Core
pip install flowscript-agents[langgraph]            # + LangGraph adapter
pip install flowscript-agents[crewai]               # + CrewAI adapter
pip install flowscript-agents[all]                  # Everything (9 frameworks)
```

Bracket syntax matters — it installs framework-specific dependencies.

---

## How It Works

FlowScript operates at three levels. Pick where you start:

**Level 1 — Reasoning graph, no API keys.** Use the `Memory` class directly to build typed nodes (thoughts, questions, decisions) with explicit relationships (causes, tensions, alternatives). Sub-ms queries, zero external deps. This is the power-user API. [Full docs →](docs/api-reference.md)

**Level 2 — Add vector search.** Pass an `embedder` to `UnifiedMemory` for semantic similarity search alongside reasoning queries. Three providers: OpenAI, SentenceTransformers, Ollama. [Details →](docs/api-reference.md#unifiedmemory)

**Level 3 — Full stack.** Add an `llm` for auto-extraction (plain text → typed nodes) and a `consolidation_provider` for contradiction handling. Or just use the MCP server, which auto-configures all of this from a single API key.

---

## Works With Your Stack

Drop-in adapters that implement your framework's native interface. Same API you already use — plus `query.tensions()`.

```python
from flowscript_agents.langgraph import FlowScriptStore

with FlowScriptStore("agent-memory.json") as store:
    # Standard LangGraph BaseStore operations
    store.put(("agents", "planner"), "db_decision", {"value": "chose Redis for speed"})
    items = store.search(("agents", "planner"), query="Redis")

    # What's new — typed reasoning queries on the same data
    tensions = store.memory.query.tensions()
    blockers = store.memory.query.blocked()

    # Resolve a store key to its full reasoning context
    node = store.resolve(("agents", "planner"), "db_decision")
```

| Framework | Adapter | Install |
|:----------|:--------|:--------|
| **LangGraph** | `FlowScriptStore` → `BaseStore` | `[langgraph]` |
| **CrewAI** | `FlowScriptStorage` → `StorageBackend` | `[crewai]` |
| **Google ADK** | `FlowScriptMemoryService` → `BaseMemoryService` | `[google-adk]` |
| **OpenAI Agents** | `FlowScriptSession` → `Session` | `[openai-agents]` |
| **Pydantic AI** | `FlowScriptDeps` → Deps + tools | `[pydantic-ai]` |
| **smolagents** | `FlowScriptMemory` → Tool protocol | `[smolagents]` |
| **LlamaIndex** | `FlowScriptMemoryBlock` → `BaseMemoryBlock` | `[llamaindex]` |
| **Haystack** | `FlowScriptMemoryStore` → `MemoryStore` | `[haystack]` |
| **CAMEL-AI** | `FlowScriptCamelMemory` → `AgentMemory` | `[camel-ai]` |

All adapters expose `.memory` for query access, support `with` blocks, and accept optional `embedder`/`llm`/`consolidation_provider` for vector search and extraction. [Per-framework examples →](docs/adapters.md)

---

## When Memories Contradict

Every other memory system handles contradictions by deleting. Mem0's consolidation uses ADD/UPDATE/DELETE/NONE — when facts contradict, the old memory is replaced. LangGraph's langmem does the same. CrewAI's consolidation is flat keep/update/delete.

FlowScript doesn't delete. It **relates**.

When consolidation detects a contradiction, it creates a `RELATE` — a tension with a named axis. Both memories survive. The disagreement itself becomes queryable knowledge.

| Action | What happens |
|:-------|:-------------|
| `ADD` | New knowledge, no existing match |
| `UPDATE` | Enriches existing node with new detail |
| `RELATE` | Contradiction detected — both sides preserved as a queryable tension |
| `RESOLVE` | Blocker condition changed — downstream decisions unblocked |
| `SKIP` | Exact duplicate, no action |

You can't audit a deletion. You can query a tension.

---

## Audit Trail

Every mutation is SHA-256 hash-chained, append-only, crash-safe. Verify the full chain in one call:

```python
from flowscript_agents import Memory, MemoryOptions, AuditConfig

mem = Memory.load_or_create("agent.json",
    options=MemoryOptions(audit=AuditConfig(retention_months=84)))

# ... agent does work ...

result = Memory.verify_audit("agent.audit.jsonl")
# → AuditVerifyResult(valid=True, total_entries=42, files_verified=1)
```

Framework attribution is automatic — every audit entry records which adapter triggered it. Query by time range, event type, adapter, or session. Rotation with gzip compression. `on_event` callback for SIEM integration. [Full audit trail docs →](docs/audit-trail.md)

---

## Memory That Evolves

Nodes graduate through four temporal tiers based on actual use — `current` → `developing` → `proven` → `foundation`. Every query touches returned nodes, so knowledge that keeps getting queried earns its place. One-off observations fade naturally. Dormant nodes are pruned to the audit trail — archived with full provenance, never destroyed.

After 20 sessions, your memory is a curated knowledge base, not a pile of notes. [Session lifecycle details →](docs/lifecycle.md)

---

## Comparison

| | FlowScript | Mem0 | Vector stores |
|:---|:---|:---|:---|
| Find similar content | Vector search | Vector search | Vector search |
| "Why did we decide X?" | `why()` — typed causal chain | — | — |
| "What's blocking?" | `blocked()` — downstream impact | — | — |
| "What tradeoffs?" | `tensions()` — named axes | — | — |
| "What if we change this?" | `whatIf()` — impact analysis | — | — |
| Contradictions | `RELATE` — both sides preserved | `DELETE` — replaced | N/A |
| Audit trail | SHA-256 hash chain | — | — |
| Temporal graduation | Automatic 4-tier | — | — |
| Token budgeting | 4 strategies | — | — |

Under the hood: a local semantic graph with typed nodes, typed relationships, and typed states. Queries traverse structure — no embeddings required, no LLM calls, no network. Sub-ms on project-scale graphs. Vector search and reasoning queries are orthogonal — use both.

---

## Ecosystem

| Package | What | Install |
|:--------|:-----|:--------|
| [flowscript-agents](https://pypi.org/project/flowscript-agents/) | Python SDK — 9 adapters, unified memory, consolidation, audit trail | `pip install flowscript-agents` |
| [flowscript-core](https://www.npmjs.com/package/flowscript-core) | TypeScript SDK — Memory class, 15 tools, token budgeting, audit trail | `npm install flowscript-core` |
| [flowscript.org](https://flowscript.org) | Web editor, D3 visualization, live query panel | Browser |

**1,272 tests** across Python (581) and TypeScript (691). Same audit trail format and canonical JSON serialization across both languages.

### Docs

- [API Reference](docs/api-reference.md) — Memory, UnifiedMemory, AuditConfig, queries
- [Framework Adapters](docs/adapters.md) — per-framework examples and integration guides
- [Audit Trail](docs/audit-trail.md) — configuration, SIEM integration, compliance
- [Session Lifecycle](docs/lifecycle.md) — temporal tiers, persistence, multi-session patterns

---

MIT. Built by [Phillip Clapham](https://phillipclapham.com).
