<p align="center">
  <img src="docs/brand/logo-512.png" alt="FlowScript" width="120" />
</p>

<h1 align="center">flowscript-agents</h1>

<p align="center"><strong>Your AI agents make decisions they can't explain.<br>FlowScript makes those decisions queryable.</strong></p>

<p align="center"><em>Vector stores remember what. FlowScript remembers why.</em></p>

<p align="center">
  <a href="https://pypi.org/project/flowscript-agents/"><img src="https://img.shields.io/pypi/v/flowscript-agents" alt="PyPI"></a>
  <a href="https://github.com/phillipclapham/flowscript-agents"><img src="https://img.shields.io/badge/tests-717%20passing-brightgreen" alt="Tests"></a>
  <a href="https://pypi.org/project/flowscript-agents/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
</p>

---

Your agent chose PostgreSQL three sessions ago. Now it's recommending Redis. Why did it change its mind? Without reasoning memory, you can't know.

```python
# Your agent stores reasoning as it works:
add_memory("Chose PostgreSQL — $15/mo, ACID compliant. Redis eliminated: $200/mo exceeds $50 budget.")
add_memory("Revisiting: Redis latency (sub-ms) may justify the cost for real-time features.")

# Sessions later, you ask "why?" and get the actual reasoning chain:
query_why("chose-postgresql")
# → budget_constraint ($50/mo limit)
#   → eliminated Redis ($200/mo cluster)
#     → chose PostgreSQL ($15/mo, ACID compliant)

query_tensions()
# → performance vs cost
#     Redis: sub-ms reads, $200/month
#     PostgreSQL: 10-50ms, $15/month
#     constraint: $50/month infrastructure budget
```

Six typed reasoning queries. Nine framework adapters. Session memory that learns across conversations. And when memories contradict, FlowScript doesn't delete — it creates a queryable [tension](#when-memories-contradict).

<p align="center">
  <img src="https://raw.githubusercontent.com/phillipclapham/flowscript/main/docs/flowscript-demo.png" alt="FlowScript — reasoning graph visualization with typed nodes and query results" width="800">
</p>

---

## Install

```bash
pip install flowscript-agents
```

## Quick Start: MCP Server

Add FlowScript to Claude Code, Cursor, Windsurf, or any MCP-compatible editor.

**Claude Code** — add to `.claude/settings.json`:

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

**Cursor / Windsurf / VS Code** — add to `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "flowscript": {
      "type": "stdio",
      "command": "flowscript-mcp",
      "args": ["--memory", "./project-memory.json"],
      "env": {
        "OPENAI_API_KEY": "your-key"
      }
    }
  }
}
```

Restart your editor. You now have [20 MCP tools](#tools-at-a-glance) for reasoning memory.

**Then add the [CLAUDE.md snippet](examples/CLAUDE.md.example) to your project** — this tells your agent *when* to record decisions and surface tensions. Without it, the tools are available but passive. With it, your agent proactively tracks your project's reasoning.

---

## What You Get

**With an API key**, plain text is auto-extracted into typed reasoning nodes. Vector search finds memories by meaning. Contradictions become queryable tensions instead of silent overwrites.

| API Key | What auto-configures |
|:--------|:--------------------|
| `OPENAI_API_KEY` | Typed extraction (gpt-4o-mini) + vector search (text-embedding-3-small) + consolidation |
| `ANTHROPIC_API_KEY` | Typed extraction (claude-haiku) + consolidation. Keyword search (no embeddings). |
| Neither | Raw text storage + keyword search. Tools work, but no typed extraction or vector search. |

**Add session persistence** with one env var. Your agent's memory is compressed at session boundaries — patterns that recur get promoted, noise fades naturally:

```json
"env": {
  "OPENAI_API_KEY": "your-key",
  "FLOWSCRIPT_CONTINUITY": "true"
}
```

**Local embeddings (free, no API key for embeddings):**

| Provider | Install | Flag |
|:---------|:--------|:-----|
| Ollama | [Install Ollama](https://ollama.com) + `ollama pull nomic-embed-text` | `--embedder ollama --embedding-model nomic-embed-text` |
| SentenceTransformers | `pip install sentence-transformers` | `--embedder sentence-transformers --embedding-model BAAI/bge-m3` |

You still need an LLM API key for typed extraction, even with local embeddings.

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

Every other memory system handles contradictions by deleting. Mem0's consolidation uses ADD/UPDATE/DELETE — when facts contradict, the old one is replaced. LangGraph's langmem does the same.

FlowScript doesn't delete. It **relates**.

When consolidation detects a contradiction, it creates a tension with a named axis. Both perspectives survive. The disagreement itself becomes queryable knowledge. Call `query_tensions()` to see every active tradeoff your agent is navigating.

---

## Tools at a Glance

20 MCP tools, grouped by purpose:

| Group | Tools |
|:------|:------|
| **Memory** | `add_memory`, `search_memory`, `get_context`, `remove_memory`, `memory_stats` |
| **Reasoning Queries** | `query_tensions`, `query_blocked`, `query_why`, `query_alternatives`, `query_what_if`, `query_counterfactual` |
| **Session** | `session_wrap`, `encode_exchange` |
| **Compliance** | `explain_decision`, `query_audit`, `verify_audit`, `verify_integrity` |
| **Thinking** | `think_deeper`, `think_creative`, `think_breakthrough` |

---

## Works With Your Stack

Each adapter implements your framework's native memory interface — `BaseStore` for LangGraph, `StorageBackend` for CrewAI, and so on. You don't learn a new API. You get reasoning queries on top of the one you already use.

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

All adapters expose `.memory` for direct query access and support `with` blocks for automatic session lifecycle. Install everything: `pip install flowscript-agents[all]`.

---

## Session Lifecycle

```
MCP server starts → session_start (automatic)
  → agent works, calls add_memory / encode_exchange
    → session_wrap (explicit call, or auto after 5 min idle)
      → dormant nodes pruned to audit trail
      → continuity compressed (if enabled)
      → memory saved
```

**Session wraps are where the magic happens.** When continuity is enabled, compression isn't just storage optimization — it's how your agent decides what matters. Patterns that recur across sessions get promoted through temporal tiers. One-off observations fade. After 20 sessions, you have curated project knowledge, not a pile of notes.

---

## Configuration

| Setting | Type | Default | Description |
|:--------|:-----|:--------|:------------|
| `OPENAI_API_KEY` | env | — | Enables extraction (gpt-4o-mini) + embeddings (text-embedding-3-small) |
| `ANTHROPIC_API_KEY` | env | — | Enables extraction (claude-haiku). No embeddings. |
| `FLOWSCRIPT_CONTINUITY` | env | `false` | Session memory compression across conversations |
| `FLOWSCRIPT_AUTO_WRAP_MINUTES` | env | `5` | Auto-wrap after N minutes idle. `0` to disable. |
| `--memory PATH` | flag | required | Path to memory JSON file |
| `--embedder PROVIDER` | flag | auto | `openai`, `sentence-transformers`, or `ollama` |
| `--embedding-model MODEL` | flag | `text-embedding-3-small` | Embedding model (provider-specific) |
| `--llm-model MODEL` | flag | `gpt-4o-mini` | LLM for extraction and consolidation |
| `--no-auto` | flag | off | Disable auto-configuration from API keys |

---

## Files FlowScript Creates

All sidecar files are co-located with your memory file:

| File | Purpose |
|:-----|:--------|
| `*.json` | Reasoning graph — nodes, relationships, temporal metadata |
| `*.continuity.md` | Compressed session memory (when `FLOWSCRIPT_CONTINUITY` enabled) |
| `*.continuity.meta.json` | Session count + citation tracking metadata |
| `*.embeddings.json` | Vector index for semantic search |
| `*.audit.jsonl` | Hash-chained audit trail (append-only, SHA-256) |

---

## Audit Trail

Every mutation to the reasoning graph is recorded in a hash-chained audit trail. Tamper-evident: modify any entry and the chain breaks. Queryable via `query_audit`, verifiable via `verify_audit`.

Default retention: 7 years (SOX-compatible). Monthly gzip rotation with manifest index. `on_event` callbacks for SIEM integration.

---

## SDK Wrapper

EU AI Act enforcement begins August 2026. For organizations deploying autonomous AI agents, FlowScript wraps your existing SDK client at the transport layer — two lines of code, every API call is captured, extracted into a queryable reasoning graph, and backed by a hash-chained audit trail. The agent never knows.

```python
from flowscript_agents.wrapper import FlowScriptOpenAI

client = FlowScriptOpenAI()  # wraps your OpenAI client
# Every exchange is now extracted, auditable, and queryable.
```

Structured extraction with `why()` and `query_counterfactual()` satisfies Articles 12, 13, and 86 of the EU AI Act — and [CJEU C-203/22](https://curia.europa.eu/juris/liste.jsf?num=C-203/22) requirements for counterfactual explanations. Wrapper validation complete — memory integration shipping this week.

---

## FlowScript Cloud

Independent cryptographic witnessing for your agent's reasoning chains. Live at [`api.flowscript.org`](https://api.flowscript.org/v1/health).

Your local audit trail proves what happened. FlowScript Cloud proves it independently — chain verification on ingestion, witness attestations, and compliance export.

```python
from flowscript_agents.cloud import CloudClient

cloud = CloudClient(api_key="fsk_...", namespace="myorg/agent")
# Wire into audit trail — events auto-forwarded to Cloud for witnessing
```

---

## TypeScript SDK

For Node.js developers: [flowscript-core](https://www.npmjs.com/package/flowscript-core) provides the same reasoning memory with a fluent builder API, 15 agent tools in OpenAI function calling format, and a built-in MCP server.

```bash
npm install flowscript-core
```

See the [FlowScript repo](https://github.com/phillipclapham/flowscript) for the TypeScript SDK, web editor, D3 visualization, and the FlowScript notation spec.

---

## Links

- **[flowscript.org](https://flowscript.org)** — Web editor with D3 reasoning graph visualization
- **[FlowScript (TypeScript)](https://github.com/phillipclapham/flowscript)** — Core SDK, parser, notation spec
- **[FlowScript Cloud](https://github.com/phillipclapham/flowscript-cloud)** — Compliance witnessing service (BSL 1.1)
- **[CLAUDE.md snippet](examples/CLAUDE.md.example)** — Paste into your project to activate proactive reasoning
- **[API Reference](docs/api-reference.md)** — Full Python SDK documentation
- **[Adapter Guide](docs/adapters.md)** — Per-framework integration examples

---

<p align="center">
  Built by <a href="https://phillipclapham.com">Phill Clapham</a> · <a href="https://claphamdigital.com">Clapham Digital LLC</a>
</p>
