# flowscript-agents

**Drop-in reasoning memory for AI agent frameworks.**

[![Tests](https://img.shields.io/badge/tests-409%20passing-brightgreen)](https://github.com/phillipclapham/flowscript-agents) [![PyPI](https://img.shields.io/pypi/v/flowscript-agents)](https://pypi.org/project/flowscript-agents/) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/flowscript-agents/)

---

## The Problem

Agent memory today is vector search over blobs. Your agent made a decision — why? What's blocking it? What tradeoffs did it weigh? Embeddings can't answer that.

**flowscript-agents** replaces flat memory with queryable reasoning for LangGraph, CrewAI, Google ADK, and OpenAI Agents SDK. Same interfaces your framework expects, but now `memory.query.tensions()` actually works.

Built on [flowscript-core](https://www.npmjs.com/package/flowscript-core) (TypeScript SDK) and [flowscript-ldp](https://pypi.org/project/flowscript-ldp/) (Python IR + query engine).

---

## Install

```bash
# Core (framework-agnostic Memory class)
pip install flowscript-agents

# With your framework
pip install flowscript-agents[langgraph]
pip install flowscript-agents[crewai]
pip install flowscript-agents[google-adk]
pip install flowscript-agents[openai-agents]

# Everything
pip install flowscript-agents[all]
```

---

## Quick Start (Framework-Agnostic)

The `Memory` class works standalone — no framework required.

```python
from flowscript_agents import Memory

mem = Memory()

q = mem.question("Which database for agent sessions?")
mem.alternative(q, "Redis").decide(rationale="speed critical")
mem.alternative(q, "SQLite").block(reason="no concurrent writes")
mem.tension(
    mem.thought("Redis gives sub-ms reads"),
    mem.thought("cluster costs $200/mo"),
    axis="performance vs cost"
)

# Semantic queries — the thing no other memory gives you
print(mem.query.tensions())       # tradeoffs with named axes
print(mem.query.blocked())        # blockers + downstream impact
print(mem.query.alternatives(q.id))  # options + their states

# Persist
mem.save("./agent-memory.json")

# Next session
mem2 = Memory.load_or_create("./agent-memory.json")
```

---

## LangGraph

Drop-in `BaseStore` implementation. Use as your LangGraph store — every item becomes a queryable FlowScript node.

```python
from flowscript_agents.langgraph import FlowScriptStore

store = FlowScriptStore("./agent-memory.json")

# Standard LangGraph store operations
store.put(("agents", "planner"), "db_decision", {"value": "chose Redis for speed"})
items = store.search(("agents", "planner"), query="Redis")

# FlowScript queries on the same data
blockers = store.memory.query.blocked()
tensions = store.memory.query.tensions()

# Async support included
items = await store.aget(("agents",), "key")
await store.aput(("agents",), "key", {"value": "data"})
```

**Install:** `pip install flowscript-agents[langgraph]`

---

## CrewAI

Duck-typed `StorageBackend` — plug into CrewAI's memory system.

```python
from flowscript_agents.crewai import FlowScriptStorage

storage = FlowScriptStorage("./crew-memory.json")

# Standard CrewAI storage operations
storage.save({"content": "User prefers concise answers", "score": 0.9})
results = storage.search("user preferences", limit=5)

# Scoped storage
storage.save({"content": "API rate limit hit"}, metadata={"scope": "errors"})
scoped = storage.search("rate limit", scope="errors")

# FlowScript queries
tensions = storage.memory.query.tensions()
blockers = storage.memory.query.blocked()
```

**Install:** `pip install flowscript-agents[crewai]`

---

## Google ADK

`BaseMemoryService` implementation for ADK agents.

```python
from flowscript_agents.google_adk import FlowScriptMemoryService

memory_service = FlowScriptMemoryService("./adk-memory.json")

# Use with ADK Runner
# runner = Runner(agent=agent, memory_service=memory_service, ...)

# Session events are automatically extracted as FlowScript nodes
await memory_service.add_session_to_memory(session)

# Search enriched with FlowScript query results
results = await memory_service.search_memory("my-app", "user-1", "database decision")
# Results include tensions, blockers when search matches reasoning patterns

# Direct query access
tensions = memory_service.memory.query.tensions()
```

**Install:** `pip install flowscript-agents[google-adk]`

---

## OpenAI Agents SDK

Session protocol implementation for the OpenAI Agents SDK.

```python
from flowscript_agents.openai_agents import FlowScriptSession

session = FlowScriptSession("conversation_123", "./openai-memory.json")

# Standard session operations
session.add_items([
    {"role": "user", "content": "Which database should we use?"},
    {"role": "assistant", "content": "I recommend Redis for the speed requirement."}
])
history = session.get_items(limit=10)

# FlowScript queries on conversation reasoning
tensions = session.memory.query.tensions()
blockers = session.memory.query.blocked()
```

**Install:** `pip install flowscript-agents[openai-agents]`

---

## What You Get That Vector Memory Doesn't

| Capability | Vector stores | flowscript-agents |
|:-----------|:-------------|:-----------------|
| "Why did we decide X?" | Dig through logs | `memory.query.why(node_id)` |
| "What's blocking progress?" | Hope you logged it | `memory.query.blocked()` |
| "What tradeoffs exist?" | Good luck | `memory.query.tensions()` |
| "What alternatives were considered?" | Not tracked | `memory.query.alternatives(q_id)` |
| "What if we remove this?" | Rebuild from scratch | `memory.query.what_if(node_id)` |
| Human-readable export | JSON blobs | `.fs` files your PM can read |

These aren't complementary to embeddings — they're orthogonal. Use both: vector search for "find similar," FlowScript for "understand reasoning."

---

## API Reference

### Memory (core)

```python
from flowscript_agents import Memory

mem = Memory()                           # new empty
mem = Memory.load("./memory.json")       # from file
mem = Memory.load_or_create("./mem.json") # zero-friction entry

# Build reasoning
node = mem.thought("content")            # also: statement, question, action, insight, completion
alt = mem.alternative(question, "option") # linked to question
node.causes(other)                       # causal relationship
node.tension_with(other, axis="speed vs cost")
node.decide(rationale="reason")          # state: decided
node.block(reason="why")                 # state: blocked
node.unblock()                           # remove blocked state

# Query
mem.query.why(node_id)                   # causal chain
mem.query.tensions()                     # all tensions with axes
mem.query.blocked()                      # all blockers + impact
mem.query.alternatives(question_id)      # options + states
mem.query.what_if(node_id)               # downstream impact

# Persist
mem.save("./memory.json")               # atomic write
mem.save()                               # re-save to loaded path
```

### Adapters

| Framework | Class | Interface |
|:----------|:------|:----------|
| LangGraph | `FlowScriptStore` | `BaseStore` (get/put/search/delete + async) |
| CrewAI | `FlowScriptStorage` | `StorageBackend` (save/search/update/delete + scopes) |
| Google ADK | `FlowScriptMemoryService` | `BaseMemoryService` (add_session/search_memory) |
| OpenAI Agents | `FlowScriptSession` | `Session` (get_items/add_items/pop_item/clear) |
| Pydantic AI | `FlowScriptDeps` | Deps + `create_memory_tools()` |
| smolagents | `FlowScriptMemory` | 5 Tool-protocol classes |
| LlamaIndex | `FlowScriptMemoryBlock` | `BaseMemoryBlock[str]` (get/put/truncate) |
| Haystack | `FlowScriptMemoryStore` | `MemoryStore` (add/search/delete) |
| CAMEL-AI | `FlowScriptCamelMemory` | `AgentMemory` (retrieve/write/get_context/clear) |

All adapters expose `.memory` for direct FlowScript query access and support context managers (`with` blocks).

---

## Session Lifecycle

Memory gets smarter every time you use it. Every query touches returned nodes — incrementing frequency and updating timestamps. Nodes that keep getting queried graduate through tiers:

`current` → `developing` → `proven` → `foundation`

After 5 sessions, your most-used decisions load first. After 20, your memory is a curated knowledge base, not a pile of notes. Dormant nodes get pruned to an audit trail — nothing is deleted, just archived.

### The `with` Pattern (Recommended)

All adapters and `UnifiedMemory` support context managers. Session lifecycle is automatic:

```python
from flowscript_agents.langgraph import FlowScriptStore

with FlowScriptStore("./agent-memory.json") as store:
    # session_start() called automatically on construction
    store.put(("agents",), "decision", {"value": "chose Redis"})
    results = store.search(("agents",), query="database")
    # ... your agent does work ...
# close() called automatically: prunes dormant nodes + saves to disk
```

This works the same way for all adapters:

```python
with FlowScriptStorage("./memory.json") as storage:     # CrewAI
with FlowScriptMemoryService("./memory.json") as svc:    # Google ADK
with FlowScriptSession("id", "./memory.json") as sess:   # OpenAI Agents
with FlowScriptDeps("./memory.json") as deps:            # Pydantic AI
with FlowScriptMemory("./memory.json") as mem:           # smolagents
with FlowScriptMemoryStore("./memory.json") as store:    # Haystack
with FlowScriptCamelMemory("./memory.json") as mem:      # CAMEL-AI
```

### Manual Lifecycle

If you can't use `with` blocks (long-running services, notebooks):

```python
store = FlowScriptStore("./agent-memory.json")
# session_start() called automatically on construction

# ... work happens, auto-saves on each put/add ...

# End of session: call close() explicitly
store.close()  # prunes dormant nodes + saves
```

All adapters auto-save on `put`/`save`/`add_items` operations, so data is never lost between calls. `close()` adds the pruning step that keeps memory healthy over time.

---

## Ecosystem

- **[flowscript-core](https://www.npmjs.com/package/flowscript-core)** — TypeScript SDK with `Memory` class, `asTools()` (12 OpenAI-format tools), token budgeting, audit trail
- **[flowscript-ldp](https://pypi.org/project/flowscript-ldp/)** — LDP Mode 3 reference implementation (historical artifact, v0.2.1 frozen)
- **[flowscript.org](https://flowscript.org)** — Web editor, D3 visualization, live query panel

---

## License

MIT
