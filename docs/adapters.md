# Framework Adapters

All adapters implement their framework's native interface. Same API you already use — plus `.memory` for direct FlowScript query access.

Every adapter:
- Supports context managers (`with` blocks) for automatic session lifecycle
- Accepts optional `embedder`, `llm`, and `consolidation_provider` for vector search and extraction
- Exposes `.memory` for direct query access (`tensions()`, `blocked()`, `why()`, etc.)
- Auto-saves on mutations (`put`/`save`/`add_items`)
- Calls `close()` on `__exit__` (prunes dormant nodes, saves to disk)

---

## LangGraph

Drop-in `BaseStore` implementation. Every item becomes a queryable FlowScript node.

```bash
pip install flowscript-agents[langgraph]
```

```python
from flowscript_agents.langgraph import FlowScriptStore

with FlowScriptStore("agent-memory.json") as store:
    # Standard LangGraph store operations
    store.put(("agents", "planner"), "db_decision", {"value": "chose Redis for speed"})
    items = store.search(("agents", "planner"), query="Redis")

    # Async support
    items = await store.aget(("agents",), "key")
    await store.aput(("agents",), "key", {"value": "data"})

    # FlowScript queries on the same data
    tensions = store.memory.query.tensions()
    blockers = store.memory.query.blocked()

    # Resolve a store key to its reasoning node
    node = store.resolve(("agents", "planner"), "db_decision")
    # → NodeRef with full graph context: causes, tensions, state
```

---

## CrewAI

`StorageBackend` implementation with scoped storage support.

```bash
pip install flowscript-agents[crewai]
```

```python
from flowscript_agents.crewai import FlowScriptStorage

with FlowScriptStorage("agent-memory.json") as storage:
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

---

## Google ADK

`BaseMemoryService` implementation for ADK agents.

```bash
pip install flowscript-agents[google-adk]
```

```python
from flowscript_agents.google_adk import FlowScriptMemoryService

with FlowScriptMemoryService("agent-memory.json") as svc:
    # Session events extracted as FlowScript nodes
    await svc.add_session_to_memory(session)

    # Search enriched with reasoning context
    results = await svc.search_memory("my-app", "user-1", "database decision")

    # Direct query access
    tensions = svc.memory.query.tensions()
```

---

## OpenAI Agents SDK

Session protocol implementation.

```bash
pip install flowscript-agents[openai-agents]
```

```python
from flowscript_agents.openai_agents import FlowScriptSession

with FlowScriptSession("conv_123", "agent-memory.json") as session:
    session.add_items([
        {"role": "user", "content": "Which database should we use?"},
        {"role": "assistant", "content": "Redis for the speed requirement."}
    ])
    history = session.get_items(limit=10)

    tensions = session.memory.query.tensions()
```

---

## Pydantic AI

Deps with auto-generated memory tools.

```bash
pip install flowscript-agents[pydantic-ai]
```

```python
from flowscript_agents.pydantic_ai import FlowScriptDeps

with FlowScriptDeps("agent-memory.json") as deps:
    # Create tools for your Pydantic AI agent
    tools = deps.create_memory_tools()

    tensions = deps.memory.query.tensions()
```

---

## smolagents

Five Tool-protocol classes for HuggingFace's smolagents.

```bash
pip install flowscript-agents[smolagents]
```

```python
from flowscript_agents.smolagents import FlowScriptMemory

with FlowScriptMemory("agent-memory.json") as mem:
    # Tools available for smolagent tool use
    tensions = mem.memory.query.tensions()
```

---

## LlamaIndex

`BaseMemoryBlock[str]` implementation.

```bash
pip install flowscript-agents[llamaindex]
```

```python
from flowscript_agents.llamaindex import FlowScriptMemoryBlock

with FlowScriptMemoryBlock("agent-memory.json") as block:
    block.put("User prefers detailed explanations")
    context = block.get()

    tensions = block.memory.query.tensions()
```

**Note:** LlamaIndex's default flush threshold (0.7) means ~21k tokens before memory flushes to the agent. For demos or interactive testing, set a lower threshold (0.1-0.2).

---

## Haystack

`MemoryStore` implementation. FlowScript is the second Haystack MemoryStore implementation (after Mem0) and the first with reasoning queries.

```bash
pip install flowscript-agents[haystack]
```

```python
from flowscript_agents.haystack import FlowScriptMemoryStore

with FlowScriptMemoryStore("agent-memory.json") as store:
    store.add({"content": "Database migration completed successfully"})
    results = store.search("database", limit=5)

    tensions = store.memory.query.tensions()
```

---

## CAMEL-AI

`AgentMemory` implementation. Note: `clear()` is a deliberate no-op — reasoning memory persists through ChatAgent lifecycle, unlike built-in memories that reset.

```bash
pip install flowscript-agents[camel-ai]
```

```python
from flowscript_agents.camel_ai import FlowScriptCamelMemory

with FlowScriptCamelMemory("agent-memory.json") as mem:
    mem.write_records([record])
    context = mem.get_context_creator()

    tensions = mem.memory.query.tensions()
```

---

## Adding Vector Search to Adapters

All adapters accept `embedder`, `llm`, and `consolidation_provider` as constructor kwargs:

```python
from flowscript_agents.langgraph import FlowScriptStore
from flowscript_agents.embeddings import OpenAIEmbeddings

with FlowScriptStore("agent-memory.json", embedder=OpenAIEmbeddings()) as store:
    # Now store.search() uses vector similarity instead of keyword matching
    items = store.search(("agents",), query="fast database options")
```

This works identically across all 9 adapters.
