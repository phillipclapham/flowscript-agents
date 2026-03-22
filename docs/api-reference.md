# API Reference

## Memory

The core reasoning graph. No external dependencies, no API keys, sub-millisecond queries.

```python
from flowscript_agents import Memory

mem = Memory()                                # new empty
mem = Memory.load("file.json")                # from file
mem = Memory.load_or_create("file.json")      # load if exists, create if not
```

### Building Reasoning

```python
# Node types
node = mem.thought("content")                 # also: statement, question, action, insight, completion
alt = mem.alternative(question, "option")     # linked to a question node

# Relationships
node.causes(other)                            # causal relationship
node.tension_with(other, axis="speed vs cost")

# States
node.decide(rationale="reason")               # state: decided
node.block(reason="why")                      # state: blocked
node.unblock()                                # remove blocked state
```

### Queries

Five typed semantic queries that traverse the reasoning graph:

```python
mem.query.why(node_id)                        # causal chain backward
mem.query.tensions()                          # all tensions with named axes
mem.query.blocked()                           # all blockers + downstream impact
mem.query.alternatives(question_id)           # all options + their states
mem.query.what_if(node_id)                    # downstream impact analysis
```

### Lifecycle

```python
report = mem.garden()                         # GardenReport dataclass
print(report.stats)                           # {"total": 39, "growing": 12, "resting": 8, "dormant": 4}
print(report.growing)                         # list of node IDs in growing tier
print(report.dormant)                         # list of node IDs candidates for pruning
```

### Persistence

```python
mem.save("file.json")                         # save to path
mem.save()                                    # re-save to loaded path
```

---

## UnifiedMemory

Composable layers on top of Memory — vector search, auto-extraction, and consolidation.

```python
from flowscript_agents import UnifiedMemory
from flowscript_agents.embeddings import OpenAIEmbeddings

mem = UnifiedMemory(
    "file.json",
    embedder=OpenAIEmbeddings(),              # or SentenceTransformerEmbeddings(), OllamaEmbeddings()
    llm=my_extract_fn,                        # Callable[[str], str] — takes prompt, returns response
    consolidation_provider=my_provider,        # ConsolidationProvider protocol (tool_call method)
)

mem.add("plain text")                         # → typed extraction + consolidation
mem.search("query", limit=5)                  # → vector search + reasoning context
mem.memory.query.tensions()                   # → full reasoning query API

mem.close()                                   # prune + save (or use context manager)
```

### Embedding Providers

```python
from flowscript_agents.embeddings import (
    OpenAIEmbeddings,                         # cloud, best quality — requires openai package
    SentenceTransformerEmbeddings,            # local, free — requires sentence-transformers
    OllamaEmbeddings,                         # local, free, zero additional deps (raw HTTP)
)

# OpenAI (default model: text-embedding-3-small, 1536 dims, $0.02/MTok)
embedder = OpenAIEmbeddings()
embedder = OpenAIEmbeddings(model="text-embedding-3-large")

# Local (default: all-MiniLM-L6-v2, 384 dims)
embedder = SentenceTransformerEmbeddings()

# Ollama (default: nomic-embed-text, 768 dims — requires Ollama running)
embedder = OllamaEmbeddings()
```

### LLM Extraction

The `llm` parameter is any callable that takes a prompt string and returns a response string:

```python
from openai import OpenAI

client = OpenAI()

def extract(prompt: str) -> str:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content or ""

mem = UnifiedMemory("file.json", embedder=OpenAIEmbeddings(), llm=extract)
```

### Context Manager

`UnifiedMemory` supports `with` blocks. `close()` is called automatically on exit — prunes dormant nodes and saves to disk:

```python
with UnifiedMemory("file.json", embedder=OpenAIEmbeddings()) as mem:
    mem.add("content")
    results = mem.search("query")
# close() called automatically
```

---

## AuditConfig

```python
from flowscript_agents import Memory, MemoryOptions, AuditConfig

config = AuditConfig(
    retention_months=84,                      # default: 84 (7 years, SOX)
    rotation="monthly",                       # monthly | weekly | daily | size
    verbosity="standard",                     # standard (mutations) | full (+ reads)
    on_event=my_callback,                     # SIEM integration — receives each entry after write
)

mem = Memory.load_or_create("file.json", options=MemoryOptions(audit=config))
```

### Verify and Query

Both are static methods that operate on audit files directly:

```python
# Verify hash chain integrity
result = Memory.verify_audit("file.audit.jsonl")
# → AuditVerifyResult(valid=True, total_entries=42, files_verified=1, legacy_entries=0)

# Query with filters
result = Memory.query_audit("file.audit.jsonl",
    events=["node_create", "state_change"],   # filter by event type
    after="2026-01-01",                       # time range
    before="2026-12-31",
    node_id="abc123",                         # specific node
    session_id="sess_xyz",                    # specific session
    adapter="langgraph",                      # specific framework
    limit=100,                                # max results
    verify_chain=False)                       # also verify chain integrity
# → AuditQueryResult(entries=[...], total_scanned=42, files_searched=1)
```

See [Audit Trail docs](audit-trail.md) for full configuration, SIEM integration, and compliance details.
