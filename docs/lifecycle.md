# Session Lifecycle

## Temporal Tiers

Nodes graduate through four tiers based on how often they're actually queried:

| Tier | What it means | Behavior |
|:-----|:--------------|:---------|
| `current` | Recent observation | Pruned if not reinforced |
| `developing` | Emerging pattern (2+ queries) | Building confidence |
| `proven` | Validated through use (3+ queries) | Protected from pruning |
| `foundation` | Core truth | Always preserved, even under budget pressure |

Every query — `why()`, `tensions()`, `blocked()`, `alternatives()`, `whatIf()` — touches the returned nodes, incrementing frequency and updating timestamps. Knowledge that keeps getting queried earns its place. One-off observations fade naturally.

```python
report = mem.garden()  # GardenReport dataclass
print(report.stats)    # {"total": 39, "growing": 12, "resting": 8, "dormant": 4}
print(report.growing)  # list of node IDs in growing tier
print(report.dormant)  # candidates for pruning
```

Dormant nodes are pruned to the audit trail during `close()` or `sessionWrap()` — archived with full hash-chain provenance, never destroyed.

## The `with` Pattern (Recommended)

All adapters and `UnifiedMemory` support context managers:

```python
from flowscript_agents.langgraph import FlowScriptStore

with FlowScriptStore("agent-memory.json") as store:
    # session_start() called on entry
    store.put(("agents",), "key", {"value": "data"})
    # ... agent does work ...
# close() on exit: prunes dormant nodes, saves to disk
```

Works identically across all adapters:

```python
with FlowScriptStorage("memory.json") as storage:       # CrewAI
with FlowScriptMemoryService("memory.json") as svc:      # Google ADK
with FlowScriptSession("id", "memory.json") as sess:     # OpenAI Agents
with FlowScriptDeps("memory.json") as deps:              # Pydantic AI
with FlowScriptMemory("memory.json") as mem:             # smolagents
with FlowScriptMemoryStore("memory.json") as store:      # Haystack
with FlowScriptCamelMemory("memory.json") as mem:        # CAMEL-AI
```

## Manual Lifecycle

For long-running services or notebooks where `with` blocks don't work:

```python
store = FlowScriptStore("agent-memory.json")
# session_start() called automatically on construction

# ... work happens, auto-saves on each put/add ...

# End of session: call close() explicitly
store.close()  # prunes dormant nodes + saves
```

All adapters auto-save on mutation operations (`put`/`save`/`add_items`), so data is never lost between calls. `close()` adds the pruning step that keeps memory healthy over time.

## Multi-Session Warning

**NodeRef objects do not survive persistence boundaries.**

After `load_or_create()`, any `NodeRef` from the previous Memory instance still points to the old object. Using stale refs corrupts the audit hash chain (stale sequence numbers, wrong previous hash, old adapter context).

```python
# ❌ WRONG — stale reference after reload
node = mem.thought("important decision")
mem.save("agent-memory.json")
mem = Memory.load_or_create("agent-memory.json")
node.decide(rationale="...")  # 'node' points to OLD Memory → corrupted chain

# ✅ RIGHT — get fresh references after reload
mem = Memory.load_or_create("agent-memory.json")
node = next(n for n in mem.nodes if "important decision" in n.content)
node.decide(rationale="...")  # correct Memory instance
```

This is the #1 footgun for multi-session usage. After any `load_or_create()` call, always get fresh `NodeRef` handles from the reloaded Memory.

## Touch-on-Query

By default, all five query methods touch returned nodes (incrementing `frequency` and updating `lastTouched`). This drives graduation — knowledge that keeps getting queried earns its place.

Disable with `touchOnQuery=False` on `MemoryOptions` for read-only analysis:

```python
mem = Memory.load_or_create("file.json",
    options=MemoryOptions(touch_on_query=False))
```

## Session Start Deduplication

`sessionStart()` calls both `blocked()` and `tensions()` internally (for the orientation summary). Touches from these calls are deduplicated — nodes aren't double-touched just because they appeared in both query results.
