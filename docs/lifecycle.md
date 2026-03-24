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

Dormant nodes are pruned to the audit trail during `close()` or `session_wrap()` — archived with full hash-chain provenance, never destroyed.

## Why Session Wraps Matter

Just like a mind needs sleep to consolidate memories, the reasoning graph needs regular session wraps to develop intelligence over time. A session wrap is the consolidation cycle — without it, knowledge accumulates as noise instead of maturing through the temporal tiers above.

**Three mechanisms ensure consolidation happens:**

1. **Explicit wrap** — the LLM calls `session_wrap` when the user signals session end (best results)
2. **Auto-wrap** — the MCP server auto-consolidates after 5 minutes of inactivity (configurable via `FLOWSCRIPT_AUTO_WRAP_MINUTES` env var, `0` to disable)
3. **Process exit** — a final consolidation runs automatically when the MCP server shuts down

For SDK users, all adapters call `session_wrap()` via their `close()` method or context manager exit.

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

## Writing Your Own Adapter

If you're building an adapter for a framework not yet supported, wire session wraps using this pattern:

```python
class MyFrameworkAdapter:
    def __init__(self, file_path, embedder=None, llm=None, consolidation_provider=None):
        self._memory = Memory.load_or_create(file_path)
        self._unified = UnifiedMemory(
            file_path=file_path, embedder=embedder,
            llm=llm, consolidation_provider=consolidation_provider,
        )
        self._memory.set_adapter_context("my_framework", "MyFrameworkAdapter", "init")
        self._memory.session_start()

    def close(self):
        """End the session: prune dormant nodes, save. Returns SessionWrapResult."""
        try:
            if self._unified:
                return self._unified.close()
            return self._memory.session_wrap()
        finally:
            self._memory.clear_adapter_context()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise  # close() failure IS the error when no prior exception
```

**Key points:**
- `close()` should call `session_wrap()` (via `UnifiedMemory.close()` or directly)
- `clear_adapter_context()` goes in the `finally` block AFTER `session_wrap()` — session lifecycle events need adapter attribution
- Context managers (`__enter__`/`__exit__`) make the `with` pattern work
- `set_adapter_context()` should be called on construction, then `set_adapter_operation()` per-operation for granular audit attribution
