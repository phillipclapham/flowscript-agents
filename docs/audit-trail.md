# Audit Trail

Every mutation to the reasoning graph is recorded in a SHA-256 hash-chained, append-only audit trail. Each entry contains the previous entry's hash — creating a verifiable chain of reasoning provenance.

## Configuration

```python
from flowscript_agents import Memory, MemoryOptions, AuditConfig

config = AuditConfig(
    retention_months=84,        # default: 84 (7 years, SOX). Set to None to disable cleanup.
    rotation="monthly",         # monthly | weekly | daily | size (size-based at ~10MB)
    verbosity="standard",       # standard (mutations only) | full (mutations + reads, for HIPAA)
    on_event=my_callback,       # receives each audit entry dict after it's written to disk
)

mem = Memory.load_or_create("agent-memory.json",
    options=MemoryOptions(audit=config))
```

**Important:** `AuditConfig` is not serialized to the JSON file (callbacks can't be serialized). You must re-supply it on every `load_or_create()` call. The config is applied even on the load path.

## Event Types

### Python (13 events)
`node_create`, `node_update`, `node_merge`, `node_remove`, `relationship_create`, `state_change`, `graduation`, `prune`, `session_start`, `session_end`, `session_wrap`, `consolidation`, `audit_cleanup`

### TypeScript (14 events)
`node_create`, `relationship_create`, `state_change`, `modifier_add`, `session_start`, `session_end`, `session_wrap`, `graduation`, `prune`, `snapshot`, `restore`, `transcript_extract`, `budget_apply`, `audit_cleanup`

## Framework Attribution

Every audit entry automatically records which adapter triggered it. When using any framework adapter, the adapter context is set on the underlying Memory:

```python
from flowscript_agents.langgraph import FlowScriptStore

with FlowScriptStore("agent-memory.json",
    options=MemoryOptions(audit=AuditConfig())) as store:
    store.put(("agents", "planner"), "key", {"value": "data"})
    # → audit entry includes:
    # {
    #   "adapter": {
    #     "framework": "langgraph",
    #     "class": "FlowScriptStore",
    #     "operation": "put"
    #   }
    # }

# Query by adapter
entries = Memory.query_audit("agent-memory.audit.jsonl", adapter="langgraph")
```

## Verify and Query

Both are static methods on `Memory`:

```python
# Verify the full hash chain
result = Memory.verify_audit("agent-memory.audit.jsonl")
# → AuditVerifyResult(valid=True, total_entries=42, files_verified=1, legacy_entries=0)

# Query with filters
result = Memory.query_audit("agent-memory.audit.jsonl",
    events=["node_create", "state_change"],
    after="2026-01-01",
    adapter="langgraph",
    session_id="sess_abc",
    limit=100)
# → AuditQueryResult(entries=[...], total_scanned=42, files_searched=1)
```

## SIEM Integration

Stream audit events to Datadog, Splunk, or any monitoring system via the `on_event` callback:

```python
import requests

def send_to_datadog(entry: dict) -> None:
    requests.post("https://http-intake.logs.datadoghq.com/api/v2/logs",
        headers={"DD-API-KEY": "your-key"},
        json={"message": entry, "service": "flowscript-agent"})

mem = Memory.load_or_create("agent-memory.json",
    options=MemoryOptions(
        audit=AuditConfig(on_event=send_to_datadog)
    ))
```

The callback receives the full audit entry dict **after** it's written to disk. Audit persistence always takes priority over callback delivery. Callback exceptions are logged to stderr but never block the audit write.

## Rotation and Compression

Rotated files are gzip-compressed with a manifest index (`{stem}.audit.manifest.json`) tracking time ranges, entry counts, file hashes, and cross-file chain continuity.

| Rotation | File naming | Use case |
|:---------|:-----------|:---------|
| `monthly` | `agent.audit.2026-03.jsonl.gz` | Default, good for most use cases |
| `weekly` | `agent.audit.2026-W12.jsonl.gz` | High-volume agents |
| `daily` | `agent.audit.2026-03-21.jsonl.gz` | Very high volume |
| `size` | `agent.audit.0001.jsonl.gz` | Size-based (~10MB per file) |

## Retention Cleanup

Configure `retention_months` to automatically clean up old audit files. Cleanup runs on rotation and writes an `audit_cleanup` event recording what was removed. Set to `None` to keep everything forever.

## Competitive Comparison

| | FlowScript | Mem0 | Zep | LangGraph |
|:---|:---|:---|:---|:---|
| Hash-chained audit | SHA-256, append-only | — | — | — |
| Mutation provenance | All paths audited | — | — | Checkpoints only |
| Framework attribution | Per-entry (adapter, class, operation) | — | — | — |
| SIEM callback | `on_event` hook | — | — | — |
| Configurable retention | Monthly/weekly/daily/size rotation | — | — | — |
| Chain verification | One-liner static method | — | — | — |
| Cross-language format | Python + TypeScript (same canonical JSON) | N/A | N/A | N/A |

## Cross-Language Compatibility

Python and TypeScript audit trails use identical file formats and canonical JSON serialization (`json.dumps(sort_keys=True, separators=(",",":"), ensure_ascii=True)` in Python, equivalent `canonicalStringify()` in TypeScript with Unicode escaping). An audit chain started in Python can be verified in TypeScript and vice versa.

**Known limitation:** JavaScript cannot distinguish `1` from `1.0` (float serialization). All current audit fields use integers and strings. If future audit payloads include Python floats serialized as `X.0`, cross-language chain verification may fail for those entries.

## Privacy Note

The audit trail records node content and metadata. If your agent processes sensitive data (PII, medical, financial), apply the same data handling policies to `.audit.jsonl` files as to the memory file itself. Encryption at rest is deferred to v2 — most production environments handle this at the storage layer (encrypted filesystems, S3 server-side encryption).

## Design Constraints

- **Single-writer only.** Two processes writing the same audit file will corrupt the hash chain. Use separate memory files per agent process. File locking is planned for v2.
- **Write-first guarantee.** Audit entries are written and fsynced **before** the corresponding mutation takes effect. Worst case on crash: a duplicate entry (harmless for append-only).
- **`verbosity="full"` access auditing** is configurable but read/query events are not yet emitted. Don't claim HIPAA access audit compliance until this is implemented. Mutation auditing works fully.
