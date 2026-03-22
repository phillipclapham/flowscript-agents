# FlowScript Audit Trail — Design Specification

**Status:** APPROVED — Ready for implementation
**Author:** Claude + Phill (flow partnership)
**Date:** March 21, 2026
**Scope:** Both flowscript-core (TypeScript) and flowscript-agents (Python)

---

## 1. Problem Statement

FlowScript's competitive differentiator is RELATE > DELETE — contradictions become queryable tensions, not lost data. But this only matters if the reasoning chain is provably auditable. Enterprise customers in regulated industries (finance, healthcare, government) need to answer:

1. **Why does the agent believe X?** — Show the full reasoning chain
2. **What was considered and rejected?** — Consolidation decisions with reasoning
3. **What was extracted from this conversation?** — Typed extraction provenance
4. **What changed and when?** — Temporal mutation history
5. **Can we prove this trail hasn't been tampered with?** — Verifiable integrity

Nobody else can provide this because nobody else has typed consolidation decisions to audit. Mem0 deletes contradictions — there's nothing to audit. Our RELATE/RESOLVE/UPDATE actions with LLM reasoning ARE the compliance artifact.

**The adapter/agentic workflow story is the greater enterprise need.** An enterprise running a CrewAI fleet or LangGraph pipeline needs auditable reasoning across sessions. The MCP story is important but secondary — most enterprise customers will need both, with adapter-level audit being the harder problem for competitors to solve. This is MORE important than MCP audit because: (a) enterprise customers will have both needs, (b) adapter-level audit is the greater of the two, and (c) it's harder for competitors to solve since they lack typed consolidation.

---

## 2. Design Principles

1. **Append-only entries, rotatable files.** Entries are never modified or deleted. Files are sealed and rotated. The "append-only" guarantee applies to the logical stream, not a single physical file.

2. **Write-first (crash-safe).** Audit entry is written BEFORE the mutation it describes. Worst case = duplicate entry (harmless). Already implemented for prune/update_node.

3. **Hash-chained (tamper-evident).** Each entry includes `prev_hash = SHA256(previous_entry_json)`. Broken chain = detectable tampering. Turns a log file into a verifiable decision ledger. Convergent with AWS AgentCore's hash-chained Fact Ledger — independent industry arrival at the same architecture.

4. **One specification, two implementations.** Shared file format, event taxonomy, rotation pattern, hash-chaining. TS and Python produce identical audit trail format. Some events are language-specific (consolidation = Python-only, fromTranscript = TS-only). An enterprise using both (TS for their web app, Python for their agent fleet) gets one consistent audit story.

5. **Composable, not mandatory.** Audit trail writes when `file_path` is set (existing behavior). No audit overhead for in-memory-only usage.

6. **Rotation is infrastructure, not policy.** The SDK provides rotation mechanics. Retention policy is the user's decision (configurable).

7. **Observable by design.** An `on_event` callback enables real-time integration with monitoring systems (Datadog, Splunk, SIEM tools) and FlowScript Observatory (executive visibility layer — see section 11). The callback is trivially cheap to implement now and prevents refactoring the audit system later when Observatory ships.

---

## 3. Audit Event Taxonomy

### 3.1 Shared Events (Both TS and Python)

| Event | Trigger | Data Captured | Why It Matters |
|-------|---------|---------------|----------------|
| `session_start` | `sessionStart()` called | session_id, timestamp, config snapshot | Session boundary for temporal grouping |
| `session_end` | `sessionEnd()` called | session_id, nodes_touched, garden_report | Session completion with activity summary |
| `session_wrap` | `sessionWrap()` called | session_id, nodes_graduated, nodes_pruned | Full lifecycle event (graduation + prune) |
| `node_create` | New node added to graph | node (full), source (api/extraction/transcript/mcp) | Provenance: how did this node enter the system |
| `node_update` | `update_node()` mutates content | old_id, new_id, old_content, new_content, reason | Content mutation trail |
| `node_merge` | `update_node()` causes hash collision | old_id, merged_into, old_content, target_content, reason | Merge provenance (two nodes became one) |
| `relationship_create` | New relationship added | relationship (full), source (api/consolidation/extraction) | How structural connections form |
| `state_change` | State added/modified on node | state (full), previous_state (if update), source | Decision/blocker/parking lifecycle |
| `prune` | `prune()` removes dormant nodes | nodes[], relationships[], states[], temporal{}, reason | Full snapshot of everything removed |
| `graduation` | Node tier changes (current→developing→proven→foundation) | node_id, old_tier, new_tier, frequency | Knowledge maturation trail |

### 3.2 Python-Only Events (flowscript-agents)

| Event | Trigger | Data Captured | Why It Matters |
|-------|---------|---------------|----------------|
| `extraction` | `AutoExtract` produces typed nodes from raw text | raw_text (truncated), extracted_nodes[], model, actor | What the LLM extracted and what types it assigned |
| `consolidation` | `ConsolidationEngine` decides action | action (ADD/UPDATE/RELATE/RESOLVE/NONE), reasoning, target_node_id, candidates_count, is_fallback | THE differentiator — LLM's reasoning for every memory management decision |
| `consolidation_batch` | Full batch completes | batch_size, novel_count, contested_count, actions_summary, fallback_count, health_metrics | Batch-level overview for monitoring and Observatory |
| `vector_index` | Node indexed in vector store | node_id, embedding_model, dimensions, content_hash | Embedding provenance (model + content hash, NOT the vector — see section 10.2) |

### 3.3 TypeScript-Only Events (flowscript-core)

| Event | Trigger | Data Captured | Why It Matters |
|-------|---------|---------------|----------------|
| `transcript_extract` | `fromTranscript()` processes conversation | transcript_length, extracted_nodes[], model | Conversation → typed nodes provenance |
| `snapshot` | `snapshot()` creates named checkpoint | snapshot_name, node_count, relationship_count | Point-in-time captures |
| `budget_apply` | Token budgeting applied | strategy, budget_tokens, nodes_before, nodes_after | What was excluded for token limits |

### 3.4 MCP/Tool Call Auditing — Two-Tier Design

**Mutations via MCP tools (always audited):** When an MCP tool handler calls `memory.thought()`, `memory.tension_with()`, etc., the resulting Memory-level events (`node_create`, `relationship_create`, `state_change`) fire automatically with `"adapter": {"framework": "mcp", ...}` attribution. No separate MCP event needed — the Memory-level event IS the audit record, with MCP as the attributed source. Zero duplication, zero gaps.

**Reads/queries via MCP tools (gated behind verbosity):** `search_memory`, `get_context`, `query_tensions`, etc. don't mutate state but may be compliance-relevant (HIPAA requires audit of who accessed what data). These are high-volume events that would bloat the audit trail for most users.

| Verbosity Level | Mutations | Reads/Queries | Use Case |
|----------------|-----------|---------------|----------|
| `standard` (default) | ✅ Always | ❌ Not logged | Most users — mutation trail sufficient |
| `full` | ✅ Always | ✅ Logged as `query_access` events | Regulated industries (HIPAA, SOX) needing access auditing |

### 3.5 Adapter Attribution (Python — on all mutation events)

When an event originates from an adapter context, include:

```json
{
  "adapter": {
    "framework": "langgraph",
    "adapter_class": "LangGraphStore",
    "operation": "put"
  }
}
```

This lets an enterprise query: "Show me all consolidation decisions made by our CrewAI agents" or "What did the LangGraph pipeline extract from this conversation?"

When an event originates from direct Memory API usage (no adapter), the `adapter` field is `null`.

When an event originates from an MCP tool call, the adapter attribution is:
```json
{
  "adapter": {
    "framework": "mcp",
    "adapter_class": "FlowScriptMCPServer",
    "operation": "add_memory"
  }
}
```

---

## 4. Entry Format

### 4.1 Base Entry Structure

Every audit entry follows this schema:

```json
{
  "v": 1,
  "seq": 42,
  "timestamp": "2026-03-21T16:30:00.000Z",
  "event": "consolidation",
  "prev_hash": "sha256:a1b2c3d4e5f6...",
  "session_id": "ses_abc123",
  "data": {
    "action": "RELATE",
    "reasoning": "These represent opposing architectural approaches — caching vs direct queries — creating a tension worth preserving for future decision-making.",
    "new_content": "Use Redis caching layer for session data",
    "new_type": "thought",
    "target_node_id": "tho_abc123",
    "candidates_count": 3,
    "is_fallback": false
  },
  "adapter": {
    "framework": "crewai",
    "adapter_class": "CrewAIStorage",
    "operation": "save"
  }
}
```

**Field definitions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `v` | integer | yes | Schema version. Always `1` for this spec. Enables future format evolution without breaking readers. |
| `seq` | integer | yes | Monotonically increasing sequence number within the current file. Resets to 0 on rotation. Combined with `prev_hash`, provides ordering + integrity. |
| `timestamp` | ISO 8601 string | yes | UTC. When the event occurred. |
| `event` | string | yes | Event type from taxonomy (section 3). |
| `prev_hash` | string | yes | `"sha256:<hex>"` of the previous entry's complete JSON string (before newline). First entry in a file uses `"sha256:GENESIS"` (new file) or `"sha256:<last_hash_from_previous_file>"` (rotation continuity). |
| `session_id` | string | nullable | Session identifier if within an active session. Null for events outside session lifecycle. |
| `data` | object | yes | Event-specific payload. Schema varies by event type. |
| `adapter` | object | nullable | Present when event originates from an adapter or MCP. Null for direct Memory API usage. |

### 4.2 Hash-Chaining

```
Entry 0: { ..., "prev_hash": "sha256:GENESIS", ... }
Entry 1: { ..., "prev_hash": "sha256:<SHA256(json_string_of_entry_0)>", ... }
Entry 2: { ..., "prev_hash": "sha256:<SHA256(json_string_of_entry_1)>", ... }
```

The hash is computed over the **complete JSON string of the previous entry** (the exact bytes written to disk, before the trailing newline). This means:
- JSON serialization must be deterministic (sorted keys, no trailing whitespace)
- The hash covers all fields including `prev_hash` itself — creating a true chain where modifying any entry breaks all subsequent hashes

**Verification algorithm:**
```python
def verify_chain(entries: list[str]) -> tuple[bool, int]:
    """Verify hash chain integrity. Returns (valid, first_broken_index)."""
    prev_hash = "sha256:GENESIS"
    for i, line in enumerate(entries):
        entry = json.loads(line)
        if entry["prev_hash"] != prev_hash:
            return (False, i)
        prev_hash = "sha256:" + hashlib.sha256(line.encode("utf-8")).hexdigest()
    return (True, -1)
```

**Cross-file continuity:** When rotating, the manifest records the last hash of the sealed file. The first entry of the new active file uses that hash as its `prev_hash`, creating an unbroken chain across rotations. Verification can walk the entire history by following manifest → file → manifest → file.

**Why this matters competitively:** AWS AgentCore has a hash-chained Fact Ledger (append-only, hash-chained). We have it too — independently arrived at, open source, with typed reasoning (not just flat facts). arXiv:2603.17244 Principle 5 states "memory states never overwritten; versioned" — our audit trail is the implementation of that principle.

### 4.3 Sensitive Data

Audit entries capture node content, which may contain sensitive information. The SDK does NOT redact — that's the user's responsibility at the application layer. Document this clearly:

> **Privacy note:** Audit trail entries contain the full content of reasoning nodes, including any information extracted from conversations. If your application processes PII, PHI, or other regulated data, configure content redaction at the application layer before data reaches FlowScript, or apply redaction to audit files during rotation.

---

## 5. File Layout & Rotation

### 5.1 File Naming

```
{memory_stem}.audit.jsonl                    # active (hot) — appending
{memory_stem}.audit.2026-02.jsonl.gz         # sealed February (compressed)
{memory_stem}.audit.2026-01.jsonl.gz         # sealed January (compressed)
{memory_stem}.audit.manifest.json            # index (time ranges, counts, hashes)
```

Examples:
```
agent_memory.audit.jsonl
agent_memory.audit.2026-02.jsonl.gz
agent_memory.audit.2026-01.jsonl.gz
agent_memory.audit.manifest.json
```

### 5.2 Rotation Trigger

**Monthly rotation** (default). Configurable via `AuditConfig`:

```python
# Python
AuditConfig(
    rotation="monthly",           # "monthly" | "weekly" | "daily" | "size:50MB" | "none"
    compression="gzip",           # "gzip" | "none"
    retention_months=84,          # 7 years (SOX). None = keep forever.
    hash_chain=True,              # default True
    verbosity="standard",         # "standard" | "full" (see section 3.4)
    on_event=None,                # optional callback: Callable[[dict], None]
)
```

```typescript
// TypeScript
interface AuditConfig {
    rotation: 'monthly' | 'weekly' | 'daily' | `size:${string}` | 'none';
    compression: 'gzip' | 'none';
    retentionMonths: number | null;   // 84 = 7 years (SOX). null = keep forever.
    hashChain: boolean;               // default true
    verbosity: 'standard' | 'full';   // default 'standard'
    onEvent?: (entry: AuditEntry) => void;  // optional real-time callback
}
```

**Rotation process (atomic):**
1. Close active file handle
2. Rename active file with period suffix (e.g., `.2026-03.jsonl`)
3. Compress renamed file (→ `.2026-03.jsonl.gz`)
4. Delete uncompressed version after successful compression
5. Update manifest
6. Create new active file
7. First entry in new file chains from last hash of sealed file

**When rotation checks happen:** On every `_write_audit()` call, check if the active file's period has changed. If current month ≠ file's month → rotate before writing. Lazy rotation — no background timers, no cron. Triggered by the next write after the period boundary.

### 5.3 Manifest

```json
{
  "version": 1,
  "memory_file": "agent_memory.json",
  "active_file": "agent_memory.audit.jsonl",
  "active_last_hash": "sha256:f7e8d9c0...",
  "active_last_seq": 147,
  "files": [
    {
      "filename": "agent_memory.audit.2026-02.jsonl.gz",
      "period": "2026-02",
      "entries": 1847,
      "first_timestamp": "2026-02-01T00:00:12Z",
      "last_timestamp": "2026-02-28T23:58:44Z",
      "first_hash": "sha256:GENESIS",
      "last_hash": "sha256:a1b2c3d4...",
      "size_bytes": 24819,
      "uncompressed_bytes": 187432,
      "sha256_file": "sha256:9f8e7d6c..."
    }
  ],
  "retention_months": 84,
  "last_cleanup": "2026-03-01T00:05:00Z"
}
```

**Manifest enables:**
- Skip decompressing files outside query time range
- Verify file integrity via stored hashes (tamper detection at file level)
- Show audit statistics without reading any log files
- Cross-file hash chain verification (last_hash → next file's first entry prev_hash)
- **Observatory integration:** quick statistics view without scanning logs

### 5.4 Retention Cleanup

On rotation, check for files older than `retention_months`:
1. Files past retention → delete
2. Update manifest (remove deleted file entries)
3. Log cleanup to active audit trail (meta-event: `audit_cleanup`)

Retention = delete. Not "move to cold storage" (that's the user's infrastructure concern). The SDK's job is: default to keeping everything, delete only when explicitly configured.

**Retention reference (for README documentation):**

| Framework | Minimum Retention | Hot/Immediate Access | Notes |
|-----------|------------------|---------------------|-------|
| PCI DSS 4.0 | 12 months | 3 months | Tamper-resistant and reviewable |
| HIPAA | 6 years | Not specified | From creation or last in effect |
| SOX | 7 years | Not specified | Audit documents and control evidence |
| ISO 27001 | 12 months (baseline) | Risk-based | Adjust by risk assessment |
| NIST 800-53 | 3 years | Not specified | Federal systems |
| SOC 2 | No fixed period | N/A | Align with applicable regulations |
| GDPR | As short as possible | N/A | Storage limitation principle — must justify retention |
| SEC 17a-4 | 6 years | Not specified | Financial records |

Default `retention_months=84` (7 years) covers the strictest common standard (SOX). Users can configure down for GDPR-sensitive contexts.

---

## 6. Query API

### 6.1 Static Methods (Both Languages)

```python
# Python
Memory.read_audit_log(audit_path: str) -> list[dict]
    # Existing method — reads single file. Keep for backwards compat.

Memory.query_audit(
    audit_path: str,                    # path to active file OR manifest
    after: str | None = None,           # ISO timestamp
    before: str | None = None,          # ISO timestamp
    events: list[str] | None = None,    # filter by event type
    node_id: str | None = None,         # filter by node involvement
    session_id: str | None = None,      # filter by session
    adapter: str | None = None,         # filter by adapter framework
    limit: int = 100,
    verify_chain: bool = False,         # verify hash chain integrity
) -> AuditQueryResult

Memory.verify_audit(audit_path: str) -> AuditVerifyResult
    # Convenience method — full chain verification across all files
```

```typescript
// TypeScript
static readAuditLog(auditPath: string): AuditEntry[]
    // Existing method — keep for backwards compat.

static queryAudit(
    auditPath: string,
    options?: {
        after?: string;
        before?: string;
        events?: string[];
        nodeId?: string;
        sessionId?: string;
        adapter?: string;
        limit?: number;
        verifyChain?: boolean;
    }
): AuditQueryResult

static verifyAudit(auditPath: string): AuditVerifyResult
```

### 6.2 Result Types

```python
@dataclass
class AuditQueryResult:
    entries: list[dict]
    total_scanned: int
    files_searched: int
    chain_valid: bool | None     # None if verify_chain=False
    chain_break_at: int | None   # seq number of first break, if any

@dataclass
class AuditVerifyResult:
    valid: bool
    total_entries: int
    files_verified: int
    legacy_entries: int          # pre-hash-chain entries (unverifiable but not broken)
    chain_break_at: int | None   # seq number of first break, if any
    chain_break_file: str | None # filename containing the break
```

### 6.3 MCP Audit Tools (Python MCP Server)

Two new tools for the MCP server:

| Tool | Description | Parameters |
|------|-------------|------------|
| `query_audit_trail` | Search the audit trail for reasoning provenance | `after`, `before`, `events`, `node_id`, `session_id`, `adapter`, `limit` |
| `verify_audit_integrity` | Verify the hash chain is unbroken | (none — runs full verification) |

**Behavioral descriptions (for tool definitions — what the LLM sees):**
- `query_audit_trail`: "Call this when asked to explain WHY a decision was made, WHEN a memory changed, or to show the reasoning history behind a specific piece of knowledge. Also use to audit what an agent fleet has done across sessions."
- `verify_audit_integrity`: "Call this when asked to prove the audit trail hasn't been tampered with, or during compliance audits. Returns chain integrity status across all audit files."

### 6.4 TypeScript MCP Audit Tool (asTools)

One new tool added to asTools:

| Tool | Description | Parameters |
|------|-------------|------------|
| `query_audit` | Search the audit trail | `after`, `before`, `events`, `nodeId`, `sessionId`, `limit` |

---

## 7. Implementation Plan

### 7.1 Shared Infrastructure (implement first, both languages)

1. **AuditWriter class** — encapsulates file writing, hash-chaining, rotation, manifest management
   - `write(event, data, session_id?, adapter?)` — main entry point
   - `rotate()` — seal current file, compress, update manifest, open new
   - `verify()` — full chain verification across all files
   - `query(filters)` — filtered read across active + rotated files
   - Lazy rotation check on every write
   - `on_event` callback invocation on every write (before file I/O — callback failure must not block audit)

2. **Hash-chaining** — SHA256, `prev_hash` field, GENESIS sentinel, cross-file continuity via manifest

3. **Manifest** — JSON sidecar, updated on rotation, enables efficient querying

4. **Entry format** — v1 schema with `v`, `seq`, `timestamp`, `event`, `prev_hash`, `session_id`, `data`, `adapter`

5. **Deterministic JSON serialization** — sorted keys, no trailing whitespace, consistent float formatting. Critical for hash-chaining (same logical entry must produce same bytes).

### 7.2 Python Implementation (flowscript-agents)

1. **Create `flowscript_agents/audit.py`** — AuditWriter class
2. **Replace `_write_audit()` in Memory** with AuditWriter delegation
3. **Add event hooks** in Memory for: session_start, session_end, session_wrap, node_create, relationship_create, state_change, graduation
4. **Add event hooks** in ConsolidationEngine for: consolidation (already exists — rewire to AuditWriter), consolidation_batch summary (new)
5. **Add event hooks** in AutoExtract for: extraction events (new)
6. **Add adapter attribution** — adapters pass framework context through to Memory, Memory passes to AuditWriter
7. **Add `query_audit()`, `verify_audit()`** static methods on Memory
8. **Add MCP tools** — query_audit_trail, verify_audit_integrity
9. **Migration** — existing `.audit.jsonl` files (no hash chain) are valid; verification reports "legacy entries (no hash chain)" rather than "broken chain"

### 7.3 TypeScript Implementation (flowscript-core)

1. **Create `src/audit.ts`** — AuditWriter class
2. **Replace inline audit in `prune()`** with AuditWriter delegation
3. **Expand AuditEntry type** — union type of all event shapes (replaces current `event: 'prune'` restriction)
4. **Add event hooks** in Memory for: session_start, session_end, session_wrap, node creation (thought/decide/etc.), relationship creation, state changes, graduation, fromTranscript, snapshot, budget_apply
5. **Add `queryAudit()`, `verifyAudit()`** static methods on Memory
6. **Add MCP tool** — query_audit in asTools
7. **Migration** — same as Python: legacy files valid, verification aware

### 7.4 Test Plan

**Per language (unit tests):**
- Hash chain creation + verification (happy path)
- Hash chain tamper detection (modified entry detected)
- Deterministic JSON serialization (same entry → same hash across runs)
- Rotation trigger (monthly boundary)
- Compression of rotated files (gzip round-trip)
- Manifest creation + update on rotation
- Cross-file chain continuity (last hash of file N → first hash of file N+1)
- Query by time range, event type, node ID, session ID, adapter
- Query across rotated + active files
- Legacy file compatibility (pre-hash-chain files)
- GENESIS handling (first entry in fresh file)
- LEGACY_BRIDGE handling (first new entry in legacy file)
- Concurrent write safety (append-only = safe for single-writer)
- In-memory mode (no file_path) = no audit (existing behavior preserved)
- on_event callback invocation (fires for every entry)
- on_event callback failure does not block audit write
- Verbosity levels: standard (mutations only) vs full (mutations + reads)
- Retention cleanup (old files deleted, manifest updated)

**Integration tests:**
- Full session lifecycle → audit trail captures all events in order
- Consolidation pipeline → every decision audited with reasoning
- Adapter attribution → framework context flows through to audit entries
- Prune → audit → verify chain → all consistent
- Rotation → query across rotated + active files → results correct
- MCP tool call → Memory-level event fires with MCP attribution
- End-to-end: create memory → add nodes → consolidate → prune → verify chain → query history

---

## 8. Migration & Backwards Compatibility

**Existing `.audit.jsonl` files** lack `v`, `seq`, `prev_hash` fields. Strategy:

1. **Reading:** `read_audit_log()` (existing method) continues to work unchanged — it returns raw dicts. No breaking change.
2. **Verification:** `verify_chain()` detects legacy entries (missing `prev_hash`) and reports them as "legacy (unverifiable)" rather than "broken chain." The `AuditVerifyResult.legacy_entries` count tells the user how many pre-chain entries exist.
3. **Appending:** New entries written to an existing legacy file start a new hash chain from that point forward. The first new entry uses `prev_hash: "sha256:LEGACY_BRIDGE"` sentinel to indicate the transition from unchained to chained entries.
4. **No migration required.** Old files are valid. New entries chain from where they start. Users who want full chain coverage can rotate the legacy file manually (it becomes a sealed archive, new writes start fresh with GENESIS).

---

## 9. Competitive Positioning

### 9.1 Capability Matrix

| Capability | FlowScript | Mem0 | LangChain/langmem | AWS AgentCore | Others |
|-----------|-----------|------|-------------------|---------------|--------|
| Consolidation reasoning trail | ✅ Every decision with LLM reasoning | ❌ Deletes silently | ❌ Deletes silently | ❌ Flat facts only | ❌ |
| Hash-chained tamper evidence | ✅ SHA256 chain | ❌ | ❌ | ✅ Hash-chained Fact Ledger | ❌ |
| Typed extraction provenance | ✅ Decisions, tensions, causal chains | ❌ Flat facts | ❌ Flat facts | ❌ Flat facts | ❌ |
| Cross-session audit continuity | ✅ Session lifecycle events | ❌ | ❌ | Partial | ❌ |
| Framework-attributed audit | ✅ Adapter context on every event | ❌ | ❌ | ❌ | ❌ |
| Rotation + compression | ✅ Monthly, gzip, manifest | ❌ | ❌ | ✅ (managed) | ❌ |
| Access auditing (reads) | ✅ (verbosity: full) | ❌ | ❌ | ❌ | ❌ |
| Real-time event streaming | ✅ on_event callback | ❌ | ❌ | ✅ (CloudWatch) | ❌ |
| Compliance retention config | ✅ Configurable (default 7yr SOX) | ❌ | ❌ | ✅ (managed) | ❌ |

### 9.2 README Copy (Draft — For Future README Rewrite)

**Section: Audit Trail — Compliance-Grade Reasoning Provenance**

> Every memory management decision — why a memory was updated, why two ideas were related as a tension instead of one deleting the other, why a blocker was resolved — is captured with the LLM's own reasoning in an append-only, hash-chained audit trail.
>
> ```python
> # Verify your audit trail hasn't been tampered with
> result = Memory.verify_audit("agent_memory.audit.jsonl")
> assert result.valid  # SHA256 hash chain intact
>
> # Query: why does the agent believe X?
> history = Memory.query_audit("agent_memory.audit.jsonl",
>     node_id="tho_abc123",
>     events=["consolidation", "node_create", "node_update"])
> # Returns: full reasoning chain with LLM explanations
> ```
>
> **Why this matters:** Mem0 deletes contradictions — you can't audit a deletion. LangChain's langmem deletes contradictions. FlowScript's RELATE > DELETE architecture means the reasoning IS the data. Every tension, every consolidation decision, every extraction — preserved with provenance, hash-chained for tamper evidence.
>
> **Compliance-ready:**
> - Hash-chained entries (tamper-evident — broken chain = detectable modification)
> - Configurable retention (default 7 years for SOX; adjust for PCI DSS, HIPAA, GDPR)
> - Monthly rotation with gzip compression (hot/warm/cold lifecycle)
> - Framework attribution (which adapter/agent generated each decision)
> - Access auditing available for HIPAA contexts (`verbosity="full"`)
> - Real-time streaming via `on_event` callback (Datadog, Splunk, SIEM integration)

**Section: Framework Integration — Auditable Agent Fleets**

> When your CrewAI fleet or LangGraph pipeline uses FlowScript memory, every adapter automatically attributes its framework context to audit events. Query by framework:
>
> ```python
> # What did our CrewAI agents decide today?
> crewai_decisions = Memory.query_audit("memory.audit.jsonl",
>     adapter="crewai",
>     events=["consolidation"],
>     after="2026-03-21T00:00:00Z")
>
> # Show me all extractions from the LangGraph pipeline
> extractions = Memory.query_audit("memory.audit.jsonl",
>     adapter="langgraph",
>     events=["extraction"])
> ```
>
> No other agent memory system provides framework-attributed audit trails. This is where FlowScript's adapter architecture meets enterprise compliance — not just "works with your stack" but "auditably works with your stack."

**Section: The Kill Shot (for competitive positioning page/blog, not README)**

> Mem0 can't build this even if they wanted to. Their architecture deletes contradictions — the reasoning that led to the deletion is gone. You can't audit what doesn't exist.
>
> FlowScript's RELATE > DELETE means contradictions become tensions (queryable knowledge), not deletions (lost data). The consolidation engine's reasoning — "I'm relating these as a tension because they represent opposing architectural approaches" — IS the audit trail. The reasoning IS the data.
>
> AWS AgentCore independently arrived at hash-chained fact ledgers (append-only, tamper-evident). arXiv:2603.17244 independently arrived at "memory states never overwritten; versioned." Three independent sources converging on the same architecture. FlowScript ships it open source with typed reasoning that neither AWS nor academic papers have implemented.

### 9.3 Show HN Angle

One sentence in the Show HN post (not the lede — credibility footnote):

> "Also includes hash-chained audit trails — every consolidation decision captured with the LLM's own reasoning, tamper-evident by default. SOX/HIPAA/PCI-ready retention."

This plants the enterprise seed without making Show HN about compliance. The lede stays: "Mem0 remembers what you said. FlowScript remembers why you decided."

---

## 10. Resolved Design Decisions

### 10.1 MCP Tool Call Auditing

**Decision:** Mutations always audited through Memory-level events with MCP/adapter attribution. Reads gated behind `verbosity: "full"`.

**Rationale:** Memory-level events (`node_create`, `relationship_create`, etc.) already fire when MCP tool handlers call Memory methods. Adding separate MCP-level events for mutations would be redundant duplication. The `adapter` attribution field on Memory events captures the MCP origin without a second event. Reads (search, query) are high-volume and only compliance-relevant in regulated industries (HIPAA access auditing) — hence the verbosity gate.

### 10.2 Embedding Vectors in Audit

**Decision:** Capture `model`, `dimensions`, `content_hash` (SHA256 of embedded text). NOT the vector itself.

**Rationale:** Compliance needs reproducibility, not the vector. Same model + same content → same embedding (deterministic enough). Storing 1536 floats per node per entry = ~6KB overhead that serves no compliance purpose. The content hash + model name is independently verifiable (reproduce the embedding and compare) — actually a stronger proof than storing the vector (stored vectors could be tampered with; reproduced vectors are independently verifiable).

### 10.3 Encryption at Rest

**Decision:** Deferred to v2. Documented as future enhancement.

**Rationale:** Most production deployments handle encryption at the storage layer (encrypted filesystems, S3 server-side encryption, database-level encryption). Storage-layer encryption is more secure than application-level (no key management burden on the SDK, covers all files not just audit). Application-level encryption (AES-256 of rotated files) remains a valid v2 feature for environments where storage-layer encryption isn't available.

**MUST NOT FORGET:** This is a documented commitment. Track in project next.md post-Show HN roadmap. Enterprise customers in air-gapped environments may need this.

### 10.4 Real-Time Event Streaming (on_event Callback)

**Decision:** Implement the callback hook now. Defer Observatory itself to post-launch.

**Rationale:** The `on_event` callback is trivially cheap (~3 lines of code in AuditWriter, one config field). But it's the integration surface for:
1. **SIEM tools** (Datadog, Splunk) — real-time compliance monitoring
2. **FlowScript Observatory** — executive visibility into agent reasoning (see section 11)
3. **Custom dashboards** — enterprise-specific monitoring

Building Observatory without this hook would require refactoring the audit system. Building the hook without Observatory costs almost nothing. The hook is infrastructure; Observatory is the product built on it.

**Implementation:**
```python
# In AuditWriter.write():
if self._config.on_event:
    try:
        self._config.on_event(entry)
    except Exception:
        pass  # Callback failure must never block audit write
```

The callback receives the full audit entry dict AFTER it's written to disk (audit persistence > callback delivery).

---

## 11. Future: FlowScript Observatory

**NOT in scope for Show HN. Documented here because audit trail design must account for it.**

Observatory is the executive visibility layer — a dashboard showing what agent fleets are deciding and why, across sessions, across frameworks. The audit trail is Observatory's data source.

**What Observatory needs from the audit trail (design the hook now, build Observatory later):**
- Real-time event stream (`on_event` callback — implemented in this spec)
- Batch-level summaries (`consolidation_batch` events — in this spec)
- Framework attribution (`adapter` field — in this spec)
- Session boundaries (`session_start`/`session_end` events — in this spec)
- Query API for historical analysis (`query_audit()` — in this spec)
- Manifest for quick statistics (`manifest.json` — in this spec)

**Observatory integration pattern (future):**
```python
from flowscript_observatory import Observatory

obs = Observatory(api_key="...")
memory = Memory.load_or_create("agent.json",
    audit=AuditConfig(on_event=obs.ingest))

# Every audit event now streams to Observatory in real-time
# Observatory provides: executive dashboard, fleet health, reasoning explorer,
# compliance reports, anomaly detection
```

**What this means for the audit trail design:** Every design decision in this spec should be evaluated against "will Observatory be able to use this?" The answer should always be yes — that's why `on_event`, `consolidation_batch`, adapter attribution, and the manifest exist. They're not just compliance features — they're Observatory's foundation.

---

## 12. README Documentation Requirements

**When rewriting READMEs (Phase 5), the audit trail sections need:**

### flowscript-agents README (Python)
1. **Audit Trail section** — positioned as compliance differentiator, not afterthought
2. **Code example:** create memory → add content → consolidate → prune → verify chain → query history
3. **Compliance table:** retention requirements by framework (section 5.4 of this doc)
4. **Configuration example:** AuditConfig with all options
5. **Framework attribution example:** querying by adapter
6. **Competitive comparison:** capability matrix (section 9.1)
7. **Privacy note:** sensitive data warning (section 4.3)
8. **SIEM integration example:** on_event callback to Datadog/Splunk
9. **The RELATE > DELETE angle:** "You can't audit a deletion" — weave into competitive narrative, not isolated in audit section

### flowscript-core README (TypeScript)
1. **Audit Trail section** — same positioning as Python
2. **Code example:** Memory with auditConfig → build graph → sessionWrap → verify → query
3. **fromTranscript provenance:** "every extraction from conversation is audited"
4. **Configuration example:** AuditConfig TypeScript interface
5. **Verify example:** `Memory.verifyAudit()` one-liner
6. **Cross-language note:** "Same audit format as Python SDK — one enterprise, one audit story"

### MCP Server Docs
1. **New tools:** query_audit_trail, verify_audit_integrity
2. **Behavioral note:** "Claude automatically queries audit trail when asked about reasoning history"
3. **Configuration:** audit config in MCP server setup

### Show HN Post (Phill writes)
1. **One sentence footnote** — hash-chained audit trails, compliance-ready (not the lede)
2. **Save the full compliance story** for the follow-up blog post / deep-dive

### Future Blog Post (Post-Show HN)
1. **Full competitive analysis:** Mem0 DELETE vs FlowScript RELATE, with audit trail as proof
2. **Hash-chaining explainer:** what it is, why it matters, how to verify
3. **Enterprise use case:** "Your CrewAI fleet made 10,000 decisions today. Can you explain any of them?"
4. **Convergent evidence:** AWS AgentCore + arXiv:2603.17244 + FlowScript = three independent arrivals

---

*This design doc is the specification. Implementation follows review. All decisions are final unless new information surfaces during implementation.*
