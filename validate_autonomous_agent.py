#!/usr/bin/env python3
"""
Validate FlowScript compliance wrapper — autonomous agent simulation.

Simulates a real autonomous agent scenario: an agent tasked with evaluating
cloud infrastructure options and making a recommendation. Multi-turn reasoning
with tradeoffs, tensions, and a final decision. No human in the loop.

The wrapper captures every exchange automatically. We examine:
1. Are the extracted nodes semantically meaningful?
2. Do tensions capture real tradeoffs?
3. Can we trace decisions back to reasoning via why()?
4. Does the audit trail provide compliance-grade provenance?

Usage: Run from flowscript-agents repo root with ANTHROPIC_API_KEY set.
"""
import os, sys, json, time
from pathlib import Path

# Load keys from .env.flow
env_file = Path.home() / "Documents/flow/.env.flow"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from anthropic import Anthropic
from flowscript_agents import UnifiedMemory
from flowscript_agents.client import FlowScriptAnthropic
from flowscript_agents.explain import explain

# --- Setup ---
raw = Anthropic()

def extraction_llm(prompt):
    """Haiku for extraction — cheap and fast."""
    r = raw.messages.create(model="claude-haiku-4-5-20251001", max_tokens=2048,
                            messages=[{"role": "user", "content": prompt}])
    return r.content[0].text

# File-backed memory with auto-save
output_dir = Path("./validation_output")
output_dir.mkdir(exist_ok=True)
mem_path = str(output_dir / "agent_memory.json")

mem = UnifiedMemory(file_path=mem_path, llm=extraction_llm, auto_save=True)
client = FlowScriptAnthropic(raw, memory=mem)

SYSTEM = """You are an autonomous infrastructure evaluation agent. Your job is to
evaluate options, identify tradeoffs, and make a clear recommendation with rationale.
Be specific about costs, risks, and tradeoffs. Make a definitive recommendation."""

def agent_exchange(messages, step_name):
    """Run one agent exchange and report."""
    print(f"\n{'─' * 50}")
    print(f"STEP: {step_name}")
    start = time.time()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        system=SYSTEM,
        messages=messages,
    )
    elapsed = time.time() - start
    text = response.content[0].text
    nodes_now = len(mem.memory.nodes)
    rels_now = mem.memory.relationship_count
    print(f"  Response: {len(text)} chars | {elapsed:.1f}s")
    print(f"  Memory: {nodes_now} nodes, {rels_now} relationships (cumulative)")
    print(f"  Preview: {text[:150]}...")
    return text

# =============================================================================
# AUTONOMOUS AGENT SCENARIO: Cloud Database Migration Evaluation
# =============================================================================

print("=" * 60)
print("AUTONOMOUS AGENT SIMULATION")
print("Scenario: Cloud database migration evaluation")
print("=" * 60)

messages = []

# Turn 1: Initial analysis request
messages.append({"role": "user", "content":
    "We're migrating from a self-hosted PostgreSQL cluster to a managed cloud database. "
    "Our system handles 50,000 requests/day, stores 2TB of data including PII, and must "
    "maintain 99.95% uptime. The current on-prem cost is $4,200/month. Evaluate AWS RDS, "
    "Google Cloud SQL, and Azure Database for PostgreSQL. Consider: cost, compliance "
    "(we're subject to EU AI Act for our agent fleet), migration complexity, and vendor lock-in."
})
r1 = agent_exchange(messages, "Initial evaluation")
messages.append({"role": "assistant", "content": r1})

# Turn 2: Deeper on compliance
messages.append({"role": "user", "content":
    "Go deeper on the compliance dimension. Our AI agents make automated decisions about "
    "customer credit scoring. Under EU AI Act Article 86, affected customers have a right "
    "to explanation. Which provider best supports maintaining audit trails of agent reasoning? "
    "Also consider GDPR data residency — all PII must stay in EU data centers."
})
r2 = agent_exchange(messages, "Compliance deep-dive")
messages.append({"role": "assistant", "content": r2})

# Turn 3: Cost analysis tension
messages.append({"role": "user", "content":
    "I'm seeing a tension: the cheapest option (AWS) has the weakest EU compliance tooling, "
    "while the most compliant option (Azure with EU Data Boundary) costs 40% more. "
    "Our budget is firm at $5,000/month. Analyze this tradeoff — is there a way to get "
    "compliance without the premium, or should we argue for a budget increase?"
})
r3 = agent_exchange(messages, "Cost vs compliance tension")
messages.append({"role": "assistant", "content": r3})

# Turn 4: Risk assessment
messages.append({"role": "user", "content":
    "What happens if we choose the cheaper option and an EU regulator audits us? "
    "What's the realistic risk? Factor in: we're a Series A startup with 200 customers, "
    "the AI Act enforcement starts August 2026, and our agent fleet makes ~1,000 automated "
    "decisions per day. Give me a risk assessment with probability and impact."
})
r4 = agent_exchange(messages, "Risk assessment")
messages.append({"role": "assistant", "content": r4})

# Turn 5: Final recommendation
messages.append({"role": "user", "content":
    "Based on everything above, make your final recommendation. One option. Clear rationale. "
    "Include: which provider, which tier, estimated monthly cost, migration timeline, "
    "and the key risk we're accepting."
})
r5 = agent_exchange(messages, "Final recommendation")
messages.append({"role": "assistant", "content": r5})

# =============================================================================
# ARTIFACT ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("ARTIFACT ANALYSIS")
print("=" * 60)

g = mem.memory

print(f"\n{'─' * 50}")
print(f"GRAPH: {len(g.nodes)} nodes, {g.relationship_count} relationships")
print(f"{'─' * 50}")

# Node type distribution
type_counts = {}
for n in g.nodes:
    t = n.node.type.value
    type_counts[t] = type_counts.get(t, 0) + 1
print(f"\nNode types: {json.dumps(type_counts, indent=2)}")

# All nodes
print(f"\n{'─' * 50}")
print("ALL NODES:")
for i, n in enumerate(g.nodes):
    st = ""
    if getattr(n.node, "state_type", None):
        reason = getattr(n.node, "state_rationale", "") or getattr(n.node, "state_reason", "") or ""
        st = f" [{n.node.state_type}" + (f": {reason[:40]}" if reason else "") + "]"
    print(f"  {i:3d}. [{n.node.type.value}]{st} {n.node.content[:100]}")

# Relationships
print(f"\n{'─' * 50}")
print("ALL RELATIONSHIPS:")
for r in g._relationships:
    ax = f" (axis: {r.axis_label})" if r.axis_label else ""
    src = g.get_node(r.source)
    tgt = g.get_node(r.target)
    s = src.content[:45] if src else "?"
    t = tgt.content[:45] if tgt else "?"
    print(f"  {r.type.value}{ax}: {s} -> {t}")

# Tensions
tension_rels = [r for r in g._relationships if r.type.value == "tension"]
print(f"\n{'─' * 50}")
print(f"TENSIONS ({len(tension_rels)}):")
for tr in tension_rels:
    src = g.get_node(tr.source)
    tgt = g.get_node(tr.target)
    axis = tr.axis_label or "unnamed"
    s = src.content[:60] if src else "?"
    t = tgt.content[:60] if tgt else "?"
    print(f"  [{axis}]")
    print(f"    {s}")
    print(f"    vs {t}")

# Decisions
decs = [n for n in g.nodes if getattr(n.node, "state_type", None) == "decided"]
print(f"\n{'─' * 50}")
print(f"DECISIONS ({len(decs)}):")
for d in decs:
    rat = getattr(d.node, "state_rationale", "") or ""
    print(f"  {d.node.content[:100]}")
    if rat:
        print(f"    rationale: {rat[:100]}")
    try:
        chain = g.query.why(d.id)
        if chain:
            print(f"    why() -> {len(chain)} step(s)")
            for step in chain[:5]:
                print(f"      -> {step}")
    except Exception as e:
        print(f"    why(): {e}")

# Blocked
blocked = [n for n in g.nodes if getattr(n.node, "state_type", None) == "blocked"]
print(f"\n{'─' * 50}")
print(f"BLOCKED ({len(blocked)}):")
for b in blocked:
    reason = getattr(b.node, "state_reason", "") or ""
    print(f"  {b.node.content[:100]}")
    if reason:
        print(f"    reason: {reason[:100]}")

# Explain first decision
if decs:
    print(f"\n{'─' * 50}")
    print("EXPLAIN (first decision, legal mode):")
    try:
        exp = explain(g, decs[0].id, audience="legal")
        print(f"  {exp[:600]}")
    except Exception as e:
        print(f"  explain() error: {e}")

# Audit trail
audit_path = mem_path.replace(".json", ".audit.jsonl")
if os.path.exists(audit_path):
    with open(audit_path) as f:
        audit_lines = f.readlines()
    print(f"\n{'─' * 50}")
    print(f"AUDIT TRAIL: {len(audit_lines)} events")
    # Show event type distribution
    event_types = {}
    for line in audit_lines:
        evt = json.loads(line)
        et = evt.get("event_type", "?")
        event_types[et] = event_types.get(et, 0) + 1
    print(f"  Event types: {json.dumps(event_types)}")
    # Show first and last entries
    first = json.loads(audit_lines[0])
    last = json.loads(audit_lines[-1])
    print(f"  First: seq={first.get('seq')}, type={first.get('event_type')}, ts={first.get('timestamp','')[:19]}")
    print(f"  Last:  seq={last.get('seq')}, type={last.get('event_type')}, ts={last.get('timestamp','')[:19]}")
    # Verify chain
    try:
        verify = g.verify_audit()
        print(f"  Chain valid: {verify.valid}")
        if verify.valid is False:
            print(f"    Break at: {verify}")
    except Exception as e:
        print(f"  Chain verification: {e}")

# JSON export stats
j = json.loads(g.to_json_string())
ir_data = j.get("ir", j)  # MemoryJSON nests under 'ir'; fall back to top-level for legacy
print(f"\n{'─' * 50}")
print(f"JSON EXPORT: {len(ir_data.get('nodes',[]))} nodes, {len(ir_data.get('relationships',[]))} rels, {len(ir_data.get('states',[]))} states")

# Final verdict
print(f"\n{'=' * 60}")
print(f"VERDICT")
print(f"  Exchanges: 5")
print(f"  Nodes: {len(g.nodes)}")
print(f"  Relationships: {g.relationship_count}")
print(f"  Tensions: {len(tension_rels)}")
print(f"  Decisions: {len(decs)}")
print(f"  Blocked: {len(blocked)}")
print(f"  Audit events: {len(audit_lines) if os.path.exists(audit_path) else 'N/A'}")
print(f"  Files: {mem_path} + {audit_path}")
print(f"{'=' * 60}")
