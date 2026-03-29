#!/usr/bin/env python3
"""Validate FlowScript wrapper — real LLM extraction on captured exchanges."""
import os, sys, json
from pathlib import Path

# Load API key
for line in (Path.home() / "Documents/flow/.env.flow").read_text().splitlines():
    if line.strip() and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from anthropic import Anthropic
from flowscript_agents import UnifiedMemory
from flowscript_agents.client import FlowScriptAnthropic

# LLM for extraction (Sonnet — cheap, fast)
raw = Anthropic()
def llm_fn(prompt):
    r = raw.messages.create(model="claude-sonnet-4-20250514", max_tokens=2048,
                            messages=[{"role": "user", "content": prompt}])
    return r.content[0].text

mem = UnifiedMemory(llm=llm_fn)
client = FlowScriptAnthropic(raw, memory=mem)

print("=" * 60)
print("WRAPPER VALIDATION — 3 exchanges, real extraction")
print("=" * 60)

# Exchange 1
print("\n--- Exchange 1: Technical decision question ---")
r1 = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=300,
    messages=[{"role": "user", "content":
        "We need to choose a database for our agent memory. Options: PostgreSQL (ACID, mature, "
        "complex) or SQLite (simple, embedded, no concurrent writes). Single-agent system, needs "
        "persistence. Which and why?"}])
print(f"  ✓ captured ({len(r1.content[0].text)} chars)")

# Exchange 2
print("--- Exchange 2: Tension / tradeoff ---")
r2 = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=300,
    messages=[{"role": "user", "content":
        "But what about multi-agent scaling? SQLite's single-writer would block us. Should we "
        "start with PostgreSQL to avoid migration pain? The tradeoff: simplicity now vs scalability later."}])
print(f"  ✓ captured ({len(r2.content[0].text)} chars)")

# Exchange 3
print("--- Exchange 3: Decision made ---")
r3 = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=300,
    messages=[{"role": "user", "content":
        "Decision: SQLite for now, migration path to PostgreSQL later. Ship 2 weeks earlier. "
        "Risk is migration cost but benefit is faster feedback. Decision is final."}])
print(f"  ✓ captured ({len(r3.content[0].text)} chars)")

# --- Analyze ---
print("\n" + "=" * 60)
print("MEMORY GRAPH")
print("=" * 60)

g = mem.memory
print(f"\nNodes ({len(g.nodes)}):")
for n in g.nodes:
    st = f" [{n.node.state_type}]" if getattr(n.node, 'state_type', None) else ""
    print(f"  [{n.node.type.value}]{st} {n.node.content[:100]}")

print(f"\nRelationships ({g.relationship_count}):")
for r in g._relationships:
    ax = f" axis={r.get('axis','')}" if isinstance(r, dict) and r.get('axis') else ""
    if isinstance(r, dict):
        print(f"  {r.get('type','?')}{ax}")
    else:
        ax = f" axis={r.axis}" if hasattr(r, 'axis') and r.axis else ""
        print(f"  {r.rel_type}{ax}: {r.source_id[:12]}→{r.target_id[:12]}")

# Queries
print("\n" + "=" * 60)
print("COMPLIANCE QUERIES")
print("=" * 60)

try:
    t = g.query.tensions()
    print(f"\ntensions(): {len(t)} found")
    for item in t:
        print(f"  {item}")
except Exception as e:
    print(f"\ntensions(): {e}")

try:
    b = g.query.blocked()
    print(f"\nblocked(): {len(b)} found")
    for item in b:
        print(f"  {item}")
except Exception as e:
    print(f"\nblocked(): {e}")

# Decisions
decisions = [n for n in g.nodes if getattr(n.node, 'state_type', None) == 'decided']
print(f"\nDecisions: {len(decisions)}")
for d in decisions:
    print(f"  {d.node.content[:80]}")
    rat = getattr(d.node, 'state_rationale', None) or getattr(d.node, 'state_reason', None)
    if rat:
        print(f"    rationale: {rat[:80]}")
    try:
        chain = g.query.why(d.id)
        print(f"    why() chain: {len(chain)} steps")
    except Exception as e:
        print(f"    why(): {e}")

# JSON snapshot
print("\n" + "=" * 60)
print("JSON EXPORT (for inspection)")
print("=" * 60)
j = json.loads(g.to_json_string())
ir_data = j.get("ir", j)  # MemoryJSON nests under 'ir'; fall back to top-level for legacy
print(f"  nodes: {len(ir_data.get('nodes',[]))}")
print(f"  relationships: {len(ir_data.get('relationships',[]))}")
print(f"  states: {len(ir_data.get('states',[]))}")

# Summary
print("\n" + "=" * 60)
total_nodes = len(g.nodes)
total_rels = g.relationship_count
print(f"RESULT: {total_nodes} nodes, {total_rels} rels, {len(decisions)} decisions")
if total_nodes >= 3 and total_rels >= 1:
    print("✓ WRAPPER PRODUCES STRUCTURED COMPLIANCE ARTIFACTS")
else:
    print("✗ NEEDS INVESTIGATION")
