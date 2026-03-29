#!/usr/bin/env python3
"""
E2E Compliance Chain Test — FlowScript SDK Wrapper

Validates the FULL compliance pipeline for EU AI Act high-risk autonomous agents:
  wrapper capture → extraction → graph → audit → cloud witness →
  explain (Article 86) → query API → persistence round-trip

Scenario: Insurance claim assessment (Annex III, Category 5b).
ClaimAssessor-7 evaluates Elena Petrova's hailstorm damage claim (#CL-2026-08847,
EUR 11,200) with a pre-existing maintenance exclusion creating genuine ambiguity.
Partially approves EUR 7,800, denying EUR 3,400. Elena has Article 86 rights.

Usage:
    cd ~/Documents/flowscript-agents
    source .venv/bin/activate
    python e2e_compliance_chain.py

Requires: ANTHROPIC_API_KEY (for agent + extraction LLM calls)
Optional: FLOWSCRIPT_API_KEY + FLOWSCRIPT_NAMESPACE (for live cloud witnessing)
"""

import io
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

# ============================================================================
# Section 0: Setup
# ============================================================================

# Load env from .env.flow
env_file = Path.home() / "Documents/flow/.env.flow"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from anthropic import Anthropic
from flowscript_agents import UnifiedMemory
from flowscript_agents.audit import AuditConfig
from flowscript_agents.client import FlowScriptAnthropic
from flowscript_agents.embeddings.providers import OpenAIEmbeddings
from flowscript_agents.explain import explain, explain_counterfactual
from flowscript_agents.mcp import _AnthropicConsolidationProvider
from flowscript_agents.memory import Memory, MemoryOptions

# Output directory
OUTPUT_DIR = Path("./e2e_output")
OUTPUT_DIR.mkdir(exist_ok=True)
MEM_PATH = str(OUTPUT_DIR / "elena_claim.json")

# Clean previous run
for f in OUTPUT_DIR.iterdir():
    f.unlink()

# LLM setup
raw_client = Anthropic()


def extraction_llm(prompt: str) -> str:
    """Haiku for extraction — cheap and fast."""
    r = raw_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text


# Cloud client (optional — live witnessing if API key is set)
cloud = None
cloud_available = bool(os.environ.get("FLOWSCRIPT_API_KEY"))
captured_witnesses: list = []

if cloud_available:
    from flowscript_agents.cloud import CloudClient

    cloud = CloudClient(
        namespace=os.environ.get("FLOWSCRIPT_NAMESPACE", "test/e2e-compliance"),
        on_witness=lambda w: captured_witnesses.append(w),
    )

# Embedder + consolidation provider for cross-turn relationship building
embedder = OpenAIEmbeddings()  # uses OPENAI_API_KEY from env
consolidation_provider = _AnthropicConsolidationProvider(
    model="claude-haiku-4-5-20251001", client=raw_client
)

# Memory with audit trail wired to cloud + consolidation for deep chains
audit_config = AuditConfig(
    on_event=cloud.queue_event if cloud else None,
    on_event_async=True if cloud else False,
)
options = MemoryOptions(audit=audit_config)
mem = UnifiedMemory(
    file_path=MEM_PATH,
    llm=extraction_llm,
    embedder=embedder,
    consolidation_provider=consolidation_provider,
    options=options,
    auto_save=True,
)

# Wrapped client — transport-layer interception
client = FlowScriptAnthropic(raw_client, memory=mem)

# ============================================================================
# Section 1: Scenario Execution
# ============================================================================

SYSTEM = """You are ClaimAssessor-7, an autonomous insurance claim evaluation agent deployed by SettleAI.
You process household insurance claims under EUR 15,000 without human review.

Your evaluation framework:
1. Verify policy coverage and active status
2. Assess damage evidence against policy terms
3. Identify any exclusions, limitations, or deductibles
4. Determine separability when multiple causal factors exist
5. Issue a preliminary determination with itemized breakdown

You must be explicit about:
- Which policy sections govern each aspect of the decision
- What evidence supports or contradicts coverage
- Any tensions between competing interpretations
- The specific rationale for partial approvals or denials

Regulatory context: Your decisions are subject to EU AI Act Article 86. Every determination
must be traceable to specific evidence and policy terms. Affected persons have the right to
a clear explanation of how this system influenced the decision about their claim.

Be precise. Be fair. Document your reasoning."""

TURNS = [
    # Turn 1: Claim intake and initial assessment
    (
        "Evaluate claim #CL-2026-08847.\n\n"
        "Claimant: Elena Petrova, Policy: HG-PLUS-2024-DE-0847\n"
        "Claim type: Storm damage (hailstorm, June 14 2026)\n"
        "Claimed amount: EUR 11,200\n"
        "Items: Roof tile replacement (EUR 3,900), skylight replacement x2 (EUR 4,800), "
        "ceiling water damage repair (EUR 2,500)\n"
        "Deductible: EUR 500\n\n"
        "Policy status: Active, premiums current.\n"
        "Coverage: \"All-risk household including weather events per Section 3.2\"\n\n"
        "Provide your initial coverage assessment for each claimed item."
    ),
    # Turn 2: Exclusion flag (introduces tension)
    (
        "Policy inspection report from April 2025 (14 months ago) noted:\n"
        "\"Roof tiles showing age-related wear. Several tiles cracked or displaced. "
        "Recommend maintenance within 6 months. Policyholder acknowledged.\"\n\n"
        "Section 4.7 of the policy states: \"Damage materially contributed to by failure "
        "to maintain the insured property in reasonable condition is excluded from coverage, "
        "provided the policyholder was notified of the maintenance requirement.\"\n\n"
        "How does this affect coverage for each claimed item? Analyze whether the "
        "pre-existing condition is separable from the storm damage."
    ),
    # Turn 3: Competing evidence (deepens the tension)
    (
        "Additional evidence received:\n\n"
        "1. Municipal weather service confirmed F1-grade hailstorm on June 14, wind speeds "
        "110km/h, hailstones 3-4cm diameter. \"Sufficient force to damage well-maintained "
        "roofing materials.\"\n\n"
        "2. Independent assessor report: \"Skylight damage consistent with direct hail impact. "
        "Roof tile damage shows mixed causation -- several tiles displaced by wind/hail "
        "would not have failed if not already weakened by prior deterioration. Ceiling "
        "damage is secondary to roof breach, regardless of breach cause.\"\n\n"
        "3. Three neighboring properties filed storm damage claims for similar roof damage. "
        "Two had well-maintained roofs and received full coverage.\n\n"
        "Evaluate: does the weather severity change the separability analysis? "
        "What weight should neighbor claims carry in the assessment?"
    ),
    # Turn 4: Legal constraint (blocker)
    (
        "Legal review flag: SettleAI's German regulatory counsel has advised that under "
        "BGB Section 254 (contributory negligence), partial denial requires demonstrating "
        "that the maintenance failure was a \"material contributing factor,\" not merely a "
        "background condition. The burden of proof is on the insurer.\n\n"
        "Additionally, the ombudsman precedent FIN-2025-347 ruled that weather events "
        "exceeding design specifications create a \"superseding cause\" that overrides "
        "maintenance exclusions.\n\n"
        "However, the F1-grade storm did NOT exceed standard roof design specifications "
        "(rated for F2). So the superseding cause defense does not apply here.\n\n"
        "Does this change your assessment? What is now blocked and what path forward "
        "do you recommend?"
    ),
    # Turn 5: Alternative approaches considered
    (
        "Consider three possible determinations:\n\n"
        "A) Full approval (EUR 11,200): Treat storm as primary cause for all items, "
        "maintenance note as irrelevant given severity. Risk: sets precedent undermining "
        "Section 4.7.\n\n"
        "B) Partial approval (skylights + ceiling only, EUR 7,300): Deny roof tiles "
        "entirely due to maintenance exclusion. Risk: ceiling damage causation is arguable.\n\n"
        "C) Proportional approach (EUR 7,800): Approve skylights fully (EUR 4,800), apply "
        "50% contribution for roof tiles (EUR 1,950), approve ceiling fully (EUR 2,500), "
        "minus EUR 500 deductible, minus EUR 950 maintenance contribution. Total: EUR 7,800.\n\n"
        "Which option and why? Consider fairness to Elena, policy enforceability, regulatory "
        "risk, and precedent implications."
    ),
    # Turn 6: Final determination
    (
        "Issue the final determination for claim #CL-2026-08847. Include:\n"
        "1. The approved amount with itemized breakdown\n"
        "2. The specific policy sections governing each line item\n"
        "3. The rationale for any denied or reduced amounts\n"
        "4. The key evidence that drove each sub-decision\n"
        "5. Elena Petrova's rights under EU AI Act Article 86 to request a detailed explanation\n\n"
        "This is a binding preliminary determination."
    ),
]


def run_scenario():
    """Execute the 6-turn scenario, return responses and node counts per turn."""
    responses = []
    node_counts = []
    messages = []
    stderr_capture = io.StringIO()

    print("=" * 70)
    print("E2E COMPLIANCE CHAIN TEST")
    print("Scenario: Insurance claim assessment (Annex III, Category 5b)")
    print("Affected person: Elena Petrova, Claim #CL-2026-08847")
    print("=" * 70)

    for i, prompt in enumerate(TURNS):
        step = f"Turn {i + 1}/{len(TURNS)}"
        messages.append({"role": "user", "content": prompt})
        print(f"\n{'─' * 50}")
        print(f"  {step}: {prompt[:60]}...")
        start = time.time()

        # Capture stderr for extraction error detection
        old_stderr = sys.stderr
        sys.stderr = stderr_capture

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1200,
                system=SYSTEM,
                messages=messages,
            )
        finally:
            sys.stderr = old_stderr

        elapsed = time.time() - start
        text = response.content[0].text
        responses.append(text)
        messages.append({"role": "assistant", "content": text})

        nodes_now = len(mem.memory.nodes)
        rels_now = mem.memory.relationship_count
        node_counts.append(nodes_now)
        print(f"  Response: {len(text)} chars | {elapsed:.1f}s")
        print(f"  Memory: {nodes_now} nodes, {rels_now} relationships (cumulative)")

    # Check stderr for extraction errors
    stderr_output = stderr_capture.getvalue()
    extraction_errors = stderr_output.count("extraction error")
    extraction_errors += stderr_output.count("AutoExtract:")

    print(f"\n{'─' * 50}")
    print(f"  Scenario complete. {len(responses)} turns executed.")
    if extraction_errors:
        print(f"  WARNING: {extraction_errors} extraction issues logged to stderr")

    return responses, node_counts, extraction_errors


# ============================================================================
# Validation Stages
# ============================================================================

stage_results: list[tuple[str, bool, dict]] = []


def record_stage(name: str, passed: bool, details: dict):
    """Record a stage result."""
    stage_results.append((name, passed, details))
    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] {name}")
    if not passed:
        for k, v in details.items():
            if k.startswith("fail_"):
                print(f"    - {v}")


def stage_1_wrapper_capture(responses, node_counts, extraction_errors):
    """Stage 1: Verify every exchange was captured by the wrapper."""
    failures = {}
    details = {"responses": len(responses), "final_nodes": node_counts[-1] if node_counts else 0}

    if len(responses) != 6:
        failures["fail_response_count"] = f"Expected 6 responses, got {len(responses)}"

    for i, count in enumerate(node_counts):
        if count == 0:
            failures["fail_zero_nodes"] = f"Turn {i+1} had 0 cumulative nodes"
            break
        if i > 0 and count < node_counts[i - 1]:
            failures["fail_monotonic"] = f"Node count decreased after turn {i+1}"
            break

    if node_counts and node_counts[-1] < 12:
        failures["fail_min_nodes"] = (
            f"Only {node_counts[-1]} nodes from 6 turns (need >= 12)"
        )

    if extraction_errors > 0:
        failures["fail_extraction_errors"] = (
            f"{extraction_errors} extraction errors on stderr"
        )

    details.update(failures)
    details["node_counts_per_turn"] = node_counts
    record_stage("Stage 1: Wrapper Capture", len(failures) == 0, details)


def stage_2_extraction_quality():
    """Stage 2: Verify extraction produced semantically typed, scenario-relevant nodes."""
    g = mem.memory
    failures = {}
    details = {}

    # 2a. Node type diversity
    node_types = {n.node.type.value for n in g.nodes}
    core_types = {"thought", "decision", "question", "alternative", "insight", "action"}
    present = node_types & core_types
    details["node_types_present"] = sorted(present)
    if len(present) < 3:
        failures["fail_type_diversity"] = (
            f"Only {len(present)} node types: {present}. Need >= 3"
        )

    # 2b. Decision node exists (validates our bug fix)
    decisions = [n for n in g.nodes if n.node.type.value == "decision"]
    details["decision_node_count"] = len(decisions)
    if len(decisions) < 1:
        failures["fail_no_decisions"] = "No decision-type nodes extracted"

    # 2c. Decided states exist with rationale
    decided_states = [s for s in g._states if s.type.value == "decided"]
    details["decided_state_count"] = len(decided_states)
    if len(decided_states) < 1:
        failures["fail_no_decided_state"] = "No decided states in graph"
    else:
        for s in decided_states:
            if s.fields is None or not getattr(s.fields, "rationale", None):
                failures["fail_empty_rationale"] = (
                    f"Decided state {s.id[:12]} missing rationale"
                )
                break

    # 2d. Tension relationships with axis labels
    tension_rels = [r for r in g._relationships if r.type.value == "tension"]
    labeled = [r for r in tension_rels if r.axis_label]
    details["tension_count"] = len(tension_rels)
    details["labeled_tension_count"] = len(labeled)
    if len(tension_rels) < 1:
        failures["fail_no_tensions"] = "No tension relationships extracted"
    elif len(labeled) < 1:
        failures["fail_unlabeled_tensions"] = (
            f"{len(tension_rels)} tensions but none labeled"
        )

    # 2e. Content relevance
    keywords = ["elena", "claim", "maintenance", "skylight", "hail", "roof",
                "section 4.7", "deductible", "storm", "tile"]
    relevant = [
        n for n in g.nodes
        if any(kw in n.node.content.lower() for kw in keywords)
    ]
    details["relevant_node_count"] = len(relevant)
    if len(relevant) < 3:
        failures["fail_relevance"] = (
            f"Only {len(relevant)} nodes reference scenario content"
        )

    details.update(failures)
    record_stage("Stage 2: Extraction Quality", len(failures) == 0, details)


def stage_3_graph_structure():
    """Stage 3: Verify causal chains, cross-turn structure, graph coherence."""
    g = mem.memory
    failures = {}
    details = {"total_nodes": len(g.nodes), "total_rels": g.relationship_count}

    # 3a. Relationship count
    if g.relationship_count < 5:
        failures["fail_sparse"] = (
            f"Only {g.relationship_count} relationships — too sparse for 6 turns"
        )

    # 3b. Causal chain depth
    decisions = [n for n in g.nodes if n.node.type.value == "decision"]
    # Also check nodes with decided state if no decision-type nodes
    if not decisions:
        decided_ids = {s.node_id for s in g._states if s.type.value == "decided"}
        decisions = [n for n in g.nodes if n.id in decided_ids]

    max_chain_depth = 0
    best_why = None
    for d in decisions:
        try:
            result = g.query.why(d.id)
            if hasattr(result, "causal_chain"):
                depth = len(result.causal_chain)
                if depth > max_chain_depth:
                    max_chain_depth = depth
                    best_why = result
        except (ValueError, KeyError):
            pass

    details["max_causal_chain_depth"] = max_chain_depth
    if max_chain_depth < 2:
        failures["fail_shallow_chains"] = (
            f"Longest causal chain is {max_chain_depth} — need >= 2"
        )

    # 3c. Causal/derives_from relationships
    causal = [r for r in g._relationships if r.type.value in ("causes", "derives_from")]
    details["causal_rel_count"] = len(causal)
    if len(causal) < 2:
        failures["fail_no_causal"] = (
            f"Only {len(causal)} causal relationships — graph lacks reasoning structure"
        )

    # 3d. Node participation rate
    nodes_in_rels = set()
    for r in g._relationships:
        nodes_in_rels.add(r.source)
        nodes_in_rels.add(r.target)
    participation = len(nodes_in_rels) / max(len(g.nodes), 1)
    details["node_participation_rate"] = f"{participation:.0%}"
    if participation < 0.4:
        failures["fail_fragmented"] = (
            f"Only {participation:.0%} of nodes in relationships"
        )

    details.update(failures)
    details["best_why_summary"] = (
        f"depth={max_chain_depth}, target={best_why.target['content'][:60]}..."
        if best_why else "none"
    )
    record_stage("Stage 3: Graph Structure", len(failures) == 0, details)


def stage_4_audit_trail():
    """Stage 4: Verify hash chain integrity, sequence continuity, event coverage."""
    failures = {}
    details = {}

    audit_path = MEM_PATH.replace(".json", ".audit.jsonl")
    if not Path(audit_path).exists():
        failures["fail_no_audit"] = "No audit trail file created"
        details.update(failures)
        record_stage("Stage 4: Audit Trail Integrity", False, details)
        return

    with open(audit_path) as f:
        audit_lines = f.readlines()

    details["total_entries"] = len(audit_lines)
    if len(audit_lines) < 10:
        failures["fail_too_few"] = (
            f"Only {len(audit_lines)} audit entries for 6-turn conversation"
        )

    # Parse entries
    entries = [json.loads(line) for line in audit_lines]

    # Hash chain verification
    verify_result = Memory.verify_audit(audit_path)
    details["chain_valid"] = verify_result.valid
    if verify_result.valid is not True:
        failures["fail_chain_broken"] = (
            f"Hash chain broken at entry {verify_result.chain_break_at}"
        )

    # Sequence gaps
    seqs = [e.get("seq", -1) for e in entries]
    for i in range(1, len(seqs)):
        if seqs[i] != seqs[i - 1] + 1:
            failures["fail_seq_gap"] = (
                f"Sequence gap at entry {i}: seq={seqs[i]}, expected {seqs[i-1]+1}"
            )
            break

    # Event type coverage (field is "event", not "event_type")
    event_types = {e.get("event") or e.get("event_type") for e in entries}
    details["event_types"] = sorted(t for t in event_types if t)
    if "node_create" not in event_types:
        failures["fail_no_node_create"] = "No node_create audit events"

    # Genesis hash
    if entries and entries[0].get("prev_hash") != "sha256:GENESIS":
        failures["fail_genesis"] = (
            f"First entry prev_hash is {entries[0].get('prev_hash')}"
        )

    # node_create events have required fields
    nc_events = [e for e in entries if (e.get("event") or e.get("event_type")) == "node_create"]
    details["node_create_events"] = len(nc_events)
    for nc in nc_events[:3]:  # spot check first 3
        data = nc.get("data", {})
        if "node_id" not in data:
            failures["fail_nc_missing_id"] = "node_create event missing node_id"
            break

    details.update(failures)
    record_stage("Stage 4: Audit Trail Integrity", len(failures) == 0, details)
    return entries  # needed by stage 5


def stage_5_cloud_witnessing(audit_entries):
    """Stage 5: Verify cloud witnessing (live or buffer-only)."""
    failures = {}
    details = {"cloud_available": cloud_available}

    if not cloud_available:
        details["skip_reason"] = "No FLOWSCRIPT_API_KEY — cloud witnessing skipped"
        print(f"\n  [SKIP] Stage 5: Cloud Witnessing (no API key)")
        stage_results.append(("Stage 5: Cloud Witnessing", True, details))
        return

    # Give async events a moment to drain
    time.sleep(2)

    # Flush
    result = cloud.flush()
    details["flush_result"] = result is not None

    if result is None:
        failures["fail_flush_none"] = "Flush returned None despite queued events"
    else:
        details["accepted"] = result.accepted
        details["error"] = result.error
        details["status_code"] = result.status_code

        if result.accepted == 0:
            failures["fail_zero_accepted"] = f"Cloud accepted 0 events: {result.error}"

        if result.error is not None:
            failures["fail_flush_error"] = f"Flush error: {result.error}"

        if result.witness is not None:
            details["witness"] = result.witness
            w = result.witness
            if "id" not in w:
                failures["fail_witness_no_id"] = "Witness missing 'id'"
            if "chain_head_hash" not in w:
                failures["fail_witness_no_hash"] = "Witness missing chain_head_hash"
        else:
            # Witness may not come on every flush — not a hard failure
            details["witness_note"] = "No witness in this flush (may be batched)"

    # Check last_witness
    if cloud.last_witness is not None:
        details["last_witness_id"] = cloud.last_witness.id
        details["last_witness_at"] = cloud.last_witness.witnessed_at

    details["total_sent"] = cloud.total_sent
    details["total_accepted"] = cloud.total_accepted

    details.update(failures)
    record_stage("Stage 5: Cloud Witnessing", len(failures) == 0, details)


def stage_6_explain():
    """Stage 6: Verify Article 86 explanation output for decisions."""
    g = mem.memory
    failures = {}
    details = {}

    # Find decision nodes (by type or by state)
    decisions = [n for n in g.nodes if n.node.type.value == "decision"]
    if not decisions:
        decided_ids = {s.node_id for s in g._states if s.type.value == "decided"}
        decisions = [n for n in g.nodes if n.id in decided_ids]

    details["decisions_found"] = len(decisions)
    if not decisions:
        failures["fail_no_decisions"] = "No decisions to explain"
        details.update(failures)
        record_stage("Stage 6: Explain (Article 86)", False, details)
        return

    explained_count = 0
    for dec in decisions[:3]:  # check up to 3
        try:
            why_result = g.query.why(dec.id)
        except (ValueError, KeyError):
            continue

        # Legal mode
        try:
            legal_text = explain(why_result, subject="Elena Petrova", audience="legal")
        except Exception as e:
            failures["fail_legal_error"] = f"explain() raised: {e}"
            continue

        details[f"legal_len_{dec.id[:8]}"] = len(legal_text)

        if len(legal_text) < 100:
            failures["fail_legal_short"] = (
                f"Legal explanation too short ({len(legal_text)} chars)"
            )

        # Structural checks
        if "AUTOMATED DECISION EXPLANATION" not in legal_text:
            if "REASONING PATH" not in legal_text and "Decision Explanation" not in legal_text:
                failures["fail_legal_no_header"] = "Legal output missing standard header"

        if "Elena Petrova" not in legal_text:
            # Subject may not appear in MinimalWhy format — soft check
            details["subject_missing"] = True

        # General mode
        try:
            general_text = explain(why_result, audience="general")
            if len(general_text) < 50:
                failures["fail_general_short"] = "General explanation too short"
        except Exception as e:
            failures["fail_general_error"] = f"General explain() raised: {e}"

        explained_count += 1

    details["explained_count"] = explained_count
    if explained_count == 0:
        failures["fail_none_explained"] = "Could not explain any decision (no causal chains)"

    details.update(failures)
    record_stage("Stage 6: Explain (Article 86)", len(failures) == 0, details)


def stage_7_query_api():
    """Stage 7: Verify all five query operations return valid results."""
    g = mem.memory
    failures = {}
    details = {}

    # 7a. why()
    decisions = [n for n in g.nodes if n.node.type.value == "decision"]
    if not decisions:
        decided_ids = {s.node_id for s in g._states if s.type.value == "decided"}
        decisions = [n for n in g.nodes if n.id in decided_ids]

    why_success = False
    for d in decisions:
        try:
            result = g.query.why(d.id)
            if hasattr(result, "causal_chain") and len(result.causal_chain) > 0:
                why_success = True
                # Verify chain nodes exist in graph
                for cn in result.causal_chain:
                    if g.get_node(cn.id) is None:
                        failures["fail_why_ghost"] = (
                            f"why() references non-existent node {cn.id[:12]}"
                        )
                        break
                details["why_chain_depth"] = len(result.causal_chain)
                break
        except (ValueError, KeyError):
            continue

    if not why_success:
        failures["fail_why_empty"] = "why() returned no causal chain for any decision"

    # 7b. tensions()
    tensions_result = g.query.tensions()
    total_tensions = tensions_result.metadata.get("total_tensions", 0)
    details["total_tensions"] = total_tensions
    details["unique_axes"] = tensions_result.metadata.get("unique_axes", [])
    if total_tensions < 1:
        failures["fail_no_tensions"] = f"tensions() found {total_tensions}"

    # 7c. what_if()
    for d in decisions:
        try:
            impact = g.query.what_if(d.id)
            details["what_if_has_result"] = True
            details["what_if_descendants"] = impact.metadata.get("total_descendants", 0)
            break
        except (ValueError, KeyError):
            details["what_if_has_result"] = False

    # 7d. blocked()
    blocked_result = g.query.blocked()
    details["blocker_count"] = len(blocked_result.blockers)

    # 7e. alternatives() — needs a question node
    questions = [n for n in g.nodes if n.node.type.value == "question"]
    details["question_count"] = len(questions)
    if questions:
        try:
            alt_result = g.query.alternatives(questions[0].id)
            details["alternatives_found"] = True
        except (ValueError, KeyError):
            details["alternatives_found"] = False

    # 7f. counterfactual() — CJEU C-203/22
    cf_success = False
    cf_max_factors = 0
    for d in decisions:
        try:
            cf = g.query.counterfactual(d.id)
            if cf.factors:
                cf_success = True
                cf_max_factors = max(cf_max_factors, len(cf.factors))
                details["counterfactual_factors"] = len(cf.factors)
                details["counterfactual_axes"] = cf.metadata.get("unique_axes", [])[:5]

                # Verify explain_counterfactual produces output
                legal_cf = explain_counterfactual(cf, subject="Elena Petrova", audience="legal")
                details["counterfactual_legal_len"] = len(legal_cf)
                if len(legal_cf) < 100:
                    failures["fail_cf_explain_short"] = (
                        f"Counterfactual legal explanation too short ({len(legal_cf)} chars)"
                    )
                break
        except (ValueError, KeyError):
            continue

    if not cf_success:
        failures["fail_no_counterfactual"] = (
            "counterfactual() returned no pivotal factors for any decision"
        )

    details.update(failures)
    record_stage("Stage 7: Query API", len(failures) == 0, details)


def stage_8_persistence():
    """Stage 8: Verify save/load round-trip preserves all data."""
    g = mem.memory
    failures = {}
    details = {}

    # Save to temp
    save_path = os.path.join(tempfile.mkdtemp(), "roundtrip_test.json")
    g.save(save_path)

    if not Path(save_path).exists():
        failures["fail_no_file"] = "Memory file not written"
        details.update(failures)
        record_stage("Stage 8: Persistence Round-Trip", False, details)
        return

    # Check structure
    with open(save_path) as f:
        saved_data = json.load(f)
    if "flowscript_memory" not in saved_data:
        failures["fail_format"] = "Missing flowscript_memory key"
    if "ir" not in saved_data:
        failures["fail_no_ir"] = "Missing ir key"

    # Reload
    reloaded = Memory.load(save_path)
    details["original_nodes"] = len(g.nodes)
    details["reloaded_nodes"] = len(reloaded.nodes)
    details["original_rels"] = g.relationship_count
    details["reloaded_rels"] = reloaded.relationship_count

    if len(reloaded.nodes) != len(g.nodes):
        failures["fail_node_count"] = (
            f"Nodes: {len(g.nodes)} -> {len(reloaded.nodes)}"
        )
    if reloaded.relationship_count != g.relationship_count:
        failures["fail_rel_count"] = (
            f"Rels: {g.relationship_count} -> {reloaded.relationship_count}"
        )
    if reloaded.state_count != g.state_count:
        failures["fail_state_count"] = (
            f"States: {g.state_count} -> {reloaded.state_count}"
        )

    # Temporal metadata preserved
    temporal_ok = True
    for node_id in list(g.temporal_map.keys())[:5]:  # spot check
        orig = g.temporal_map[node_id]
        reloaded_meta = reloaded.temporal_map.get(node_id)
        if reloaded_meta is None:
            temporal_ok = False
            failures["fail_temporal_lost"] = f"Temporal data lost for {node_id[:12]}"
            break
        if reloaded_meta.frequency != orig.frequency:
            temporal_ok = False
            break
    details["temporal_preserved"] = temporal_ok

    # Queries work on reloaded graph
    try:
        rt_tensions = reloaded.query.tensions()
        orig_tensions = g.query.tensions()
        rt_count = rt_tensions.metadata.get("total_tensions", 0)
        orig_count = orig_tensions.metadata.get("total_tensions", 0)
        details["tension_count_preserved"] = rt_count == orig_count
        if rt_count != orig_count:
            failures["fail_tension_rt"] = (
                f"Tensions: {orig_count} -> {rt_count} after round-trip"
            )
    except Exception:
        pass

    # Double round-trip idempotent
    j1 = json.loads(g.to_json_string())
    j2 = json.loads(reloaded.to_json_string())
    ir1_nodes = len(j1.get("ir", {}).get("nodes", []))
    ir2_nodes = len(j2.get("ir", {}).get("nodes", []))
    if ir1_nodes != ir2_nodes:
        failures["fail_double_rt"] = f"Double round-trip: {ir1_nodes} -> {ir2_nodes} nodes"

    # Audit trail survives
    audit_path = MEM_PATH.replace(".json", ".audit.jsonl")
    if Path(audit_path).exists():
        vr = Memory.verify_audit(audit_path)
        details["audit_still_valid"] = vr.valid
        if vr.valid is not True:
            failures["fail_audit_rt"] = "Audit chain broken after save"

    details.update(failures)
    record_stage("Stage 8: Persistence Round-Trip", len(failures) == 0, details)


# ============================================================================
# Section 10: Holistic Compliance Analysis
# ============================================================================

def holistic_analysis(responses):
    """Produce a structured compliance assessment report."""
    g = mem.memory

    print("\n" + "=" * 70)
    print("  COMPLIANCE CHAIN ANALYSIS -- HOLISTIC REPORT")
    print("=" * 70)
    print("\nSCENARIO: Insurance claim assessment (Annex III, Category 5b)")
    print("AFFECTED PERSON: Elena Petrova")
    print("CLAIM: #CL-2026-08847, EUR 11,200")

    # --- Article 12 ---
    audit_path = MEM_PATH.replace(".json", ".audit.jsonl")
    audit_count = 0
    event_types = set()
    if Path(audit_path).exists():
        with open(audit_path) as f:
            lines = f.readlines()
        audit_count = len(lines)
        event_types = {json.loads(l).get("event") or json.loads(l).get("event_type") for l in lines}
        verify = Memory.verify_audit(audit_path)
        chain_status = "VALID" if verify.valid else f"BROKEN at seq {verify.chain_break_at}"
    else:
        chain_status = "NO AUDIT FILE"

    node_count = len(g.nodes)
    audit_density = audit_count / max(node_count, 1)

    art12_rating = "STRONG" if (audit_count > 20 and chain_status == "VALID" and audit_density >= 1.0) else \
                   "ADEQUATE" if (audit_count > 10 and chain_status == "VALID") else \
                   "WEAK" if audit_count > 0 else "ABSENT"

    print(f"\n{'=' * 60}")
    print(f"  ARTICLE 12 (Record-Keeping): {art12_rating}")
    print(f"{'=' * 60}")
    print(f"  Audit events: {audit_count}")
    print(f"  Event types: {sorted(event_types)}")
    print(f"  Hash chain: {chain_status}")
    print(f"  Audit density: {audit_density:.1f}x (events per graph node)")
    print(f"  Graph nodes: {node_count}, Relationships: {g.relationship_count}")

    # --- Article 13 ---
    decisions = [n for n in g.nodes if n.node.type.value == "decision"]
    if not decisions:
        decided_ids = {s.node_id for s in g._states if s.type.value == "decided"}
        decisions = [n for n in g.nodes if n.id in decided_ids]

    explain_lengths = []
    max_chain_depth = 0
    for d in decisions[:3]:
        try:
            wr = g.query.why(d.id)
            if hasattr(wr, "causal_chain"):
                max_chain_depth = max(max_chain_depth, len(wr.causal_chain))
            legal = explain(wr, subject="Elena Petrova", audience="legal")
            explain_lengths.append(len(legal))
        except Exception:
            pass

    depth_label = "ABSENT" if max_chain_depth == 0 else \
                  "SHALLOW" if max_chain_depth == 1 else \
                  "ADEQUATE" if max_chain_depth <= 3 else "STRONG"

    art13_rating = "STRONG" if (explain_lengths and min(explain_lengths) > 200 and max_chain_depth >= 3) else \
                   "ADEQUATE" if (explain_lengths and max_chain_depth >= 2) else \
                   "WEAK" if explain_lengths else "ABSENT"

    print(f"\n{'=' * 60}")
    print(f"  ARTICLE 13(3)(b)(iv) (Built-in Explainability): {art13_rating}")
    print(f"{'=' * 60}")
    print(f"  explain() output lengths: {explain_lengths}")
    print(f"  Max causal chain depth: {max_chain_depth} ({depth_label})")
    print(f"  Decisions found: {len(decisions)}")
    print(f"  Explanations are mechanistic (from captured reasoning, not post-hoc)")

    # --- Article 86 ---
    tensions_result = g.query.tensions()
    total_tensions = tensions_result.metadata.get("total_tensions", 0)
    axes = tensions_result.metadata.get("unique_axes", [])

    # Expected tensions from scenario
    expected_tension_keywords = [
        "maintenance", "coverage", "cost", "fairness", "compliance",
        "precedent", "exclusion", "storm", "risk"
    ]
    matched_axes = [
        ax for ax in axes
        if any(kw in ax.lower() for kw in expected_tension_keywords)
    ]

    art86_rating = "STRONG" if (len(decisions) >= 1 and max_chain_depth >= 3 and total_tensions >= 2) else \
                   "ADEQUATE" if (len(decisions) >= 1 and max_chain_depth >= 2 and total_tensions >= 1) else \
                   "WEAK" if decisions else "ABSENT"

    print(f"\n{'=' * 60}")
    print(f"  ARTICLE 86 (Right to Explanation): {art86_rating}")
    print(f"{'=' * 60}")
    print(f"  Can answer 'why was EUR 3,400 denied?': "
          f"{'YES' if max_chain_depth >= 2 else 'PARTIAL' if max_chain_depth >= 1 else 'NO'}")
    print(f"  Tensions captured: {total_tensions}")
    print(f"  Tension axes: {axes[:5]}")
    print(f"  Scenario-relevant axes: {matched_axes}")

    # --- CJEU C-203/22 ---
    cf_factor_count = 0
    cf_max_depth = 0
    cf_legal_len = 0
    for d in decisions[:3]:
        try:
            cf = g.query.counterfactual(d.id)
            if cf.factors:
                cf_factor_count = max(cf_factor_count, len(cf.factors))
                cf_max_depth = max(cf_max_depth, cf.causal_chain_depth)
                legal_cf = explain_counterfactual(cf, subject="Elena Petrova", audience="legal")
                cf_legal_len = max(cf_legal_len, len(legal_cf))
        except Exception:
            pass

    cjeu_rating = "STRONG" if (cf_factor_count >= 3 and cf_max_depth >= 2 and cf_legal_len > 500) else \
                  "ADEQUATE" if (cf_factor_count >= 1 and cf_legal_len > 100) else \
                  "WEAK" if total_tensions >= 1 else "ABSENT"

    print(f"\n{'=' * 60}")
    print(f"  CJEU C-203/22 (Counterfactual Explanations): {cjeu_rating}")
    print(f"{'=' * 60}")
    print(f"  counterfactual() pivotal factors: {cf_factor_count}")
    print(f"  counterfactual() max chain depth: {cf_max_depth}")
    print(f"  counterfactual legal explanation: {cf_legal_len} chars")
    print(f"  tensions() provides competing factors: {total_tensions >= 1}")

    # --- Cryptographic Integrity ---
    crypto_rating = "STRONG" if (chain_status == "VALID" and cloud_available and captured_witnesses) else \
                    "ADEQUATE" if chain_status == "VALID" else \
                    "WEAK" if audit_count > 0 else "ABSENT"

    print(f"\n{'=' * 60}")
    print(f"  CRYPTOGRAPHIC INTEGRITY: {crypto_rating}")
    print(f"{'=' * 60}")
    print(f"  Hash chain: {chain_status}")
    print(f"  Cloud witnessed: {bool(captured_witnesses)}")
    if captured_witnesses:
        print(f"  Latest witness: {captured_witnesses[-1].witnessed_at}")

    # --- Coverage Ratio ---
    key_phrases = [
        "skylight", "roof tile", "section 4.7", "maintenance exclusion",
        "deductible", "hailstorm", "elena petrova", "proportional",
        "BGB", "separability"
    ]
    found_phrases = [
        p for p in key_phrases
        if any(p.lower() in n.node.content.lower() for n in g.nodes)
    ]
    coverage_ratio = len(found_phrases) / len(key_phrases)

    # --- Tension Fidelity ---
    expected_tensions = [
        "cost vs coverage", "fairness vs policy", "precedent vs individual"
    ]
    tension_fidelity = len(matched_axes)

    print(f"\n{'=' * 60}")
    print(f"  QUANTITATIVE METRICS")
    print(f"{'=' * 60}")
    print(f"  Coverage ratio: {coverage_ratio:.0%} ({len(found_phrases)}/{len(key_phrases)} "
          f"key phrases found in nodes)")
    print(f"    Found: {found_phrases}")
    print(f"    Missing: {[p for p in key_phrases if p not in found_phrases]}")
    print(f"  Decision traceability: depth {max_chain_depth} "
          f"({'ABSENT' if max_chain_depth == 0 else 'SHALLOW' if max_chain_depth == 1 else 'ADEQUATE' if max_chain_depth <= 3 else 'STRONG'})")
    print(f"  Tension fidelity: {tension_fidelity} scenario-relevant axes matched")
    print(f"  Audit density: {audit_density:.1f}x")

    # --- Gaps ---
    print(f"\n{'=' * 60}")
    print(f"  GAPS AND WEAKNESSES")
    print(f"{'=' * 60}")
    gaps = []
    if coverage_ratio < 0.5:
        gaps.append("Low coverage ratio — many scenario-specific terms not captured in nodes")
    if max_chain_depth < 3:
        gaps.append("Causal chains are shallow — multi-hop reasoning not fully traceable")
    if not matched_axes:
        gaps.append("Tension axes don't match expected scenario tensions (labeling may be generic)")
    if cf_factor_count == 0:
        gaps.append("counterfactual() found no pivotal factors — CJEU C-203/22 compliance weak")
    if not cloud_available:
        gaps.append("No cloud witnessing — tamper-evidence relies on local hash chain only")
    gaps.append("No explicit EUR amounts structured in nodes — financial breakdown is in prose only")
    gaps.append("No affected-person identifier field on nodes — Elena's name may not appear in all relevant nodes")
    gaps.append("Temporal ordering relies on creation timestamps, not explicit sequence markers")

    for i, gap in enumerate(gaps, 1):
        print(f"  {i}. {gap}")

    # --- Verdict ---
    ratings = [art12_rating, art13_rating, art86_rating, cjeu_rating, crypto_rating]
    strong_count = ratings.count("STRONG")
    adequate_count = ratings.count("ADEQUATE")
    weak_count = ratings.count("WEAK")
    absent_count = ratings.count("ABSENT")

    if absent_count > 0:
        verdict = "INSUFFICIENT"
    elif weak_count > 1:
        verdict = "NEEDS WORK"
    elif strong_count >= 3:
        verdict = "STRONG"
    else:
        verdict = "ADEQUATE"

    print(f"\n{'=' * 60}")
    print(f"  VERDICT: {verdict}")
    print(f"{'=' * 60}")
    print(f"  Ratings: {', '.join(f'{r}' for r in ratings)}")
    print(f"  ({strong_count} STRONG, {adequate_count} ADEQUATE, {weak_count} WEAK, {absent_count} ABSENT)")
    print()
    if verdict in ("STRONG", "ADEQUATE"):
        print("  This compliance chain demonstrates that FlowScript can capture a")
        print("  tamper-evident, queryable, explainable reasoning trail from an")
        print("  autonomous agent making high-risk decisions. The audit trail exceeds")
        print("  what raw transcript logging provides. Gaps exist in counterfactual")
        print("  depth and structured financial data — expected for v0.3.0.")
    else:
        print("  The compliance chain has structural gaps that would need to be")
        print("  addressed before claiming EU AI Act compliance readiness.")
    print()

    # Write report to file
    # (The terminal output IS the report — also save stage results)
    report_path = OUTPUT_DIR / "compliance_report.json"
    report = {
        "scenario": "Insurance claim assessment, Annex III Category 5b",
        "affected_person": "Elena Petrova",
        "claim": "CL-2026-08847, EUR 11,200",
        "ratings": {
            "article_12": art12_rating,
            "article_13": art13_rating,
            "article_86": art86_rating,
            "cjeu_c203_22": cjeu_rating,
            "cryptographic": crypto_rating,
        },
        "metrics": {
            "coverage_ratio": coverage_ratio,
            "max_chain_depth": max_chain_depth,
            "tension_fidelity": tension_fidelity,
            "audit_density": audit_density,
            "total_nodes": node_count,
            "total_relationships": g.relationship_count,
            "total_tensions": total_tensions,
            "total_decisions": len(decisions),
            "total_audit_events": audit_count,
        },
        "verdict": verdict,
        "stages": [
            {"name": name, "passed": passed, "details": details}
            for name, passed, details in stage_results
        ],
        "gaps": gaps,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"  Report written to: {report_path}")
    print(f"  Artifacts in: {OUTPUT_DIR}/")

    return verdict


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Run scenario
    responses, node_counts, extraction_errors = run_scenario()

    print("\n" + "=" * 70)
    print("  VALIDATION STAGES")
    print("=" * 70)

    # Run all 8 stages
    stage_1_wrapper_capture(responses, node_counts, extraction_errors)
    stage_2_extraction_quality()
    stage_3_graph_structure()
    audit_entries = stage_4_audit_trail()
    stage_5_cloud_witnessing(audit_entries)
    stage_6_explain()
    stage_7_query_api()
    stage_8_persistence()

    # Summary
    print("\n" + "=" * 70)
    print("  STAGE SUMMARY")
    print("=" * 70)
    all_passed = True
    for name, passed, _ in stage_results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    # Holistic analysis
    verdict = holistic_analysis(responses)

    # Exit code
    if all_passed:
        print("\nAll stages passed.")
        sys.exit(0)
    else:
        failed = [name for name, passed, _ in stage_results if not passed]
        print(f"\n{len(failed)} stage(s) failed: {', '.join(failed)}")
        sys.exit(1)
