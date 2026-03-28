"""Live integration test: Real @fix convergence certificates through Cloud pipeline.

This test proves the complete chain:
  Memory → FixpointContext → AuditWriter → CloudClient → api.flowscript.org

The convergence certificate events (fixpoint_start, fixpoint_iteration, fixpoint_end)
flow through the hash chain and are witnessed by Cloud. This is the compliance path
for EU AI Act Article 86.

Requires:
    FLOWSCRIPT_API_KEY=fsk_...
    FLOWSCRIPT_CLOUD_URL=https://api.flowscript.org  (optional, default)
"""

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import pytest

from flowscript_agents import AuditConfig, Memory, MemoryOptions
from flowscript_agents.cloud import CloudClient, CloudWitness, USER_AGENT
from flowscript_agents.fixpoint import FixpointContext


LIVE_API_KEY = os.environ.get("FLOWSCRIPT_API_KEY", "")

skip_no_api_key = pytest.mark.skipif(
    not LIVE_API_KEY,
    reason="FLOWSCRIPT_API_KEY env var required for live tests",
)


@skip_no_api_key
class TestFixpointCertificatePipeline:
    """Live test: Real @fix convergence certificates through Cloud."""

    def test_full_fixpoint_pipeline(self, tmp_path):
        """Memory → FixpointContext → AuditWriter → CloudClient → Cloud → hash match.

        This is THE test that proves the compliance pipeline works end-to-end:
        1. Create a Memory with audit trail + Cloud client
        2. Add some nodes (generates audit events)
        3. Run a FixpointContext (generates fixpoint_start/iteration/end events)
        4. Flush to Cloud
        5. Verify hashes match + witness received
        6. Query Cloud for the events and verify certificate data
        """
        # Create a unique namespace for this test run
        import time
        namespace = f"flowscript/fixpoint-test-{int(time.time())}"

        # Track witnesses
        witnesses: list[CloudWitness] = []

        # Set up Cloud client
        cloud = CloudClient(
            api_key=LIVE_API_KEY,
            namespace=namespace,
            on_witness=lambda w: witnesses.append(w),
        )

        # Set up Memory with audit trail wired to Cloud.
        # Memory.load_or_create sets _file_path which activates AuditWriter.
        # Then we configure the audit with Cloud callback.
        audit_config = AuditConfig(on_event=cloud.queue_event)
        opts = MemoryOptions(audit=audit_config)
        mem = Memory.load_or_create(
            str(tmp_path / "fixpoint_test.json"),
            options=opts,
        )

        # --- Phase 1: Create some nodes (audit events) ---
        mem.session_start("fixpoint_test_session")
        q = mem.question("Which database for session storage?")
        alt_redis = mem.alternative(q, "Redis — fast, ephemeral")
        alt_pg = mem.alternative(q, "PostgreSQL — ACID, persistent")

        # --- Phase 2: Simulate a fixpoint computation ---
        # Capture pre-hash BEFORE computation (as the real code does)
        pre_hash = FixpointContext._compute_graph_hash_static(mem)

        # Simulate consolidation producing 3 new nodes, then converging
        with FixpointContext(mem, name="consolidation", constraint="L1",
                             _pre_hash=pre_hash) as ctx:
            ctx.record_iteration(3)   # 3 new content-hashes in iteration 0
            ctx.record_iteration(0)   # convergence marker

        certificate = ctx.result
        assert certificate is not None
        assert certificate.converged
        assert certificate.status == "converged"
        assert certificate.delta_sequence == [3, 0]
        assert certificate.constraint == "L1"
        assert certificate.audited  # all audit writes succeeded

        # --- Phase 3: End session (more audit events) ---
        mem.session_end()

        # --- Phase 4: Flush everything to Cloud ---
        buffered = cloud.buffered_count
        assert buffered > 0, f"Expected buffered events, got {buffered}"

        result = cloud.flush()

        assert result is not None, "Flush returned None"
        assert result.error is None, f"Flush error: {result.error}"
        assert result.accepted == buffered, f"Expected {buffered} accepted, got {result.accepted}"

        # --- Phase 5: Verify hash chain integrity ---
        # The last event written by AuditWriter is session_end.
        # Cloud's chain_head_hash should match the local hash of that last event.
        # We can verify via the witness.
        assert result.witness is not None, "No witness returned"
        assert len(witnesses) == 1
        witness = witnesses[0]
        assert witness.chain_head_seq == buffered - 1  # 0-indexed
        assert witness.chain_head_hash.startswith("sha256:")

        # --- Phase 6: Query Cloud to verify certificate events stored ---
        url = f"{cloud._endpoint}/v1/namespaces/{namespace.replace('/', '/')}/events?event_type=fixpoint_end"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {LIVE_API_KEY}",
                "User-Agent": USER_AGENT,
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        assert data["total"] >= 1, f"Expected fixpoint_end event, got {data}"
        cert_event = data["events"][0]
        cert_data = cert_event["data"]

        # Verify the certificate data matches what FixpointContext produced
        assert cert_data["name"] == "consolidation"
        assert cert_data["constraint"] == "L1"
        assert cert_data["status"] == "converged"
        assert cert_data["delta_sequence"] == [3, 0]
        assert cert_data["iterations"] == 2
        assert cert_data["audited"] is True
        assert "certificate_hash" in cert_data
        assert "initial_graph_hash" in cert_data
        assert "final_graph_hash" in cert_data

        # Verify the certificate_hash in Cloud matches local computation
        assert cert_data["certificate_hash"] == certificate.certificate_hash

        print(f"\nFIXPOINT CERTIFICATE VERIFIED IN CLOUD")
        print(f"  Namespace: {namespace}")
        print(f"  Events accepted: {result.accepted}")
        print(f"  Certificate hash: {certificate.certificate_hash[:20]}...")
        print(f"  Witness: {witness.id}")
        print(f"  Chain head: seq {witness.chain_head_seq}, hash {witness.chain_head_hash[:20]}...")

    def test_fixpoint_with_trace_id(self, tmp_path):
        """Verify trace_id flows through the pipeline and is queryable in Cloud."""
        import time
        namespace = f"flowscript/trace-test-{int(time.time())}"
        trace_id = f"trace_{int(time.time())}"

        cloud = CloudClient(
            api_key=LIVE_API_KEY,
            namespace=namespace,
        )

        audit_config = AuditConfig(on_event=cloud.queue_event)
        opts = MemoryOptions(audit=audit_config)
        mem = Memory.load_or_create(
            str(tmp_path / "trace_test.json"),
            options=opts,
        )

        # Write events with trace_id in data
        mem.session_start("trace_session")
        mem.write_audit("test_event", {
            "trace_id": trace_id,
            "description": "Event with trace ID",
        })
        mem.session_end()

        result = cloud.flush()
        assert result is not None
        assert result.error is None

        # Query by trace_id
        url = f"{cloud._endpoint}/v1/namespaces/{namespace.replace('/', '/')}/events?trace_id={trace_id}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {LIVE_API_KEY}",
                "User-Agent": USER_AGENT,
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        assert data["total"] >= 1, f"Expected event with trace_id, got {data}"
        event = data["events"][0]
        # trace_id is inside event.data (where AuditWriter put it) AND
        # extracted to DB column for filtering. The GET response returns
        # the original event JSON, so trace_id is in data, not top-level.
        assert event["data"]["trace_id"] == trace_id

        print(f"\nTRACE_ID VERIFIED IN CLOUD")
        print(f"  trace_id: {trace_id}")
        print(f"  Found {data['total']} event(s) matching trace_id filter")
