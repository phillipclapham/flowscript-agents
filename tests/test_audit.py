"""Tests for AuditWriter — hash-chained, append-only audit trail."""

import gzip
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from flowscript_agents.audit import (
    GENESIS_HASH,
    LEGACY_BRIDGE_HASH,
    SCHEMA_VERSION,
    AuditConfig,
    AuditQueryResult,
    AuditVerifyResult,
    AuditWriter,
)
from flowscript_agents import Memory, MemoryOptions, AuditConfig as AuditConfigExport


# =============================================================================
# AuditWriter basics
# =============================================================================


class TestAuditWriterBasics:
    """Core write, format, and hash-chain functionality."""

    def test_write_creates_file(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path)
            writer.write("test_event", {"key": "value"})

            audit_path = Path(td) / "mem.audit.jsonl"
            assert audit_path.exists()

    def test_entry_has_v1_schema(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path)
            entry = writer.write("test_event", {"key": "value"})

            assert entry["v"] == SCHEMA_VERSION
            assert entry["seq"] == 0
            assert entry["event"] == "test_event"
            assert entry["data"] == {"key": "value"}
            assert "timestamp" in entry
            assert "prev_hash" in entry

    def test_sequential_seq_numbers(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path)

            e1 = writer.write("event_1", {})
            e2 = writer.write("event_2", {})
            e3 = writer.write("event_3", {})

            assert e1["seq"] == 0
            assert e2["seq"] == 1
            assert e3["seq"] == 2

    def test_first_entry_has_genesis_hash(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path)
            entry = writer.write("test", {})
            assert entry["prev_hash"] == GENESIS_HASH

    def test_session_id_passed_through(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path)
            entry = writer.write("test", {}, session_id="ses_abc123")
            assert entry["session_id"] == "ses_abc123"

    def test_adapter_passed_through(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path)
            adapter = {"framework": "crewai", "adapter_class": "CrewAIStorage", "operation": "save"}
            entry = writer.write("test", {}, adapter=adapter)
            assert entry["adapter"] == adapter

    def test_null_session_and_adapter_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path)
            entry = writer.write("test", {})
            assert entry["session_id"] is None
            assert entry["adapter"] is None

    def test_deterministic_json_serialization(self):
        """Sorted keys ensure same entry → same hash across runs."""
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path)
            writer.write("test", {"z_key": 1, "a_key": 2})

            audit_path = Path(td) / "mem.audit.jsonl"
            line = audit_path.read_text().strip()
            # Verify keys are sorted in the JSON
            assert '"a_key":2' in line
            assert line.index('"a_key"') < line.index('"z_key"')


# =============================================================================
# Hash chaining
# =============================================================================


class TestHashChaining:
    """SHA256 hash chain integrity."""

    def test_chain_links_entries(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path)

            e1 = writer.write("first", {})
            e2 = writer.write("second", {})

            # e2's prev_hash should be SHA256 of e1's JSON line
            audit_path = Path(td) / "mem.audit.jsonl"
            lines = audit_path.read_text().strip().split("\n")
            expected_hash = AuditWriter._compute_hash(lines[0])
            assert e2["prev_hash"] == expected_hash

    def test_verify_valid_chain(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path)
            for i in range(10):
                writer.write(f"event_{i}", {"i": i})

            audit_path = str(Path(td) / "mem.audit.jsonl")
            result = AuditWriter.verify(audit_path)
            assert result.valid is True
            assert result.total_entries == 10
            assert result.legacy_entries == 0

    def test_verify_detects_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path)
            for i in range(5):
                writer.write(f"event_{i}", {"i": i})

            # Tamper with entry 2
            audit_path = Path(td) / "mem.audit.jsonl"
            lines = audit_path.read_text().strip().split("\n")
            entry = json.loads(lines[2])
            entry["data"]["i"] = 999  # tamper
            lines[2] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
            audit_path.write_text("\n".join(lines) + "\n")

            result = AuditWriter.verify(str(audit_path))
            assert result.valid is False
            assert result.chain_break_at is not None

    def test_hash_chain_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            config = AuditConfig(hash_chain=False)
            writer = AuditWriter(mem_path, config=config)
            entry = writer.write("test", {})
            assert "prev_hash" not in entry

    def test_recovery_from_existing_file(self):
        """Writer recovers seq and prev_hash from existing audit file."""
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"

            # Write 5 entries with first writer
            writer1 = AuditWriter(mem_path)
            for i in range(5):
                writer1.write(f"batch1_{i}", {})

            # Create new writer (simulates new process) — should resume chain
            writer2 = AuditWriter(mem_path)
            e6 = writer2.write("batch2_0", {})
            assert e6["seq"] == 5  # continues from 5

            # Full chain should verify
            result = AuditWriter.verify(str(Path(td) / "mem.audit.jsonl"))
            assert result.valid is True
            assert result.total_entries == 6


# =============================================================================
# Legacy compatibility
# =============================================================================


class TestLegacyCompatibility:
    """Backwards compat with pre-hash-chain audit files."""

    def test_legacy_entries_reported_not_broken(self):
        """Legacy entries (no prev_hash) are 'unverifiable' not 'broken'."""
        with tempfile.TemporaryDirectory() as td:
            audit_path = Path(td) / "mem.audit.jsonl"
            # Write legacy-format entries (no v, seq, prev_hash)
            legacy_entries = [
                {"timestamp": "2026-03-01T00:00:00Z", "event": "prune", "nodes": []},
                {"timestamp": "2026-03-01T01:00:00Z", "event": "prune", "nodes": []},
            ]
            with open(audit_path, "w") as f:
                for entry in legacy_entries:
                    f.write(json.dumps(entry) + "\n")

            result = AuditWriter.verify(str(audit_path))
            assert result.valid is True  # NOT broken
            assert result.legacy_entries == 2

    def test_new_entries_bridge_from_legacy(self):
        """New entries after legacy ones use LEGACY_BRIDGE sentinel."""
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            audit_path = Path(td) / "mem.audit.jsonl"

            # Write legacy entry
            with open(audit_path, "w") as f:
                f.write(json.dumps({"timestamp": "2026-03-01T00:00:00Z", "event": "prune"}) + "\n")

            # New writer should detect legacy and bridge
            writer = AuditWriter(mem_path)
            entry = writer.write("new_event", {})
            assert entry["prev_hash"] == LEGACY_BRIDGE_HASH

            # Verify chain — should pass (legacy + bridge)
            result = AuditWriter.verify(str(audit_path))
            assert result.valid is True
            assert result.legacy_entries == 1


# =============================================================================
# Rotation
# =============================================================================


class TestRotation:
    """File rotation and compression."""

    def test_no_rotation_when_same_period(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path, AuditConfig(rotation="monthly"))
            writer.write("event_1", {})
            writer.write("event_2", {})

            # Should be a single active file, no .gz files
            gz_files = list(Path(td).glob("*.gz"))
            assert len(gz_files) == 0

    def test_rotation_none_never_rotates(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path, AuditConfig(rotation="none"))
            for i in range(100):
                writer.write(f"event_{i}", {})

            gz_files = list(Path(td).glob("*.gz"))
            assert len(gz_files) == 0

    def test_manual_rotation(self):
        """Force a rotation and verify compressed file created."""
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path, AuditConfig(compression="gzip"))
            writer._initialize()
            writer.write("before_rotation", {})

            # Force rotation
            writer._rotate("2026-02")

            gz_files = list(Path(td).glob("*.gz"))
            assert len(gz_files) == 1
            assert "2026-02" in gz_files[0].name

            # Verify compressed file is valid gzip with correct content
            with gzip.open(gz_files[0], "rt") as f:
                content = f.read()
            entries = [json.loads(line) for line in content.strip().split("\n")]
            assert len(entries) == 1
            assert entries[0]["event"] == "before_rotation"

    def test_rotation_updates_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path, AuditConfig(compression="gzip"))
            writer._initialize()
            writer.write("test", {})
            writer._rotate("2026-02")

            manifest_path = Path(td) / "mem.audit.manifest.json"
            assert manifest_path.exists()
            manifest = json.loads(manifest_path.read_text())
            assert len(manifest["files"]) == 1
            assert manifest["files"][0]["period"] == "2026-02"
            assert manifest["files"][0]["entries"] == 1

    def test_rotation_no_compression(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path, AuditConfig(compression="none"))
            writer._initialize()
            writer.write("test", {})
            writer._rotate("2026-02")

            # Should have a plain .jsonl file, not .gz
            jsonl_files = [f for f in Path(td).glob("*.2026-02.jsonl") if not f.name.endswith(".gz")]
            assert len(jsonl_files) == 1

    def test_cross_file_chain_continuity(self):
        """Hash chain should be unbroken across rotation boundary."""
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path, AuditConfig(compression="gzip"))
            writer._initialize()

            # Write entries, then rotate, then write more
            writer.write("before_1", {})
            writer.write("before_2", {})
            writer._rotate("2026-02")
            writer.write("after_1", {})
            writer.write("after_2", {})

            # Verify full chain across files
            audit_path = str(Path(td) / "mem.audit.jsonl")
            result = AuditWriter.verify(audit_path)
            assert result.valid is True
            assert result.total_entries == 4
            assert result.files_verified == 2  # compressed + active

    def test_seq_resets_after_rotation(self):
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path, AuditConfig(compression="gzip"))
            writer._initialize()

            writer.write("before", {})
            writer._rotate("2026-02")
            entry = writer.write("after", {})
            assert entry["seq"] == 0  # reset after rotation

    def test_chain_integrity_across_writer_restart_after_rotation(self):
        """New writer after rotation chains correctly from manifest."""
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"

            # Writer 1: write entries, rotate, then die
            writer1 = AuditWriter(mem_path, AuditConfig(compression="gzip"))
            writer1._initialize()
            writer1.write("w1_event_1", {})
            writer1.write("w1_event_2", {})
            writer1._rotate("2026-02")
            # Active file is now empty (rotation deleted it)
            del writer1

            # Writer 2: starts fresh, should chain from manifest
            writer2 = AuditWriter(mem_path, AuditConfig(compression="gzip"))
            writer2.write("w2_event_1", {})
            writer2.write("w2_event_2", {})

            # Verify full chain across both files
            audit_path = str(Path(td) / "mem.audit.jsonl")
            result = AuditWriter.verify(audit_path)
            assert result.valid is True
            assert result.total_entries == 4
            assert result.files_verified == 2

    def test_size_based_rotation(self):
        """Size-based rotation triggers when file exceeds threshold."""
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            # Set tiny size limit to trigger rotation quickly
            writer = AuditWriter(mem_path, AuditConfig(rotation="size:500B", compression="gzip"))

            # Write entries until we exceed 500 bytes
            for i in range(20):
                writer.write(f"event_{i}", {"data": "x" * 50})

            # Should have at least one rotated file
            gz_files = list(Path(td).glob("*.gz"))
            assert len(gz_files) >= 1

            # Active file should still exist and be small
            active = Path(td) / "mem.audit.jsonl"
            assert active.exists()

    def test_retention_cleanup_deletes_old_files(self):
        """Retention cleanup removes files older than retention period."""
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            # 1 month retention
            writer = AuditWriter(mem_path, AuditConfig(retention_months=1, compression="gzip"))
            writer._initialize()

            # Create a "very old" rotated file manually
            old_file = Path(td) / "mem.audit.2020-01.jsonl.gz"
            with gzip.open(old_file, "wt") as f:
                f.write(json.dumps({"event": "old", "timestamp": "2020-01-15T00:00:00Z"}) + "\n")

            # Add it to manifest
            manifest = writer._load_manifest()
            manifest["files"].append({
                "filename": "mem.audit.2020-01.jsonl.gz",
                "period": "2020-01",
                "entries": 1,
                "first_timestamp": "2020-01-15T00:00:00Z",
                "last_timestamp": "2020-01-15T00:00:00Z",
                "first_hash": GENESIS_HASH,
                "last_hash": "sha256:abc",
                "size_bytes": 100,
                "uncompressed_bytes": 200,
                "sha256_file": "sha256:def",
            })
            writer._save_manifest(manifest)

            # Write something to trigger audit (retention runs on rotation, but
            # let's call it directly)
            writer.write("current_event", {})
            writer._cleanup_retention()

            # Old file should be deleted
            assert not old_file.exists()

            # Manifest should no longer reference it
            manifest = json.loads((Path(td) / "mem.audit.manifest.json").read_text())
            old_entries = [f for f in manifest["files"] if f["period"] == "2020-01"]
            assert len(old_entries) == 0

    def test_retention_cleanup_emits_audit_event(self):
        """Retention cleanup writes an audit_cleanup event."""
        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path, AuditConfig(retention_months=1, compression="gzip"))
            writer._initialize()

            # Create old rotated file
            old_file = Path(td) / "mem.audit.2020-01.jsonl.gz"
            with gzip.open(old_file, "wt") as f:
                f.write(json.dumps({"event": "old"}) + "\n")

            manifest = writer._load_manifest()
            manifest["files"].append({
                "filename": "mem.audit.2020-01.jsonl.gz",
                "period": "2020-01",
                "entries": 1,
                "first_timestamp": "2020-01-01T00:00:00Z",
                "last_timestamp": "2020-01-31T00:00:00Z",
                "first_hash": GENESIS_HASH,
                "last_hash": "sha256:abc",
                "size_bytes": 100,
                "uncompressed_bytes": 200,
                "sha256_file": "sha256:def",
            })
            writer._save_manifest(manifest)

            writer._cleanup_retention()

            # Check that audit_cleanup event was written
            audit_path = Path(td) / "mem.audit.jsonl"
            if audit_path.exists():
                entries = [json.loads(line) for line in audit_path.read_text().strip().split("\n") if line.strip()]
                cleanup_entries = [e for e in entries if e["event"] == "audit_cleanup"]
                assert len(cleanup_entries) == 1
                assert "mem.audit.2020-01.jsonl.gz" in cleanup_entries[0]["data"]["deleted_files"]


# =============================================================================
# Query
# =============================================================================


class TestQuery:
    """Audit trail querying."""

    def _write_test_entries(self, td):
        mem_path = Path(td) / "mem.json"
        writer = AuditWriter(mem_path)
        writer.write("node_create", {"node_id": "abc"}, session_id="ses_1")
        writer.write("consolidation", {"action": "RELATE", "target_node_id": "def"}, session_id="ses_1",
                      adapter={"framework": "crewai", "adapter_class": "CrewAI", "operation": "save"})
        writer.write("prune", {"nodes": [{"id": "ghi"}]}, session_id="ses_2")
        writer.write("node_create", {"node_id": "jkl"}, session_id="ses_2",
                      adapter={"framework": "langgraph", "adapter_class": "LG", "operation": "put"})
        return str(Path(td) / "mem.audit.jsonl")

    def test_query_all(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_test_entries(td)
            result = AuditWriter.query(path)
            assert len(result.entries) == 4
            assert result.total_scanned == 4

    def test_query_by_event(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_test_entries(td)
            result = AuditWriter.query(path, events=["node_create"])
            assert len(result.entries) == 2
            assert all(e["event"] == "node_create" for e in result.entries)

    def test_query_by_session_id(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_test_entries(td)
            result = AuditWriter.query(path, session_id="ses_1")
            assert len(result.entries) == 2

    def test_query_by_adapter(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_test_entries(td)
            result = AuditWriter.query(path, adapter="crewai")
            assert len(result.entries) == 1
            assert result.entries[0]["event"] == "consolidation"

    def test_query_by_node_id(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_test_entries(td)
            result = AuditWriter.query(path, node_id="abc")
            assert len(result.entries) == 1
            assert result.entries[0]["data"]["node_id"] == "abc"

    def test_query_with_limit(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_test_entries(td)
            result = AuditWriter.query(path, limit=2)
            assert len(result.entries) == 2

    def test_query_with_verify(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_test_entries(td)
            result = AuditWriter.query(path, verify_chain=True)
            assert result.chain_valid is True

    def test_query_empty_file(self):
        with tempfile.TemporaryDirectory() as td:
            audit_path = Path(td) / "mem.audit.jsonl"
            audit_path.touch()
            result = AuditWriter.query(str(audit_path))
            assert len(result.entries) == 0


# =============================================================================
# on_event callback
# =============================================================================


class TestOnEventCallback:
    """Real-time event streaming via callback."""

    def test_callback_fires_for_every_entry(self):
        captured = []
        config = AuditConfig(on_event=lambda e: captured.append(e))

        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path, config)
            writer.write("event_1", {"a": 1})
            writer.write("event_2", {"b": 2})

        assert len(captured) == 2
        assert captured[0]["event"] == "event_1"
        assert captured[1]["event"] == "event_2"

    def test_callback_failure_does_not_block_write(self):
        def bad_callback(entry):
            raise RuntimeError("callback exploded")

        config = AuditConfig(on_event=bad_callback)

        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path, config)
            # Should NOT raise
            entry = writer.write("test", {})
            assert entry is not None

            # File should still be written
            audit_path = Path(td) / "mem.audit.jsonl"
            assert audit_path.exists()
            lines = audit_path.read_text().strip().split("\n")
            assert len(lines) == 1


# =============================================================================
# on_event_async callback
# =============================================================================


class TestOnEventAsync:
    """Async on_event dispatch — webhook latency must not block agent operations."""

    def test_async_callback_fires_for_every_entry(self):
        """All events arrive eventually when on_event_async=True."""
        import threading

        captured = []
        lock = threading.Lock()

        def callback(entry):
            with lock:
                captured.append(entry)

        config = AuditConfig(on_event=callback, on_event_async=True)

        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path, config)
            writer.write("event_1", {"a": 1})
            writer.write("event_2", {"b": 2})
            writer.write("event_3", {"c": 3})
            # close() flushes executor — waits for all callbacks to complete
            writer.close()

        assert len(captured) == 3
        assert [e["event"] for e in captured] == ["event_1", "event_2", "event_3"]

    def test_async_callback_preserves_event_ordering(self):
        """Serial executor (max_workers=1) delivers events in write order."""
        import threading

        order = []
        lock = threading.Lock()

        def callback(entry):
            with lock:
                order.append(entry["seq"])

        config = AuditConfig(on_event=callback, on_event_async=True)

        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path, config)
            for i in range(10):
                writer.write(f"event_{i}", {})
            writer.close()

        assert order == list(range(10))

    def test_async_callback_failure_does_not_propagate(self):
        """Bad async callback logs to stderr but never surfaces as exception."""
        import io

        def exploding_callback(entry):
            raise ValueError("webhook down")

        config = AuditConfig(on_event=exploding_callback, on_event_async=True)

        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path, config)
            entry = writer.write("test", {})
            writer.close()  # flush — error happens here, not during write()

        # Entry was returned synchronously (disk write succeeded before async dispatch)
        assert entry is not None
        assert entry["event"] == "test"

    def test_async_write_returns_immediately(self):
        """write() returns before the callback completes when on_event_async=True."""
        import threading
        import time

        barrier = threading.Event()
        completed = threading.Event()

        def slow_callback(entry):
            barrier.wait(timeout=5)  # Block until test releases
            completed.set()

        config = AuditConfig(on_event=slow_callback, on_event_async=True)

        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path, config)

            # write() should return immediately (callback is blocked)
            entry = writer.write("test", {})
            assert entry is not None
            assert not completed.is_set()  # Callback not done yet

            # Unblock callback and wait for completion
            barrier.set()
            writer.close()

        assert completed.is_set()

    def test_close_is_idempotent(self):
        """close() can be called multiple times without error."""
        config = AuditConfig(on_event=lambda e: None, on_event_async=True)

        with tempfile.TemporaryDirectory() as td:
            mem_path = Path(td) / "mem.json"
            writer = AuditWriter(mem_path, config)
            writer.write("test", {})
            writer.close()
            writer.close()  # Should not raise

    def test_sync_is_default(self):
        """on_event_async defaults to False — behavior unchanged for existing callers."""
        config = AuditConfig()
        assert config.on_event_async is False

    def test_sync_and_async_produce_same_entries(self):
        """Sync and async modes deliver identical entry dicts."""
        sync_captured = []
        async_captured = []

        with tempfile.TemporaryDirectory() as td:
            # Sync writer
            mem_path_sync = Path(td) / "sync.json"
            sync_writer = AuditWriter(mem_path_sync, AuditConfig(on_event=sync_captured.append))
            sync_writer.write("test_event", {"x": 42})

            # Async writer
            mem_path_async = Path(td) / "async.json"
            async_writer = AuditWriter(
                mem_path_async,
                AuditConfig(on_event=async_captured.append, on_event_async=True),
            )
            async_writer.write("test_event", {"x": 42})
            async_writer.close()

        assert len(sync_captured) == 1
        assert len(async_captured) == 1
        # Same fields (seq/prev_hash differ because different files, but event/data match)
        assert sync_captured[0]["event"] == async_captured[0]["event"]
        assert sync_captured[0]["data"] == async_captured[0]["data"]


# =============================================================================
# Memory integration
# =============================================================================


class TestMemoryAuditIntegration:
    """AuditWriter integration with Memory class."""

    def test_memory_creates_audit_writer_when_file_path_set(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            mem = Memory.load_or_create(path)
            mem.thought("test")

            audit_path = Path(td) / "mem.audit.jsonl"
            assert audit_path.exists()

    def test_node_create_events(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            mem = Memory.load_or_create(path)
            t = mem.thought("important insight")

            entries = Memory.read_audit_log(str(Path(td) / "mem.audit.jsonl"))
            creates = [e for e in entries if e["event"] == "node_create"]
            assert len(creates) == 1
            assert creates[0]["data"]["content"] == "important insight"
            assert creates[0]["data"]["node_type"] == "thought"
            assert creates[0]["data"]["source"] == "api"

    def test_relationship_create_events(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            mem = Memory.load_or_create(path)
            a = mem.thought("idea A")
            b = mem.thought("idea B")
            mem.tension(a, b, axis="approach")

            entries = Memory.read_audit_log(str(Path(td) / "mem.audit.jsonl"))
            rels = [e for e in entries if e["event"] == "relationship_create"]
            assert len(rels) == 1
            assert rels[0]["data"]["type"] == "tension"
            assert rels[0]["data"]["axis_label"] == "approach"

    def test_state_change_events(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            mem = Memory.load_or_create(path)
            t = mem.thought("something")
            t.block(reason="waiting on API keys")

            entries = Memory.read_audit_log(str(Path(td) / "mem.audit.jsonl"))
            states = [e for e in entries if e["event"] == "state_change"]
            assert len(states) == 1
            assert states[0]["data"]["state_type"] == "blocked"

    def test_graduation_events(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            mem = Memory.load_or_create(path, options=MemoryOptions(
                temporal=None  # defaults: graduation at freq 2
            ))
            t = mem.thought("test node")
            # Touch enough to trigger graduation (current → developing at freq 2)
            mem._touch_node(t.id, session_scoped=False)  # freq becomes 2 → graduates

            entries = Memory.read_audit_log(str(Path(td) / "mem.audit.jsonl"))
            grads = [e for e in entries if e["event"] == "graduation"]
            assert len(grads) == 1
            assert grads[0]["data"]["old_tier"] == "current"
            assert grads[0]["data"]["new_tier"] == "developing"

    def test_session_lifecycle_events(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            mem = Memory.load_or_create(path)
            mem.thought("test")
            mem.session_start()
            mem.session_end()

            entries = Memory.read_audit_log(str(Path(td) / "mem.audit.jsonl"))
            events = [e["event"] for e in entries]
            assert "session_start" in events
            assert "session_end" in events

    def test_session_id_correlated(self):
        """All events within a session share the same session_id."""
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            mem = Memory.load_or_create(path)
            mem.session_start()
            mem.thought("during session")
            mem.session_end()

            entries = Memory.read_audit_log(str(Path(td) / "mem.audit.jsonl"))
            # session_start, node_create, and session_end should share session_id
            session_entries = [e for e in entries if e.get("session_id") is not None]
            session_ids = set(e["session_id"] for e in session_entries)
            assert len(session_ids) == 1  # all same session

    def test_prune_uses_audit_writer(self):
        """Prune events go through AuditWriter (hash-chained)."""
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            mem = Memory.load_or_create(path, options=MemoryOptions(
                temporal=__import__("flowscript_agents.memory", fromlist=["TemporalConfig"]).TemporalConfig(
                    dormancy=__import__("flowscript_agents.memory", fromlist=["DormancyConfig"]).DormancyConfig(
                        resting="1ms", dormant="2ms"
                    )
                )
            ))
            t = mem.thought("will be pruned")
            meta = mem.get_temporal(t.id)
            meta.last_touched = "2020-01-01T00:00:00+00:00"
            time.sleep(0.01)
            mem.prune()

            entries = Memory.read_audit_log(str(Path(td) / "mem.audit.jsonl"))
            prune_entries = [e for e in entries if e["event"] == "prune"]
            assert len(prune_entries) == 1
            # Should have v1 schema fields (hash-chained)
            assert prune_entries[0]["v"] == 1
            assert "prev_hash" in prune_entries[0]

    def test_query_audit_static_method(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            mem = Memory.load_or_create(path)
            mem.thought("a")
            mem.thought("b")

            audit_path = str(Path(td) / "mem.audit.jsonl")
            result = Memory.query_audit(audit_path, events=["node_create"])
            assert len(result.entries) == 2

    def test_verify_audit_static_method(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            mem = Memory.load_or_create(path)
            mem.thought("a")
            mem.thought("b")
            mem.thought("c")

            audit_path = str(Path(td) / "mem.audit.jsonl")
            result = Memory.verify_audit(audit_path)
            assert result.valid is True
            assert result.total_entries == 3

    def test_no_audit_without_file_path(self):
        """In-memory mode = no audit trail (no crash, no file)."""
        mem = Memory()
        mem.thought("test")  # should not raise
        mem.session_start()
        mem.session_end()
        # No file_path → no audit writer → no writes → no errors

    def test_adapter_context_flows_to_audit(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            mem = Memory.load_or_create(path)
            mem.set_adapter_context("crewai", "CrewAIStorage", "save")
            mem.thought("adapter test")
            mem.clear_adapter_context()

            entries = Memory.read_audit_log(str(Path(td) / "mem.audit.jsonl"))
            creates = [e for e in entries if e["event"] == "node_create"]
            assert creates[0]["adapter"]["framework"] == "crewai"

    def test_audit_config_on_memory_options(self):
        """AuditConfig can be passed via MemoryOptions."""
        captured = []
        config = AuditConfig(on_event=lambda e: captured.append(e))
        opts = MemoryOptions(audit=config)

        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            mem = Memory.load_or_create(path, options=opts)
            mem.thought("callback test")

        assert len(captured) == 1
        assert captured[0]["event"] == "node_create"

    def test_remove_node_audit_coverage(self):
        """remove_node() writes audit trail BEFORE removal."""
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            mem = Memory.load_or_create(path)
            t = mem.thought("will be removed")
            t.block(reason="test block")
            node_id = t.id

            mem.remove_node(node_id)

            entries = Memory.read_audit_log(str(Path(td) / "mem.audit.jsonl"))
            removes = [e for e in entries if e["event"] == "node_remove"]
            assert len(removes) == 1
            assert removes[0]["data"]["node"]["content"] == "will be removed"
            assert len(removes[0]["data"]["states"]) >= 1  # blocked state captured

    def test_remove_node_captures_relationships(self):
        """remove_node() captures associated relationships in audit."""
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "mem.json")
            mem = Memory.load_or_create(path)
            a = mem.thought("idea A")
            b = mem.thought("idea B")
            mem.tension(a, b, axis="approach")

            mem.remove_node(a.id)

            entries = Memory.read_audit_log(str(Path(td) / "mem.audit.jsonl"))
            removes = [e for e in entries if e["event"] == "node_remove"]
            assert len(removes) == 1
            assert len(removes[0]["data"]["relationships"]) == 1  # tension captured
