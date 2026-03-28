"""
AuditWriter — Hash-chained, append-only audit trail for FlowScript memory.

Every memory management decision — consolidation reasoning, node mutations, prune
events, session lifecycle — is captured in a tamper-evident, rotatable JSONL log.

Design:
- Append-only entries, rotatable files (entries never modified; files sealed + compressed)
- Write-first (crash-safe): audit entry written BEFORE the mutation it describes
- Hash-chained (tamper-evident): SHA256(previous_entry) → detectable tampering
- Monthly rotation with gzip compression + manifest index
- on_event callback for real-time integration (SIEM, Observatory)

File layout:
    {stem}.audit.jsonl                    # active (hot) — appending
    {stem}.audit.2026-02.jsonl.gz         # sealed (compressed)
    {stem}.audit.manifest.json            # index (time ranges, counts, hashes)
"""

from __future__ import annotations

import copy
import gzip
import atexit
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AuditConfig:
    """Configuration for the audit trail system.

    Attributes:
        rotation: When to rotate audit files. "monthly" (default), "weekly",
            "daily", "none", or "size:50MB" for size-based rotation.
        compression: Compression for rotated files. "gzip" (default) or "none".
        retention_months: How long to keep rotated files. Default 84 (7 years,
            SOX compliance). None = keep forever.
        hash_chain: Whether to hash-chain entries. Default True. Disable only
            for testing or when tamper-evidence is not needed.
        verbosity: "standard" (default) = mutation events only. "full" =
            mutations + read/query access events (for HIPAA access auditing).
        encryption: "none" (default) or "aes-256-gcm". Encryption at rest for
            audit trail files. NOT YET IMPLEMENTED — v2 commitment for SOC2/
            enterprise compliance. Setting to anything other than "none" raises
            NotImplementedError.
        on_event: Optional callback invoked for every audit entry. Receives the
            full entry dict AFTER disk write. Callback failure never blocks
            audit persistence. Use for SIEM integration, Observatory, or custom
            monitoring.
        on_event_async: If True, fire on_event in a background thread so slow
            webhooks or network I/O don't block agent operations. Events are
            dispatched in order (single worker thread). Callback errors are
            logged to stderr but never propagate. Default False (synchronous).
    """

    rotation: str = "monthly"
    compression: str = "gzip"
    retention_months: Optional[int] = 84
    hash_chain: bool = True
    verbosity: str = "standard"
    encryption: Literal["none", "aes-256-gcm"] = "none"
    on_event: Optional[Callable[[dict[str, Any]], None]] = None
    on_event_async: bool = False

    def __post_init__(self) -> None:
        if self.encryption != "none":
            raise NotImplementedError(
                f"Encryption at rest ('{self.encryption}') is not yet implemented. "
                "This is a documented v2 commitment. Currently only 'none' is supported."
            )


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class AuditQueryResult:
    """Result of query_audit()."""

    entries: list[dict[str, Any]]
    total_scanned: int
    files_searched: int
    chain_valid: Optional[bool] = None
    chain_break_at: Optional[int] = None


@dataclass
class AuditVerifyResult:
    """Result of verify_audit()."""

    valid: Optional[bool]  # True = chain intact, False = chain broken, None = no audit trail found
    total_entries: int
    files_verified: int
    legacy_entries: int = 0
    chain_break_at: Optional[int] = None
    chain_break_file: Optional[str] = None


# =============================================================================
# Sentinels
# =============================================================================

GENESIS_HASH = "sha256:GENESIS"
LEGACY_BRIDGE_HASH = "sha256:LEGACY_BRIDGE"
SCHEMA_VERSION = 1


# =============================================================================
# AuditWriter
# =============================================================================


class AuditWriter:
    """Hash-chained, append-only audit trail writer with rotation support.

    Usage:
        writer = AuditWriter(Path("agent_memory.json"))
        writer.write("node_create", {"node": {...}}, session_id="ses_123")

    The writer manages:
    - Entry format (v1 schema with hash chaining)
    - File rotation (monthly/weekly/daily/size-based)
    - Compression of rotated files (gzip)
    - Manifest index for efficient querying
    - on_event callback for real-time streaming
    """

    def __init__(
        self,
        memory_path: Path,
        config: Optional[AuditConfig] = None,
    ) -> None:
        self._memory_path = memory_path
        self._config = config or AuditConfig()
        self._active_path = self._derive_active_path(memory_path)
        self._manifest_path = self._derive_manifest_path(memory_path)
        self._seq = 0
        self._prev_hash = GENESIS_HASH
        self._current_period = ""
        self._initialized = False
        self._in_cleanup = False  # Guard against recursive cleanup → write → rotate → cleanup
        self._executor: Optional[ThreadPoolExecutor] = None

    def _get_executor(self) -> ThreadPoolExecutor:
        """Lazily create the background executor for async on_event dispatch."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="audit_on_event",
            )
            atexit.register(self.close)
        return self._executor

    def close(self) -> None:
        """Shutdown the async executor, waiting for all pending callbacks to complete.

        Call this to ensure in-flight on_event callbacks (e.g. webhook POSTs)
        have finished before process exit. Safe to call multiple times.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _fire_on_event(self, entry: dict[str, Any]) -> None:
        """Fire the on_event callback with error isolation."""
        try:
            self._config.on_event(entry)  # type: ignore[misc]
        except Exception as e:
            print(f"AuditWriter: on_event callback failed: {e}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Path derivation
    # -------------------------------------------------------------------------

    @staticmethod
    def _derive_active_path(memory_path: Path) -> Path:
        return memory_path.parent / (memory_path.stem + ".audit.jsonl")

    @staticmethod
    def _derive_manifest_path(memory_path: Path) -> Path:
        return memory_path.parent / (memory_path.stem + ".audit.manifest.json")

    @staticmethod
    def _period_for_time(dt: datetime, rotation: str) -> str:
        """Return period string for a datetime based on rotation setting."""
        if rotation == "daily":
            return dt.strftime("%Y-%m-%d")
        elif rotation == "weekly":
            # ISO week: YYYY-WNN
            return f"{dt.strftime('%Y')}-W{dt.strftime('%W')}"
        elif rotation == "monthly":
            return dt.strftime("%Y-%m")
        return ""  # "none" or size-based

    def _rotated_filename(self, period: str) -> str:
        stem = self._memory_path.stem
        suffix = ".jsonl.gz" if self._config.compression == "gzip" else ".jsonl"
        return f"{stem}.audit.{period}{suffix}"

    # -------------------------------------------------------------------------
    # Initialization (lazy — on first write)
    # -------------------------------------------------------------------------

    def _initialize(self) -> None:
        """Load state from existing active file and manifest. Lazy — called on first write."""
        if self._initialized:
            return

        now = datetime.now(timezone.utc)
        self._current_period = self._period_for_time(now, self._config.rotation)

        # If active file exists, recover seq and prev_hash from last entry
        if self._active_path.exists():
            last_line = self._read_last_line(self._active_path)
            if last_line:
                try:
                    last_entry = json.loads(last_line)
                    self._seq = last_entry.get("seq", 0) + 1
                    # Compute hash of last entry for chaining
                    if self._config.hash_chain:
                        self._prev_hash = self._compute_hash(last_line)
                except json.JSONDecodeError:
                    # Corrupt last line — start fresh chain from legacy bridge
                    self._seq = 0
                    self._prev_hash = LEGACY_BRIDGE_HASH
            else:
                # Empty file — fresh start
                self._seq = 0
                self._prev_hash = GENESIS_HASH

            # Check if we need to detect legacy entries (no prev_hash field)
            first_line = self._read_first_line(self._active_path)
            if first_line:
                try:
                    first_entry = json.loads(first_line)
                    if "prev_hash" not in first_entry:
                        # Legacy file — bridge from it
                        self._prev_hash = LEGACY_BRIDGE_HASH
                except json.JSONDecodeError:
                    pass
        else:
            # Fresh file
            self._seq = 0
            self._prev_hash = GENESIS_HASH

        # Load manifest if exists
        if self._manifest_path.exists():
            try:
                manifest = json.loads(self._manifest_path.read_text("utf-8"))
                # If active file is fresh and manifest has last hash, chain from it
                if not self._active_path.exists() and manifest.get("files"):
                    last_file = manifest["files"][-1]
                    self._prev_hash = last_file.get("last_hash", GENESIS_HASH)
            except (json.JSONDecodeError, KeyError):
                pass

        self._initialized = True

    @staticmethod
    def _read_last_line(path: Path) -> str:
        """Read the last non-empty line from a file efficiently."""
        try:
            with open(path, "rb") as f:
                # Seek to end
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    return ""
                # Read backwards to find last newline
                pos = size - 1
                while pos > 0:
                    f.seek(pos)
                    char = f.read(1)
                    if char == b"\n" and pos < size - 1:
                        return f.read().decode("utf-8").strip()
                    pos -= 1
                # If we reached start, whole file is one line
                f.seek(0)
                return f.read().decode("utf-8").strip()
        except OSError:
            return ""

    @staticmethod
    def _read_first_line(path: Path) -> str:
        """Read the first non-empty line from a file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        return line
        except OSError:
            pass
        return ""

    # -------------------------------------------------------------------------
    # Hash computation
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_hash(json_line: str) -> str:
        """Compute SHA256 hash of a JSON line (the exact bytes written to disk)."""
        return "sha256:" + hashlib.sha256(json_line.strip().encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------

    def write(
        self,
        event: str,
        data: dict[str, Any],
        session_id: Optional[str] = None,
        adapter: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Write an audit entry. Returns the entry dict.

        This method:
        1. Checks if rotation is needed (lazy — triggered by period change)
        2. Builds the entry with hash chain
        3. Writes to active file (append-only)
        4. Fires on_event callback (if configured)

        Args:
            event: Event type from taxonomy (e.g., "node_create", "consolidation")
            data: Event-specific payload
            session_id: Current session ID, if within a session
            adapter: Adapter attribution, if event originates from an adapter/MCP

        Returns:
            The complete entry dict (including v, seq, prev_hash, etc.)
        """
        self._initialize()

        # Check rotation
        now = datetime.now(timezone.utc)
        current_period = self._period_for_time(now, self._config.rotation)
        if (
            self._config.rotation != "none"
            and not self._config.rotation.startswith("size:")
            and self._current_period
            and current_period != self._current_period
            and self._active_path.exists()
            and self._active_path.stat().st_size > 0
        ):
            self._rotate(self._current_period)
            self._current_period = current_period

        # Size-based rotation check
        if self._config.rotation.startswith("size:"):
            max_bytes = self._parse_size(self._config.rotation)
            if (
                self._active_path.exists()
                and self._active_path.stat().st_size >= max_bytes
            ):
                period = now.strftime("%Y-%m-%dT%H%M%S")
                self._rotate(period)

        self._current_period = current_period

        # Build entry
        entry: dict[str, Any] = {
            "v": SCHEMA_VERSION,
            "seq": self._seq,
            "timestamp": now.isoformat(),
            "event": event,
        }
        if self._config.hash_chain:
            entry["prev_hash"] = self._prev_hash
        entry["session_id"] = session_id
        entry["data"] = data
        entry["adapter"] = adapter

        # Serialize deterministically (sorted keys for stable hashing)
        json_line = json.dumps(entry, sort_keys=True, separators=(",", ":"))

        # Write to file (append-only, fsync for crash safety)
        self._active_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._active_path, "a", encoding="utf-8") as f:
            f.write(json_line + "\n")
            f.flush()
            os.fsync(f.fileno())

        # Update state for next entry
        if self._config.hash_chain:
            self._prev_hash = self._compute_hash(json_line)
        self._seq += 1

        # Update manifest active state
        self._update_manifest_active(entry)

        # Fire on_event callback (failure must never block audit, but log to stderr)
        if self._config.on_event:
            if self._config.on_event_async:
                # Deep copy to prevent caller mutation between write() return and
                # async callback execution from producing different canonical JSON
                # (which would break hash chain verification in CloudClient).
                self._get_executor().submit(self._fire_on_event, copy.deepcopy(entry))
            else:
                self._fire_on_event(entry)

        return entry

    @staticmethod
    def _parse_size(size_str: str) -> int:
        """Parse size string like 'size:50MB' to bytes."""
        raw = size_str.replace("size:", "").strip().upper()
        multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
        for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
            if raw.endswith(suffix):
                return int(raw[: -len(suffix)]) * mult
        return int(raw)

    # -------------------------------------------------------------------------
    # Rotation
    # -------------------------------------------------------------------------

    def _rotate(self, period: str) -> None:
        """Seal the active file and start a new one.

        Process:
        1. Rename active → period-suffixed file
        2. Compress (if configured)
        3. Update manifest with sealed file metadata
        4. Reset seq counter
        5. Chain continues from last hash
        """
        if not self._active_path.exists() or self._active_path.stat().st_size == 0:
            return

        # Read metadata from active file before sealing
        first_line = self._read_first_line(self._active_path)
        last_line = self._read_last_line(self._active_path)
        entry_count = sum(1 for _ in open(self._active_path, "r", encoding="utf-8") if _.strip())

        first_ts = ""
        last_ts = ""
        first_hash = GENESIS_HASH
        last_hash = ""
        try:
            if first_line:
                fe = json.loads(first_line)
                first_ts = fe.get("timestamp", "")
                first_hash = fe.get("prev_hash", GENESIS_HASH)
            if last_line:
                last_hash = self._compute_hash(last_line)
                le = json.loads(last_line)
                last_ts = le.get("timestamp", "")
        except json.JSONDecodeError:
            pass

        # Determine rotated filename
        rotated_name = self._rotated_filename(period)
        rotated_path = self._active_path.parent / rotated_name

        if self._config.compression == "gzip":
            # Compress to .jsonl.gz
            uncompressed_size = self._active_path.stat().st_size
            with open(self._active_path, "rb") as f_in:
                with gzip.open(rotated_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            compressed_size = rotated_path.stat().st_size

            # Remove uncompressed active file
            self._active_path.unlink()
        else:
            # Just rename (no compression)
            plain_name = f"{self._memory_path.stem}.audit.{period}.jsonl"
            rotated_path = self._active_path.parent / plain_name
            rotated_name = plain_name
            self._active_path.rename(rotated_path)
            uncompressed_size = rotated_path.stat().st_size
            compressed_size = uncompressed_size

        # Compute file hash for integrity
        file_hash = self._hash_file(rotated_path)

        # Update manifest
        file_entry = {
            "filename": rotated_name,
            "period": period,
            "entries": entry_count,
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "first_hash": first_hash,
            "last_hash": last_hash,
            "size_bytes": compressed_size,
            "uncompressed_bytes": uncompressed_size,
            "sha256_file": file_hash,
        }
        self._add_to_manifest(file_entry)

        # Retention cleanup
        self._cleanup_retention()

        # Reset seq for new file, chain continues
        self._seq = 0
        # prev_hash stays as the last entry's hash (cross-file continuity)

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Compute SHA256 of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()

    # -------------------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------------------

    def _load_manifest(self) -> dict[str, Any]:
        """Load manifest or return default."""
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "version": 1,
            "memory_file": self._memory_path.name,
            "active_file": self._active_path.name,
            "active_last_hash": None,
            "active_last_seq": 0,
            "files": [],
            "retention_months": self._config.retention_months,
            "last_cleanup": None,
        }

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        """Save manifest to disk (atomic via temp+rename)."""
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._manifest_path.with_suffix(".tmp")
        content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        tmp_path.write_text(content, "utf-8")
        tmp_path.replace(self._manifest_path)  # atomic on POSIX

    def _add_to_manifest(self, file_entry: dict[str, Any]) -> None:
        """Add a sealed file entry to the manifest."""
        manifest = self._load_manifest()
        manifest["files"].append(file_entry)
        manifest["active_file"] = self._active_path.name
        self._save_manifest(manifest)

    def _update_manifest_active(self, entry: dict[str, Any]) -> None:
        """Update manifest's active file tracking (periodic, not every write).

        Only updates every 100 writes to avoid excessive I/O.
        """
        if self._seq % 100 == 0 or self._seq <= 1:
            manifest = self._load_manifest()
            manifest["active_last_seq"] = entry["seq"]
            # Store hash OF this entry (not entry's prev_hash) — this is the
            # latest hash in the chain, used for cross-file continuity
            manifest["active_last_hash"] = self._prev_hash
            manifest["active_file"] = self._active_path.name
            self._save_manifest(manifest)

    def _cleanup_retention(self) -> None:
        """Remove files older than retention period."""
        if self._config.retention_months is None:
            return
        if self._in_cleanup:
            return  # Prevent recursive cleanup → write → rotate → cleanup
        self._in_cleanup = True

        try:
            self._do_cleanup_retention()
        finally:
            self._in_cleanup = False

    def _do_cleanup_retention(self) -> None:
        """Internal cleanup implementation (separated to support recursion guard)."""
        now = datetime.now(timezone.utc)
        manifest = self._load_manifest()
        surviving = []
        deleted_files: list[str] = []

        for f in manifest.get("files", []):
            # Parse period to determine age
            period = f.get("period", "")
            try:
                # Try YYYY-MM format
                file_date = datetime.strptime(period, "%Y-%m").replace(tzinfo=timezone.utc)
            except ValueError:
                try:
                    # Try YYYY-MM-DD format
                    file_date = datetime.strptime(period, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except ValueError:
                    try:
                        # Try YYYY-WNN format (weekly rotation)
                        parts = period.split("-W")
                        if len(parts) == 2:
                            year = int(parts[0])
                            week = int(parts[1])
                            from datetime import timedelta
                            # Jan 1 of that year + N weeks
                            file_date = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(weeks=week)
                        else:
                            surviving.append(f)
                            continue
                    except (ValueError, IndexError):
                        surviving.append(f)
                        continue

            age_months = (now.year - file_date.year) * 12 + (now.month - file_date.month)
            if age_months > self._config.retention_months:
                # Delete the file
                file_path = self._active_path.parent / f["filename"]
                try:
                    file_path.unlink(missing_ok=True)
                    deleted_files.append(f["filename"])
                except OSError:
                    surviving.append(f)
                    continue
            else:
                surviving.append(f)

        manifest["files"] = surviving
        manifest["last_cleanup"] = now.isoformat()
        self._save_manifest(manifest)

        # Emit audit event for retention cleanup (if any files were deleted)
        if deleted_files:
            self.write("audit_cleanup", {
                "deleted_files": deleted_files,
                "retention_months": self._config.retention_months,
            })

    # -------------------------------------------------------------------------
    # Verification
    # -------------------------------------------------------------------------

    @staticmethod
    def verify(audit_path: str) -> AuditVerifyResult:
        """Verify hash chain integrity across all audit files.

        Args:
            audit_path: Path to active .audit.jsonl file or .audit.manifest.json

        Returns:
            AuditVerifyResult with chain integrity status
        """
        path = Path(audit_path)

        # Determine files to verify
        files_to_verify: list[Path] = []
        manifest_path: Optional[Path] = None

        if path.name.endswith(".manifest.json"):
            manifest_path = path
        else:
            # Try to find manifest alongside active file
            stem = path.name
            if stem.endswith(".audit.jsonl"):
                manifest_stem = stem.replace(".audit.jsonl", ".audit.manifest.json")
            else:
                manifest_stem = path.stem + ".audit.manifest.json"
            candidate = path.parent / manifest_stem
            if candidate.exists():
                manifest_path = candidate

        # Collect rotated files from manifest (in order)
        if manifest_path and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text("utf-8"))
                for f in manifest.get("files", []):
                    fp = path.parent / f["filename"]
                    if fp.exists():
                        files_to_verify.append(fp)
            except (json.JSONDecodeError, OSError):
                pass

        # Add active file
        active_path = path if path.name.endswith(".audit.jsonl") else None
        if active_path is None and manifest_path:
            # Derive active path from manifest
            try:
                manifest = json.loads(manifest_path.read_text("utf-8"))
                active_name = manifest.get("active_file", "")
                if active_name:
                    candidate = path.parent / active_name
                    if candidate.exists():
                        active_path = candidate
            except (json.JSONDecodeError, OSError):
                pass
        if active_path and active_path.exists():
            files_to_verify.append(active_path)

        if not files_to_verify:
            return AuditVerifyResult(valid=None, total_entries=0, files_verified=0)

        # Verify chain across all files
        total_entries = 0
        legacy_entries = 0
        prev_hash = GENESIS_HASH
        files_verified = 0

        for file_path in files_to_verify:
            files_verified += 1

            for line in _iter_audit_lines(file_path):
                line = line.strip()
                if not line:
                    continue
                total_entries += 1

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_prev_hash = entry.get("prev_hash")

                if entry_prev_hash is None:
                    # Legacy entry (pre-hash-chain)
                    legacy_entries += 1
                    # Update prev_hash to bridge into chained entries
                    prev_hash = LEGACY_BRIDGE_HASH
                    continue

                if entry_prev_hash == LEGACY_BRIDGE_HASH:
                    # Transition entry — accept and start chain from here
                    prev_hash = AuditWriter._compute_hash(line)
                    continue

                if entry_prev_hash != prev_hash:
                    return AuditVerifyResult(
                        valid=False,
                        total_entries=total_entries,
                        files_verified=files_verified,
                        legacy_entries=legacy_entries,
                        chain_break_at=entry.get("seq", total_entries - 1),
                        chain_break_file=file_path.name,
                    )

                prev_hash = AuditWriter._compute_hash(line)

        return AuditVerifyResult(
            valid=True,
            total_entries=total_entries,
            files_verified=files_verified,
            legacy_entries=legacy_entries,
        )

    # -------------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------------

    @staticmethod
    def query(
        audit_path: str,
        after: Optional[str] = None,
        before: Optional[str] = None,
        events: Optional[list[str]] = None,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        adapter: Optional[str] = None,
        limit: int = 100,
        verify_chain: bool = False,
    ) -> AuditQueryResult:
        """Query audit trail with filters.

        Args:
            audit_path: Path to active .audit.jsonl file or .audit.manifest.json
            after: Only entries after this ISO timestamp
            before: Only entries before this ISO timestamp
            events: Filter by event types (e.g., ["consolidation", "prune"])
            node_id: Filter by node involvement (searches data recursively)
            session_id: Filter by session ID
            adapter: Filter by adapter framework name
            limit: Maximum entries to return
            verify_chain: Also verify hash chain integrity

        Returns:
            AuditQueryResult with matching entries
        """
        path = Path(audit_path)

        # Collect all files to search (rotated + active)
        files_to_search: list[Path] = []
        manifest_path: Optional[Path] = None

        if path.name.endswith(".manifest.json"):
            manifest_path = path
        else:
            manifest_stem = path.name.replace(".audit.jsonl", ".audit.manifest.json")
            candidate = path.parent / manifest_stem
            if candidate.exists():
                manifest_path = candidate

        # Add rotated files from manifest (filter by time range if possible)
        if manifest_path and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text("utf-8"))
                for f in manifest.get("files", []):
                    # Time-range filter: skip files entirely outside query range
                    f_first = f.get("first_timestamp", "")
                    f_last = f.get("last_timestamp", "")
                    if after and f_last and f_last < after:
                        continue
                    if before and f_first and f_first > before:
                        continue
                    fp = path.parent / f["filename"]
                    if fp.exists():
                        files_to_search.append(fp)
            except (json.JSONDecodeError, OSError):
                pass

        # Add active file
        active_path = path if path.name.endswith(".audit.jsonl") else None
        if active_path is None and manifest_path:
            try:
                manifest = json.loads(manifest_path.read_text("utf-8"))
                active_name = manifest.get("active_file", "")
                if active_name:
                    candidate = path.parent / active_name
                    if candidate.exists():
                        active_path = candidate
            except (json.JSONDecodeError, OSError):
                pass
        if active_path and active_path.exists():
            files_to_search.append(active_path)

        # Search
        results: list[dict[str, Any]] = []
        total_scanned = 0

        for file_path in files_to_search:
            for line in _iter_audit_lines(file_path):
                line = line.strip()
                if not line:
                    continue
                total_scanned += 1

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Apply filters (parse timestamps for reliable comparison)
                entry_ts = entry.get("timestamp", "")
                if after and not _ts_after(entry_ts, after):
                    continue
                if before and not _ts_before(entry_ts, before):
                    continue
                if events and entry.get("event") not in events:
                    continue
                if session_id and entry.get("session_id") != session_id:
                    continue
                if adapter:
                    entry_adapter = entry.get("adapter")
                    if not entry_adapter or entry_adapter.get("framework") != adapter:
                        continue
                if node_id and not _entry_mentions_node(entry, node_id):
                    continue

                results.append(entry)
                if len(results) >= limit:
                    break

            if len(results) >= limit:
                break

        # Optional chain verification
        chain_valid: Optional[bool] = None
        chain_break_at: Optional[int] = None
        if verify_chain:
            vr = AuditWriter.verify(audit_path)
            chain_valid = vr.valid
            chain_break_at = vr.chain_break_at

        return AuditQueryResult(
            entries=results,
            total_scanned=total_scanned,
            files_searched=len(files_to_search),
            chain_valid=chain_valid,
            chain_break_at=chain_break_at,
        )


# =============================================================================
# Helpers
# =============================================================================


def _iter_audit_lines(path: Path):
    """Iterate lines from an audit file (streaming — handles .gz transparently).

    Yields lines one at a time. Never loads the entire file into memory.
    Safe for multi-year audit trails (SOX 7-year compliance).
    """
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            yield from f
    else:
        with open(path, "r", encoding="utf-8") as f:
            yield from f


def _entry_mentions_node(entry: dict[str, Any], node_id: str) -> bool:
    """Check if an audit entry mentions a specific node ID (recursive search)."""
    data = entry.get("data", {})
    return _dict_contains_value(data, node_id)


def _dict_contains_value(obj: Any, target: str) -> bool:
    """Recursively search for a string value in a nested dict/list structure."""
    if isinstance(obj, str):
        return obj == target
    if isinstance(obj, dict):
        return any(_dict_contains_value(v, target) for v in obj.values())
    if isinstance(obj, list):
        return any(_dict_contains_value(item, target) for item in obj)
    return False


def _parse_iso_ts(ts: str) -> datetime:
    """Parse an ISO 8601 timestamp, handling common variants (Z suffix, missing time)."""
    # Normalize Z → +00:00
    ts = ts.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        # Might be date-only (e.g., "2026-03-21")
        try:
            return datetime.strptime(ts, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            # Last resort: return epoch (matches everything)
            return datetime.min.replace(tzinfo=timezone.utc)


def _ts_after(entry_ts: str, after: str) -> bool:
    """Check if entry timestamp is after the filter timestamp."""
    return _parse_iso_ts(entry_ts) >= _parse_iso_ts(after)


def _ts_before(entry_ts: str, before: str) -> bool:
    """Check if entry timestamp is before the filter timestamp."""
    return _parse_iso_ts(entry_ts) <= _parse_iso_ts(before)
