"""
CloudClient — Batching client for FlowScript Cloud API.

Sends audit trail events to api.flowscript.org for independent cryptographic
witnessing. Events are buffered locally and flushed in batches.

Integration with AuditWriter via on_event callback:

    from flowscript_agents import AuditConfig
    from flowscript_agents.cloud import CloudClient

    cloud = CloudClient(api_key="fsk_...", namespace="myorg/myagent")
    config = AuditConfig(on_event=cloud.queue_event, on_event_async=True)
    writer = AuditWriter(Path("agent.json"), config)
    # Events are automatically queued and flushed to Cloud

    # At shutdown:
    cloud.flush()  # Send any remaining buffered events

Manual usage:

    cloud = CloudClient(api_key="fsk_...", namespace="myorg/myagent")
    cloud.queue_event(entry_dict)
    result = cloud.flush()
    print(result.accepted, result.witness)

Configuration via environment:

    FLOWSCRIPT_API_KEY=fsk_...
    FLOWSCRIPT_CLOUD_URL=https://api.flowscript.org  (default)
    FLOWSCRIPT_NAMESPACE=myorg/myagent
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CLOUD_URL = "https://api.flowscript.org"
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_BUFFER_SIZE = 10000
DEFAULT_FLUSH_INTERVAL = 5.0  # seconds (for future auto-flush)
def _get_user_agent() -> str:
    try:
        from flowscript_agents import __version__
        return f"flowscript-agents-python/{__version__}"
    except ImportError:
        return "flowscript-agents-python/unknown"

USER_AGENT = _get_user_agent()


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class CloudFlushResult:
    """Result of a flush operation."""

    accepted: int
    witness: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


@dataclass
class CloudWitness:
    """Witness attestation from Cloud."""

    id: str
    chain_head_seq: int
    chain_head_hash: str
    witnessed_at: str


# =============================================================================
# CloudClient
# =============================================================================


class CloudClient:
    """Batching client for FlowScript Cloud event ingestion.

    Events are buffered locally and sent in batches to minimize network
    round-trips. Each flush returns a witness attestation from the Cloud
    service, proving independent verification of the hash chain.

    Thread-safe: multiple threads can call queue_event() concurrently.

    Args:
        api_key: FlowScript Cloud API key (fsk_...). Falls back to
            FLOWSCRIPT_API_KEY env var.
        namespace: Agent namespace in "owner/agent" format. Falls back to
            FLOWSCRIPT_NAMESPACE env var.
        endpoint: Cloud API base URL. Falls back to FLOWSCRIPT_CLOUD_URL
            env var, then https://api.flowscript.org.
        batch_size: Flush automatically when buffer reaches this size.
            Default 100.
        max_buffer_size: Maximum events to buffer before dropping new events.
            Prevents unbounded memory growth when Cloud is unreachable.
            Dropped events are still in the local audit trail (AuditWriter
            writes to disk before calling on_event). Use the backfill
            endpoint to recover after Cloud connectivity is restored.
            Default 10000.
        timeout: HTTP request timeout in seconds. Default 30.
        on_witness: Optional callback invoked with each CloudWitness after
            successful flush. Use for logging or monitoring.
        on_error: Optional callback invoked on flush errors. Receives the
            CloudFlushResult with error details. Default: print to stderr.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        namespace: Optional[str] = None,
        endpoint: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_buffer_size: int = DEFAULT_MAX_BUFFER_SIZE,
        timeout: int = 30,
        on_witness: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("FLOWSCRIPT_API_KEY", "")
        self._namespace = namespace or os.environ.get("FLOWSCRIPT_NAMESPACE", "")
        self._endpoint = (
            endpoint or os.environ.get("FLOWSCRIPT_CLOUD_URL", DEFAULT_CLOUD_URL)
        ).rstrip("/")
        self._batch_size = batch_size
        self._max_buffer_size = max_buffer_size
        self._timeout = timeout
        self._on_witness = on_witness
        self._on_error = on_error
        self._buffer: list[str] = []
        self._lock = threading.Lock()
        self._total_sent = 0
        self._total_accepted = 0
        self._total_dropped = 0
        self._last_witness: Optional[CloudWitness] = None

        if not self._api_key:
            raise ValueError(
                "API key required. Pass api_key= or set FLOWSCRIPT_API_KEY env var."
            )
        if not self._namespace:
            raise ValueError(
                "Namespace required. Pass namespace= or set FLOWSCRIPT_NAMESPACE env var. "
                "Format: owner/agent"
            )

    @property
    def buffered_count(self) -> int:
        """Number of events currently buffered."""
        with self._lock:
            return len(self._buffer)

    @property
    def total_sent(self) -> int:
        """Total events sent to Cloud (may include retransmissions)."""
        return self._total_sent

    @property
    def total_accepted(self) -> int:
        """Total events accepted by Cloud (excludes replays)."""
        return self._total_accepted

    @property
    def total_dropped(self) -> int:
        """Total events dropped due to buffer overflow.

        Dropped events are NOT lost — they exist in the local audit trail
        (AuditWriter writes to disk before calling on_event). Use the
        backfill endpoint to sync them to Cloud after connectivity is restored.
        """
        return self._total_dropped

    @property
    def last_witness(self) -> Optional[CloudWitness]:
        """Most recent witness attestation received."""
        return self._last_witness

    # -------------------------------------------------------------------------
    # Queue + Flush
    # -------------------------------------------------------------------------

    def queue_event(self, entry: dict[str, Any]) -> None:
        """Queue an audit event for batch upload to Cloud.

        This is designed to be used as the AuditConfig.on_event callback.
        The entry dict is re-serialized to canonical JSON (matching the
        AuditWriter's serialization) for hash-chain-compatible transmission.

        If the buffer has reached max_buffer_size, the event is dropped with
        a warning to stderr. Dropped events are NOT lost — they exist in the
        local audit trail. Use the backfill endpoint to recover.

        Args:
            entry: Audit event dict as produced by AuditWriter.write()
        """
        # Re-serialize to canonical JSON — identical to AuditWriter line 390.
        # Python json.dumps with sort_keys=True is deterministic for the same dict.
        json_line = json.dumps(entry, sort_keys=True, separators=(",", ":"))

        batch_to_send = None
        with self._lock:
            if len(self._buffer) >= self._max_buffer_size:
                self._total_dropped += 1
                if self._total_dropped == 1 or self._total_dropped % 1000 == 0:
                    print(
                        f"CloudClient: buffer full ({self._max_buffer_size} events). "
                        f"Dropping Cloud sync ({self._total_dropped} dropped total). "
                        f"Local audit trail is unaffected. Use backfill to recover.",
                        file=sys.stderr,
                    )
                return

            self._buffer.append(json_line)
            if len(self._buffer) >= self._batch_size:
                batch_to_send = list(self._buffer)
                self._buffer.clear()

        # Auto-flush outside the lock — same pattern as flush() to avoid
        # blocking other queue_event() callers during HTTP I/O.
        if batch_to_send is not None:
            self._send_batch(batch_to_send)

    def flush(self) -> Optional[CloudFlushResult]:
        """Send all buffered events to Cloud.

        Returns None if buffer is empty. Otherwise returns CloudFlushResult
        with accepted count and witness attestation.

        Safe to call from any thread, including atexit handlers.
        """
        # Take events from buffer under lock, then release lock for HTTP I/O.
        # This allows other threads to queue_event() while we're sending.
        with self._lock:
            if not self._buffer:
                return None
            events = list(self._buffer)
            self._buffer.clear()

        return self._send_batch(events)

    def _send_batch(self, events: list[str]) -> CloudFlushResult:
        """Send a batch of canonical JSON event strings to Cloud.

        Handles response parsing, witness tracking, error handling, and retry
        (putting events back in buffer on 5xx/network errors).
        """
        body = json.dumps({
            "namespace": self._namespace,
            "events": events,
        }).encode("utf-8")

        url = f"{self._endpoint}/v1/events"
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                resp_body = json.loads(resp.read().decode("utf-8"))
                self._total_sent += len(events)
                accepted = resp_body.get("accepted", 0)
                self._total_accepted += accepted

                # Parse witness — defensive against malformed responses.
                # Events are already accepted; witness parse failure should not
                # cause data loss or propagate errors.
                witness_data = resp_body.get("witness")
                witness = None
                if witness_data:
                    try:
                        witness = CloudWitness(
                            id=witness_data["id"],
                            chain_head_seq=witness_data["chain_head_seq"],
                            chain_head_hash=witness_data["chain_head_hash"],
                            witnessed_at=witness_data["witnessed_at"],
                        )
                        self._last_witness = witness
                    except (KeyError, TypeError) as e:
                        print(
                            f"CloudClient: malformed witness response: {e}",
                            file=sys.stderr,
                        )

                    if witness and self._on_witness:
                        try:
                            self._on_witness(witness)
                        except Exception as e:
                            print(
                                f"CloudClient: on_witness callback failed: {e}",
                                file=sys.stderr,
                            )

                return CloudFlushResult(
                    accepted=accepted,
                    witness=witness_data,
                    status_code=resp.status,
                )

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass

            result = CloudFlushResult(
                accepted=0,
                error=f"HTTP {e.code}: {error_body}",
                status_code=e.code,
            )

            # On 409 (chain break), don't retry — this is a data integrity issue
            # that needs human attention. On 4xx, also don't retry (client error).
            # On 5xx, put events back in buffer for retry.
            if e.code >= 500:
                with self._lock:
                    self._buffer = events + self._buffer

            self._handle_error(result)
            return result

        except (urllib.error.URLError, OSError, TimeoutError) as e:
            # Network error — put events back for retry
            with self._lock:
                self._buffer = events + self._buffer
            result = CloudFlushResult(
                accepted=0,
                error=f"Network error: {e}",
            )
            self._handle_error(result)
            return result

    def _handle_error(self, result: CloudFlushResult) -> None:
        """Handle flush errors."""
        if self._on_error:
            try:
                self._on_error(result)
            except Exception:
                pass
        else:
            print(
                f"CloudClient: flush failed: {result.error}",
                file=sys.stderr,
            )

    # -------------------------------------------------------------------------
    # Convenience
    # -------------------------------------------------------------------------

    def send_events(self, entries: list[dict[str, Any]]) -> CloudFlushResult:
        """Send a list of events immediately (no buffering).

        Useful for backfill or one-shot uploads.
        """
        json_lines = [
            json.dumps(entry, sort_keys=True, separators=(",", ":"))
            for entry in entries
        ]
        with self._lock:
            self._buffer.extend(json_lines)
        return self.flush() or CloudFlushResult(accepted=0)

    def health(self) -> dict[str, Any]:
        """Check Cloud API health. Returns health response dict."""
        url = f"{self._endpoint}/v1/health"
        req = urllib.request.Request(url, method="GET", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
