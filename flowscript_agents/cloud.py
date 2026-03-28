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
from typing import Any, Optional


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CLOUD_URL = "https://api.flowscript.org"
DEFAULT_BATCH_SIZE = 100
DEFAULT_FLUSH_INTERVAL = 5.0  # seconds (for future auto-flush)
USER_AGENT = "flowscript-agents-python/0.2.8"


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
        timeout: int = 30,
        on_witness: Optional[callable] = None,
        on_error: Optional[callable] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("FLOWSCRIPT_API_KEY", "")
        self._namespace = namespace or os.environ.get("FLOWSCRIPT_NAMESPACE", "")
        self._endpoint = (
            endpoint or os.environ.get("FLOWSCRIPT_CLOUD_URL", DEFAULT_CLOUD_URL)
        ).rstrip("/")
        self._batch_size = batch_size
        self._timeout = timeout
        self._on_witness = on_witness
        self._on_error = on_error
        self._buffer: list[str] = []
        self._lock = threading.Lock()
        self._total_sent = 0
        self._total_accepted = 0
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

        Args:
            entry: Audit event dict as produced by AuditWriter.write()
        """
        # Re-serialize to canonical JSON — identical to AuditWriter line 390.
        # Python json.dumps with sort_keys=True is deterministic for the same dict.
        json_line = json.dumps(entry, sort_keys=True, separators=(",", ":"))

        with self._lock:
            self._buffer.append(json_line)
            if len(self._buffer) >= self._batch_size:
                self._flush_locked()

    def flush(self) -> Optional[CloudFlushResult]:
        """Send all buffered events to Cloud.

        Returns None if buffer is empty. Otherwise returns CloudFlushResult
        with accepted count and witness attestation.

        Safe to call from any thread, including atexit handlers.
        """
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self) -> Optional[CloudFlushResult]:
        """Internal flush — must be called with self._lock held."""
        if not self._buffer:
            return None

        events = list(self._buffer)
        self._buffer.clear()

        # Build request body
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

                # Parse witness
                witness_data = resp_body.get("witness")
                witness = None
                if witness_data:
                    witness = CloudWitness(
                        id=witness_data["id"],
                        chain_head_seq=witness_data["chain_head_seq"],
                        chain_head_hash=witness_data["chain_head_hash"],
                        witnessed_at=witness_data["witnessed_at"],
                    )
                    self._last_witness = witness
                    if self._on_witness:
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
                self._buffer = events + self._buffer  # prepend for order preservation

            self._handle_error(result)
            return result

        except (urllib.error.URLError, OSError, TimeoutError) as e:
            # Network error — put events back for retry
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
        for entry in entries:
            json_line = json.dumps(entry, sort_keys=True, separators=(",", ":"))
            with self._lock:
                self._buffer.append(json_line)
        return self.flush() or CloudFlushResult(accepted=0)

    def health(self) -> dict[str, Any]:
        """Check Cloud API health. Returns health response dict."""
        url = f"{self._endpoint}/v1/health"
        req = urllib.request.Request(url, method="GET", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
