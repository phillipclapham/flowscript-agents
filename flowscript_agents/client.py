"""
FlowScript LLM Client Wrappers — Ambient conversation capture.

Wraps OpenAI and Anthropic clients to automatically capture every
conversation exchange into FlowScript memory. No explicit mem.add()
calls required — the wrapper intercepts each completion and ingests
the full exchange via AutoExtract.

This closes the "ambient capture" gap: the difference between
explicit instrumentation ("call add() each time") and truly ambient
capture ("it just works").

Usage (OpenAI):
    from openai import OpenAI
    from flowscript_agents import UnifiedMemory
    from flowscript_agents.client import FlowScriptOpenAI
    from flowscript_agents.embeddings import OpenAIEmbeddings

    mem = UnifiedMemory("./agent.json", embedder=OpenAIEmbeddings(), llm=my_llm)
    client = FlowScriptOpenAI(OpenAI(), memory=mem)

    # Normal OpenAI usage — memory happens automatically
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Which database should we use?"}]
    )
    # FlowScript has the exchange captured. Query reasoning:
    mem.memory.query.why(...)
    mem.memory.query.tensions()

Usage (Anthropic):
    from anthropic import Anthropic
    from flowscript_agents.client import FlowScriptAnthropic

    client = FlowScriptAnthropic(Anthropic(), memory=mem)
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Which database should we use?"}]
    )

Streaming:
    Streaming completions are supported. The wrapper buffers the streamed
    content and runs extraction after the stream completes. The caller
    receives a transparent wrapper that yields chunks normally.

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[...],
        stream=True,
    )
    for chunk in response:
        print(chunk.choices[0].delta.content or "", end="")
    # Extraction runs after the loop finishes (on stream close).

Error handling:
    If extraction fails, the original API response is still returned.
    Extraction errors are logged to stderr but never raised to the caller.

Notes:
    - Only chat.completions.create (OpenAI) and messages.create (Anthropic)
      are wrapped. Other client methods pass through unchanged.
    - Requires flowscript-agents with an LLM configured in UnifiedMemory
      for full extraction. Without an LLM, exchanges are stored as plain
      thought nodes (still queryable, just not semantically typed).
    - Not thread-safe: serialize access or use one wrapper per thread.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Iterator, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from .unified import UnifiedMemory


def _log(msg: str) -> None:
    sys.stderr.write(f"[flowscript] {msg}\n")
    sys.stderr.flush()


def _format_exchange(user_content: str, assistant_content: str) -> str:
    """Format a conversation exchange for AutoExtract ingestion."""
    parts = []
    if user_content:
        parts.append(f"User: {user_content}")
    if assistant_content:
        parts.append(f"Assistant: {assistant_content}")
    return "\n".join(parts)


def _extract_user_content(messages: list[dict[str, Any]]) -> str:
    """Extract the last user message content from a messages list."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            # Content array format (vision, etc.)
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                        parts.append(part.get("text", ""))
                return " ".join(parts)
    return ""


def _safe_add(
    memory: "UnifiedMemory",
    text: str,
    on_error: "Callable[[Exception, str], None] | None" = None,
) -> bool:
    """Add text to memory, swallowing errors so API calls always succeed.

    Returns True if extraction succeeded, False if it failed.
    If on_error is provided, calls it with (exception, exchange_text) on failure.
    Errors are always logged to stderr regardless of on_error.
    """
    try:
        memory.add(text, actor="agent")
        return True
    except Exception as exc:
        _log(f"extraction error (suppressed): {exc}")
        if on_error is not None:
            try:
                on_error(exc, text)
            except Exception:
                pass  # callback errors must not propagate
        return False


# =============================================================================
# OpenAI wrapper
# =============================================================================


class _FlowScriptCompletions:
    """Wraps openai.resources.chat.completions.Completions."""

    def __init__(
        self,
        completions: Any,
        memory: "UnifiedMemory",
        on_extraction_error: "Callable[[Exception, str], None] | None" = None,
    ) -> None:
        self._completions = completions
        self._memory = memory
        self._on_error = on_extraction_error
        self.failed_extraction_count = 0

    def _track_add(self, text: str) -> None:
        if not _safe_add(self._memory, text, self._on_error):
            self.failed_extraction_count += 1

    def create(self, **kwargs: Any) -> Any:
        """Create a chat completion and capture the exchange."""
        messages: list[dict[str, Any]] = kwargs.get("messages", [])
        stream: bool = kwargs.get("stream", False)

        response = self._completions.create(**kwargs)

        if stream:
            return _StreamingCapture(response, messages, self._memory, self._track_add)

        # Non-streaming: extract immediately
        user_content = _extract_user_content(messages)
        try:
            assistant_content = response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            assistant_content = ""

        if user_content or assistant_content:
            self._track_add(_format_exchange(user_content, assistant_content))

        return response

    async def acreate(self, **kwargs: Any) -> Any:
        """Async create — capture exchange after awaiting response.

        Works with both legacy ``acreate()`` and modern (v1+) async ``create()``.
        Falls back to ``create()`` if the underlying client has no ``acreate``.
        """
        messages: list[dict[str, Any]] = kwargs.get("messages", [])
        stream: bool = kwargs.get("stream", False)

        # Modern OpenAI SDK (v1+): async client uses create(), not acreate()
        if hasattr(self._completions, "acreate"):
            response = await self._completions.acreate(**kwargs)
        else:
            response = await self._completions.create(**kwargs)

        if stream:
            return _AsyncStreamingCapture(response, messages, self._memory, self._track_add)

        user_content = _extract_user_content(messages)
        try:
            assistant_content = response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            assistant_content = ""

        if user_content or assistant_content:
            self._track_add(_format_exchange(user_content, assistant_content))

        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _FlowScriptChat:
    """Wraps openai.resources.chat.Chat."""

    def __init__(
        self,
        chat: Any,
        memory: "UnifiedMemory",
        on_extraction_error: "Callable[[Exception, str], None] | None" = None,
    ) -> None:
        self._chat = chat
        self.completions = _FlowScriptCompletions(chat.completions, memory, on_extraction_error)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class FlowScriptOpenAI:
    """OpenAI client wrapper with ambient FlowScript memory capture.

    Drop-in replacement for openai.OpenAI (and AsyncOpenAI).
    All API methods pass through unchanged except:
    - client.chat.completions.create() → captures exchange to memory

    Note: Not thread-safe. Serialize access to the wrapped client or
    use one wrapper instance per thread.

    Args:
        client: An openai.OpenAI (or AsyncOpenAI) instance.
        memory: A UnifiedMemory instance with LLM configured for extraction.
                Without an LLM, exchanges are stored as plain thought nodes.
        on_extraction_error: Optional callback ``(exception, exchange_text) -> None``
                called when extraction fails. Errors are always logged to stderr;
                this callback provides programmatic visibility for compliance use cases.

    Attributes:
        failed_extraction_count: Number of exchanges where extraction failed.
            Always available, zero cost. Check this to detect silent gaps in
            the audit trail.

    Example::

        from openai import OpenAI
        from flowscript_agents import UnifiedMemory
        from flowscript_agents.client import FlowScriptOpenAI

        mem = UnifiedMemory("./agent.json", embedder=embedder, llm=my_llm)
        client = FlowScriptOpenAI(OpenAI(), memory=mem)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Which database?"}]
        )
        mem.memory.query.tensions()  # FlowScript has the exchange
        assert client.failed_extraction_count == 0  # no gaps
    """

    def __init__(
        self,
        client: Any,
        memory: "UnifiedMemory",
        on_extraction_error: "Callable[[Exception, str], None] | None" = None,
    ) -> None:
        self._client = client
        self._memory = memory
        self.chat = _FlowScriptChat(client.chat, memory, on_extraction_error)

    @property
    def failed_extraction_count(self) -> int:
        return self.chat.completions.failed_extraction_count

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# =============================================================================
# Anthropic wrapper
# =============================================================================


class _FlowScriptMessages:
    """Wraps anthropic.resources.Messages."""

    def __init__(
        self,
        messages_resource: Any,
        memory: "UnifiedMemory",
        on_extraction_error: "Callable[[Exception, str], None] | None" = None,
    ) -> None:
        self._messages = messages_resource
        self._memory = memory
        self._on_error = on_extraction_error
        self.failed_extraction_count = 0

    def _track_add(self, text: str) -> None:
        if not _safe_add(self._memory, text, self._on_error):
            self.failed_extraction_count += 1

    def create(self, **kwargs: Any) -> Any:
        """Create a message and capture the exchange."""
        messages: list[dict[str, Any]] = kwargs.get("messages", [])
        stream: bool = kwargs.get("stream", False)

        response = self._messages.create(**kwargs)

        if stream:
            return _AnthropicStreamingCapture(response, messages, self._memory, self._track_add)

        # Non-streaming: extract immediately
        user_content = _extract_user_content(messages)
        assistant_content = _extract_anthropic_content(response)

        if user_content or assistant_content:
            self._track_add(_format_exchange(user_content, assistant_content))

        return response

    async def acreate(self, **kwargs: Any) -> Any:
        """Async create.

        Works with both legacy ``acreate()`` and modern async ``create()``.
        """
        messages: list[dict[str, Any]] = kwargs.get("messages", [])
        stream: bool = kwargs.get("stream", False)

        if hasattr(self._messages, "acreate"):
            response = await self._messages.acreate(**kwargs)
        else:
            response = await self._messages.create(**kwargs)

        if stream:
            return _AnthropicAsyncStreamingCapture(response, messages, self._memory, self._track_add)

        user_content = _extract_user_content(messages)
        assistant_content = _extract_anthropic_content(response)

        if user_content or assistant_content:
            self._track_add(_format_exchange(user_content, assistant_content))

        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


def _extract_anthropic_content(response: Any) -> str:
    """Extract text content from an Anthropic Message response."""
    try:
        content = response.content
        if isinstance(content, list):
            parts = []
            for block in content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return " ".join(parts)
        if isinstance(content, str):
            return content
    except (AttributeError, TypeError):
        pass
    return ""


class FlowScriptAnthropic:
    """Anthropic client wrapper with ambient FlowScript memory capture.

    Drop-in replacement for anthropic.Anthropic (and AsyncAnthropic).
    All API methods pass through unchanged except:
    - client.messages.create() → captures exchange to memory

    Note: Not thread-safe. Serialize access to the wrapped client or
    use one wrapper instance per thread.

    Args:
        client: An anthropic.Anthropic (or AsyncAnthropic) instance.
        memory: A UnifiedMemory instance with LLM configured for extraction.
        on_extraction_error: Optional callback ``(exception, exchange_text) -> None``
                called when extraction fails. Errors are always logged to stderr;
                this callback provides programmatic visibility for compliance use cases.

    Attributes:
        failed_extraction_count: Number of exchanges where extraction failed.
            Always available, zero cost. Check this to detect silent gaps in
            the audit trail.

    Example::

        from anthropic import Anthropic
        from flowscript_agents import UnifiedMemory
        from flowscript_agents.client import FlowScriptAnthropic

        mem = UnifiedMemory("./agent.json", embedder=embedder, llm=my_llm)
        client = FlowScriptAnthropic(Anthropic(), memory=mem)

        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Which database?"}]
        )
        mem.memory.query.tensions()  # FlowScript has the exchange
        assert client.failed_extraction_count == 0  # no gaps
    """

    def __init__(
        self,
        client: Any,
        memory: "UnifiedMemory",
        on_extraction_error: "Callable[[Exception, str], None] | None" = None,
    ) -> None:
        self._client = client
        self._memory = memory
        self.messages = _FlowScriptMessages(client.messages, memory, on_extraction_error)

    @property
    def failed_extraction_count(self) -> int:
        return self.messages.failed_extraction_count

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# =============================================================================
# Streaming capture wrappers
# =============================================================================


class _StreamingCapture:
    """Wraps an OpenAI streaming response, captures content after iteration."""

    def __init__(
        self,
        stream: Any,
        messages: list[dict[str, Any]],
        memory: "UnifiedMemory",
        track_add: "Callable[[str], None] | None" = None,
    ) -> None:
        self._stream = stream
        self._messages = messages
        self._memory = memory
        self._track_add = track_add
        self._chunks: list[str] = []

    def __iter__(self) -> Iterator[Any]:
        try:
            for chunk in self._stream:
                try:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        self._chunks.append(delta)
                except (AttributeError, IndexError):
                    pass
                yield chunk
        finally:
            self._capture()

    def _capture(self) -> None:
        user_content = _extract_user_content(self._messages)
        assistant_content = "".join(self._chunks)
        if user_content or assistant_content:
            exchange = _format_exchange(user_content, assistant_content)
            if self._track_add:
                self._track_add(exchange)
            else:
                _safe_add(self._memory, exchange)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _AsyncStreamingCapture:
    """Wraps an OpenAI async streaming response."""

    def __init__(
        self,
        stream: Any,
        messages: list[dict[str, Any]],
        memory: "UnifiedMemory",
        track_add: "Callable[[str], None] | None" = None,
    ) -> None:
        self._stream = stream
        self._messages = messages
        self._memory = memory
        self._track_add = track_add
        self._chunks: list[str] = []

    async def __aiter__(self) -> AsyncIterator[Any]:
        try:
            async for chunk in self._stream:
                try:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        self._chunks.append(delta)
                except (AttributeError, IndexError):
                    pass
                yield chunk
        finally:
            self._capture()

    def _capture(self) -> None:
        user_content = _extract_user_content(self._messages)
        assistant_content = "".join(self._chunks)
        if user_content or assistant_content:
            exchange = _format_exchange(user_content, assistant_content)
            if self._track_add:
                self._track_add(exchange)
            else:
                _safe_add(self._memory, exchange)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _AnthropicStreamingCapture:
    """Wraps an Anthropic streaming context manager."""

    def __init__(
        self,
        stream: Any,
        messages: list[dict[str, Any]],
        memory: "UnifiedMemory",
        track_add: "Callable[[str], None] | None" = None,
    ) -> None:
        self._stream = stream
        self._messages = messages
        self._memory = memory
        self._track_add = track_add
        self._chunks: list[str] = []
        self._captured = False

    def __iter__(self) -> Iterator[Any]:
        try:
            for event in self._stream:
                # Anthropic streaming: ContentBlockDelta events
                try:
                    if hasattr(event, "delta") and hasattr(event.delta, "text"):
                        self._chunks.append(event.delta.text)
                except AttributeError:
                    pass
                yield event
        finally:
            self._capture()

    def _capture(self) -> None:
        if self._captured:
            return
        self._captured = True
        user_content = _extract_user_content(self._messages)
        assistant_content = "".join(self._chunks)
        if user_content or assistant_content:
            exchange = _format_exchange(user_content, assistant_content)
            if self._track_add:
                self._track_add(exchange)
            else:
                _safe_add(self._memory, exchange)

    def __enter__(self) -> "_AnthropicStreamingCapture":
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        try:
            if hasattr(self._stream, "__exit__"):
                return self._stream.__exit__(*args)
        finally:
            self._capture()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _AnthropicAsyncStreamingCapture:
    """Wraps an Anthropic async streaming context manager."""

    def __init__(
        self,
        stream: Any,
        messages: list[dict[str, Any]],
        memory: "UnifiedMemory",
        track_add: "Callable[[str], None] | None" = None,
    ) -> None:
        self._stream = stream
        self._messages = messages
        self._memory = memory
        self._track_add = track_add
        self._chunks: list[str] = []
        self._captured = False

    async def __aiter__(self) -> AsyncIterator[Any]:
        try:
            async for event in self._stream:
                try:
                    if hasattr(event, "delta") and hasattr(event.delta, "text"):
                        self._chunks.append(event.delta.text)
                except AttributeError:
                    pass
                yield event
        finally:
            self._capture()

    def _capture(self) -> None:
        if self._captured:
            return
        self._captured = True
        user_content = _extract_user_content(self._messages)
        assistant_content = "".join(self._chunks)
        if user_content or assistant_content:
            exchange = _format_exchange(user_content, assistant_content)
            if self._track_add:
                self._track_add(exchange)
            else:
                _safe_add(self._memory, exchange)

    async def __aenter__(self) -> "_AnthropicAsyncStreamingCapture":
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> Any:
        try:
            if hasattr(self._stream, "__aexit__"):
                return await self._stream.__aexit__(*args)
        finally:
            self._capture()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)
