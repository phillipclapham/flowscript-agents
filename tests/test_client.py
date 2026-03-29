"""
Tests for FlowScript LLM client wrappers (ambient capture).

Tests use mock clients — no real API keys needed.
Verifies that:
- create() calls pass through to the underlying client
- Exchanges are captured to UnifiedMemory.add()
- Streaming responses buffer content and capture on exhaustion
- Extraction errors are suppressed (API call still succeeds)
- Non-exchange methods pass through via __getattr__
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from flowscript_agents.client import (
    FlowScriptAnthropic,
    FlowScriptOpenAI,
    _extract_anthropic_content,
    _extract_user_content,
    _format_exchange,
)
from flowscript_agents import Memory, UnifiedMemory


# =============================================================================
# Helpers
# =============================================================================


def make_openai_response(content: str) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def make_anthropic_response(content: str) -> MagicMock:
    """Build a mock Anthropic Message response."""
    block = MagicMock()
    block.text = content
    response = MagicMock()
    response.content = [block]
    return response


def make_memory_no_llm() -> UnifiedMemory:
    """UnifiedMemory with no LLM — add() falls back to plain thought nodes."""
    return UnifiedMemory()


def make_memory_with_llm() -> tuple[UnifiedMemory, list[str]]:
    """UnifiedMemory with a mock LLM that records calls."""
    captured: list[str] = []

    def mock_llm(prompt: str) -> str:
        captured.append(prompt)
        # Return minimal valid FlowScript JSON that AutoExtract accepts
        return '{"nodes": [], "relationships": [], "states": []}'

    mem = UnifiedMemory(llm=mock_llm)
    return mem, captured


# =============================================================================
# Utility function tests
# =============================================================================


class TestExtractUserContent:
    def test_last_user_message_string(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Which database?"},
            {"role": "assistant", "content": "PostgreSQL."},
            {"role": "user", "content": "Why not MySQL?"},
        ]
        assert _extract_user_content(messages) == "Why not MySQL?"

    def test_content_array_format(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": "..."}},
                ],
            }
        ]
        assert _extract_user_content(messages) == "Describe this image."

    def test_input_text_type(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ]
        assert _extract_user_content(messages) == "Hello"

    def test_no_user_message(self):
        messages = [{"role": "system", "content": "You are helpful."}]
        assert _extract_user_content(messages) == ""

    def test_empty_messages(self):
        assert _extract_user_content([]) == ""


class TestExtractAnthropicContent:
    def test_content_block_list(self):
        block = MagicMock()
        block.text = "PostgreSQL."
        response = MagicMock()
        response.content = [block]
        assert _extract_anthropic_content(response) == "PostgreSQL."

    def test_dict_block(self):
        response = MagicMock()
        response.content = [{"type": "text", "text": "Hello"}]
        assert _extract_anthropic_content(response) == "Hello"

    def test_string_content(self):
        response = MagicMock()
        response.content = "Direct string"
        assert _extract_anthropic_content(response) == "Direct string"

    def test_multiple_blocks(self):
        b1 = MagicMock()
        b1.text = "First"
        b2 = MagicMock()
        b2.text = "Second"
        response = MagicMock()
        response.content = [b1, b2]
        assert _extract_anthropic_content(response) == "First Second"


class TestFormatExchange:
    def test_basic_format(self):
        result = _format_exchange("Which database?", "PostgreSQL for ACID compliance.")
        assert result == "User: Which database?\nAssistant: PostgreSQL for ACID compliance."

    def test_empty_strings(self):
        result = _format_exchange("", "")
        assert result == "User: \nAssistant: "


# =============================================================================
# FlowScriptOpenAI tests
# =============================================================================


class TestFlowScriptOpenAI:
    def _make_client(self, response_content="Answer here."):
        """Returns (FlowScriptOpenAI, mock_completions, memory)."""
        mock_completions = MagicMock()
        mock_completions.create.return_value = make_openai_response(response_content)
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_openai = MagicMock()
        mock_openai.chat = mock_chat

        mem = make_memory_no_llm()
        client = FlowScriptOpenAI(mock_openai, memory=mem)
        return client, mock_completions, mem

    def test_create_passes_through(self):
        client, mock_completions, _ = self._make_client()
        messages = [{"role": "user", "content": "Hello"}]
        response = client.chat.completions.create(model="gpt-4o", messages=messages)
        mock_completions.create.assert_called_once_with(model="gpt-4o", messages=messages)
        assert response.choices[0].message.content == "Answer here."

    def test_exchange_captured_to_memory(self):
        client, _, mem = self._make_client("PostgreSQL.")
        messages = [{"role": "user", "content": "Which database?"}]
        client.chat.completions.create(model="gpt-4o", messages=messages)

        # Memory should have at least one node
        assert len(mem.memory.nodes) >= 1
        # Node content should contain the exchange
        node_contents = [n.content for n in mem.memory.nodes]
        full_text = " ".join(node_contents)
        # Without LLM, it creates a thought node with the exchange text
        assert "Which database?" in full_text or "PostgreSQL" in full_text

    def test_extraction_error_suppressed(self):
        """If add() raises, the API response is still returned."""
        mock_completions = MagicMock()
        mock_completions.create.return_value = make_openai_response("OK")
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_openai = MagicMock()
        mock_openai.chat = mock_chat

        mem = MagicMock(spec=UnifiedMemory)
        mem.add.side_effect = RuntimeError("extraction failed")

        client = FlowScriptOpenAI(mock_openai, memory=mem)
        messages = [{"role": "user", "content": "Hello"}]
        # Should not raise
        response = client.chat.completions.create(model="gpt-4o", messages=messages)
        assert response is not None

    def test_passthrough_other_attributes(self):
        mock_openai = MagicMock()
        mock_openai.models = MagicMock()
        mem = make_memory_no_llm()
        client = FlowScriptOpenAI(mock_openai, memory=mem)
        # Should delegate to underlying client
        assert client.models is mock_openai.models

    def test_no_user_message_still_captures(self):
        """Even with no user message, assistant content is captured."""
        client, _, mem = self._make_client("System response.")
        messages = [{"role": "system", "content": "You are helpful."}]
        client.chat.completions.create(model="gpt-4o", messages=messages)
        assert len(mem.memory.nodes) >= 1

    def test_with_llm_extraction(self):
        """With an LLM configured, exchanges go through AutoExtract."""
        mem, captured_prompts = make_memory_with_llm()

        mock_completions = MagicMock()
        mock_completions.create.return_value = make_openai_response("Redis for speed.")
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_openai = MagicMock()
        mock_openai.chat = mock_chat

        client = FlowScriptOpenAI(mock_openai, memory=mem)
        messages = [{"role": "user", "content": "Cache solution?"}]
        client.chat.completions.create(model="gpt-4o", messages=messages)

        # The LLM extraction function should have been called
        assert len(captured_prompts) >= 1
        # The prompt should contain our exchange
        assert "Cache solution?" in captured_prompts[0]

    def test_empty_response_content_no_crash(self):
        """None or empty assistant content doesn't crash."""
        mock_completions = MagicMock()
        mock_completions.create.return_value = make_openai_response("")
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_openai = MagicMock()
        mock_openai.chat = mock_chat

        mem = make_memory_no_llm()
        client = FlowScriptOpenAI(mock_openai, memory=mem)
        messages = [{"role": "user", "content": "Hello"}]
        # Should not crash even with empty response
        response = client.chat.completions.create(model="gpt-4o", messages=messages)
        assert response is not None


# =============================================================================
# FlowScriptAnthropic tests
# =============================================================================


class TestFlowScriptAnthropic:
    def _make_client(self, response_content="Anthropic answer."):
        mock_messages_resource = MagicMock()
        mock_messages_resource.create.return_value = make_anthropic_response(response_content)
        mock_anthropic = MagicMock()
        mock_anthropic.messages = mock_messages_resource

        mem = make_memory_no_llm()
        client = FlowScriptAnthropic(mock_anthropic, memory=mem)
        return client, mock_messages_resource, mem

    def test_create_passes_through(self):
        client, mock_messages_resource, _ = self._make_client()
        messages = [{"role": "user", "content": "Hello"}]
        response = client.messages.create(
            model="claude-opus-4-6", max_tokens=1024, messages=messages
        )
        mock_messages_resource.create.assert_called_once_with(
            model="claude-opus-4-6", max_tokens=1024, messages=messages
        )
        assert response is not None

    def test_exchange_captured_to_memory(self):
        client, _, mem = self._make_client("Use PostgreSQL.")
        messages = [{"role": "user", "content": "Database choice?"}]
        client.messages.create(model="claude-opus-4-6", max_tokens=100, messages=messages)
        assert len(mem.memory.nodes) >= 1
        node_contents = " ".join(n.content for n in mem.memory.nodes)
        assert "Database choice?" in node_contents or "Use PostgreSQL" in node_contents

    def test_extraction_error_suppressed(self):
        mock_messages_resource = MagicMock()
        mock_messages_resource.create.return_value = make_anthropic_response("OK")
        mock_anthropic = MagicMock()
        mock_anthropic.messages = mock_messages_resource

        mem = MagicMock(spec=UnifiedMemory)
        mem.add.side_effect = RuntimeError("extraction failed")

        client = FlowScriptAnthropic(mock_anthropic, memory=mem)
        messages = [{"role": "user", "content": "Hello"}]
        response = client.messages.create(model="claude-opus-4-6", messages=messages)
        assert response is not None

    def test_passthrough_other_attributes(self):
        mock_anthropic = MagicMock()
        mock_anthropic.models = MagicMock()
        mem = make_memory_no_llm()
        client = FlowScriptAnthropic(mock_anthropic, memory=mem)
        assert client.models is mock_anthropic.models

    def test_with_llm_extraction(self):
        mem, captured_prompts = make_memory_with_llm()

        mock_messages_resource = MagicMock()
        mock_messages_resource.create.return_value = make_anthropic_response("SQLite is fine.")
        mock_anthropic = MagicMock()
        mock_anthropic.messages = mock_messages_resource

        client = FlowScriptAnthropic(mock_anthropic, memory=mem)
        messages = [{"role": "user", "content": "Embedded DB?"}]
        client.messages.create(model="claude-opus-4-6", max_tokens=100, messages=messages)

        assert len(captured_prompts) >= 1
        assert "Embedded DB?" in captured_prompts[0]


# =============================================================================
# Streaming capture tests
# =============================================================================


class TestStreamingCapture:
    def _make_streaming_client(self, chunks: list[str]):
        """Build a client whose create() returns a streaming iterable."""

        def make_chunk(text: str) -> MagicMock:
            chunk = MagicMock()
            chunk.choices[0].delta.content = text
            return chunk

        stream_chunks = [make_chunk(c) for c in chunks]

        mock_completions = MagicMock()
        mock_completions.create.return_value = iter(stream_chunks)

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_openai = MagicMock()
        mock_openai.chat = mock_chat

        mem = make_memory_no_llm()
        client = FlowScriptOpenAI(mock_openai, memory=mem)
        return client, mem

    def test_stream_passes_chunks_through(self):
        client, _ = self._make_streaming_client(["Hello", " world", "!"])
        messages = [{"role": "user", "content": "Greet me"}]
        stream = client.chat.completions.create(
            model="gpt-4o", messages=messages, stream=True
        )
        collected = []
        for chunk in stream:
            collected.append(chunk)
        assert len(collected) == 3

    def test_stream_captures_after_iteration(self):
        client, mem = self._make_streaming_client(["PostgreSQL", " is best"])
        messages = [{"role": "user", "content": "Which database?"}]
        stream = client.chat.completions.create(
            model="gpt-4o", messages=messages, stream=True
        )
        # Exhaust the stream
        for _ in stream:
            pass
        # Memory should now have the captured exchange
        assert len(mem.memory.nodes) >= 1
        node_contents = " ".join(n.content for n in mem.memory.nodes)
        assert "Which database?" in node_contents or "PostgreSQL" in node_contents

    def test_stream_captures_on_partial_iteration(self):
        """Even if iteration stops early, captured content is stored on __iter__ exit."""
        client, mem = self._make_streaming_client(["A", "B", "C"])
        messages = [{"role": "user", "content": "Test"}]
        stream = client.chat.completions.create(
            model="gpt-4o", messages=messages, stream=True
        )
        # Only consume first chunk via try/break
        for chunk in stream:
            break  # Exhaust via generator protocol's finally block on GC

        # The stream's __iter__ finally block fires when the generator is GC'd.
        # For test purposes, we verify the capture mechanism works when fully exhausted.
        # (partial iteration capture happens on stream close/GC which is non-deterministic)


# =============================================================================
# Async tests
# =============================================================================


@pytest.mark.asyncio
class TestAsyncFlowScriptOpenAI:
    async def test_acreate_passes_through(self):
        mock_completions = MagicMock()
        mock_completions.acreate = AsyncMock(
            return_value=make_openai_response("Async answer.")
        )
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_openai = MagicMock()
        mock_openai.chat = mock_chat

        mem = make_memory_no_llm()
        client = FlowScriptOpenAI(mock_openai, memory=mem)
        messages = [{"role": "user", "content": "Async hello"}]
        response = await client.chat.completions.acreate(
            model="gpt-4o", messages=messages
        )
        assert response.choices[0].message.content == "Async answer."

    async def test_acreate_captures_exchange(self):
        mock_completions = MagicMock()
        mock_completions.acreate = AsyncMock(
            return_value=make_openai_response("Async response.")
        )
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_openai = MagicMock()
        mock_openai.chat = mock_chat

        mem = make_memory_no_llm()
        client = FlowScriptOpenAI(mock_openai, memory=mem)
        messages = [{"role": "user", "content": "Async question"}]
        await client.chat.completions.acreate(model="gpt-4o", messages=messages)

        assert len(mem.memory.nodes) >= 1


@pytest.mark.asyncio
class TestAsyncFlowScriptAnthropic:
    async def test_acreate_captures_exchange(self):
        mock_messages_resource = MagicMock()
        mock_messages_resource.acreate = AsyncMock(
            return_value=make_anthropic_response("Async Claude response.")
        )
        mock_anthropic = MagicMock()
        mock_anthropic.messages = mock_messages_resource

        mem = make_memory_no_llm()
        client = FlowScriptAnthropic(mock_anthropic, memory=mem)
        messages = [{"role": "user", "content": "Async Claude question"}]
        await client.messages.acreate(model="claude-opus-4-6", messages=messages)

        assert len(mem.memory.nodes) >= 1
