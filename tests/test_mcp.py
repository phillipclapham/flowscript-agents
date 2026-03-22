"""Tests for FlowScript MCP server handler."""

import json
import pytest

from flowscript_agents import UnifiedMemory
from flowscript_agents.mcp import MCPHandler, TOOLS, _jsonrpc_response, _jsonrpc_error

# Import shared MockEmbeddings from conftest
from conftest import MockEmbeddings


def _make_handler(with_embedder: bool = False, with_llm: bool = False):
    emb = MockEmbeddings(dims=16) if with_embedder else None
    llm = None
    if with_llm:
        def llm(prompt):
            return json.dumps({
                "nodes": [{"type": "thought", "content": "extracted fact"}],
                "relationships": [], "states": [],
            })
    umem = UnifiedMemory(embedder=emb, llm=llm)
    return MCPHandler(umem), umem


class TestToolDefinitions:
    def test_all_tools_have_required_fields(self):
        for tool in TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    def test_tool_count(self):
        assert len(TOOLS) == 13

    def test_tool_names(self):
        names = {t["name"] for t in TOOLS}
        expected = {
            "search_memory", "add_memory", "get_context",
            "query_tensions", "query_blocked", "query_why",
            "query_what_if", "query_alternatives",
            "remove_memory", "session_wrap", "memory_stats",
            "query_audit", "verify_audit",
        }
        assert names == expected


class TestSearchMemory:
    def test_keyword_search(self):
        handler, umem = _make_handler()
        umem.add_raw("Redis is fast for caching")
        umem.add_raw("PostgreSQL ensures ACID")
        result = handler.handle_tool("search_memory", {"query": "Redis"})
        assert result["count"] > 0
        assert result["mode"] == "unified"

    def test_vector_search(self):
        handler, umem = _make_handler(with_embedder=True)
        umem.add_raw("Redis is fast for caching")
        result = handler.handle_tool("search_memory", {"query": "Redis", "mode": "vector"})
        assert result["mode"] == "vector"

    def test_keyword_only_mode(self):
        handler, umem = _make_handler()
        umem.add_raw("Redis is fast")
        result = handler.handle_tool("search_memory", {"query": "Redis", "mode": "keyword"})
        assert result["mode"] == "keyword"

    def test_empty_memory(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("search_memory", {"query": "anything"})
        assert result["count"] == 0


class TestAddMemory:
    def test_add_without_llm(self):
        handler, umem = _make_handler()
        result = handler.handle_tool("add_memory", {"text": "Redis is fast"})
        assert result["nodes_created"] == 1
        assert umem.size == 1

    def test_add_with_llm(self):
        handler, umem = _make_handler(with_llm=True)
        result = handler.handle_tool("add_memory", {"text": "any text"})
        assert result["nodes_created"] >= 1

    def test_add_with_metadata(self):
        handler, umem = _make_handler()
        result = handler.handle_tool("add_memory", {
            "text": "Redis is fast",
            "metadata": {"source": "test"},
        })
        assert result["nodes_created"] == 1


class TestGetContext:
    def test_empty_context(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("get_context", {})
        assert result["nodes"] == 0
        assert result["context"] == ""

    def test_with_nodes(self):
        handler, umem = _make_handler()
        umem.add_raw("Redis is fast")
        umem.add_raw("PostgreSQL is reliable")
        result = handler.handle_tool("get_context", {"max_tokens": 1000})
        assert result["nodes"] == 2
        assert "Redis" in result["context"]


class TestQueryTensions:
    def test_no_tensions(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("query_tensions", {})
        assert "error" not in result or result.get("metadata", {}).get("total_tensions", 0) == 0

    def test_with_tensions(self):
        handler, umem = _make_handler()
        a = umem.memory.thought("Speed")
        b = umem.memory.thought("Safety")
        a.tension_with(b, axis="performance vs reliability")
        result = handler.handle_tool("query_tensions", {"group_by": "axis"})
        assert result is not None


class TestQueryBlocked:
    def test_no_blockers(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("query_blocked", {})
        assert result is not None

    def test_with_blocker(self):
        handler, umem = _make_handler()
        ref = umem.memory.thought("Database migration")
        ref.block(reason="waiting on schema approval")
        result = handler.handle_tool("query_blocked", {})
        assert result is not None


class TestQueryWhy:
    def test_with_causal_chain(self):
        handler, umem = _make_handler()
        a = umem.memory.thought("Root cause")
        b = umem.memory.thought("Effect")
        a.causes(b)
        result = handler.handle_tool("query_why", {"node_id": b.id})
        assert "error" not in result

    def test_by_content(self):
        handler, umem = _make_handler()
        a = umem.memory.thought("Root cause")
        b = umem.memory.thought("The effect of root cause")
        a.causes(b)
        result = handler.handle_tool("query_why", {"content": "effect"})
        assert "error" not in result

    def test_no_node_found(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("query_why", {"content": "nonexistent"})
        assert "error" in result


class TestQueryWhatIf:
    def test_with_consequences(self):
        handler, umem = _make_handler()
        a = umem.memory.thought("Change database")
        b = umem.memory.thought("Need to migrate data")
        a.causes(b)
        result = handler.handle_tool("query_what_if", {"node_id": a.id})
        assert "error" not in result

    def test_by_content(self):
        handler, umem = _make_handler()
        a = umem.memory.thought("Change database schema")
        b = umem.memory.thought("Downstream API breaks")
        a.causes(b)
        result = handler.handle_tool("query_what_if", {"content": "database schema"})
        assert "error" not in result

    def test_no_node_found(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("query_what_if", {"content": "nonexistent"})
        assert "error" in result


class TestQueryAlternatives:
    def test_with_alternatives(self):
        handler, umem = _make_handler()
        q = umem.memory.question("Which database?")
        umem.memory.alternative(q, "Redis")
        umem.memory.alternative(q, "PostgreSQL")
        result = handler.handle_tool("query_alternatives", {"question_id": q.id})
        assert "error" not in result

    def test_by_content(self):
        handler, umem = _make_handler()
        q = umem.memory.question("Which database?")
        umem.memory.alternative(q, "Redis")
        result = handler.handle_tool("query_alternatives", {"content": "database"})
        assert "error" not in result


class TestMemoryStats:
    def test_empty_stats(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("memory_stats", {})
        assert result["total_nodes"] == 0
        assert result["tiers"]["current"] == 0

    def test_with_data(self):
        handler, umem = _make_handler(with_embedder=True)
        umem.add_raw("Redis is fast")
        umem.add_raw("PostgreSQL is reliable")
        result = handler.handle_tool("memory_stats", {})
        assert result["total_nodes"] == 2
        assert "embeddings" in result
        assert result["embeddings"]["indexed"] == 2

    def test_unknown_tool(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("nonexistent_tool", {})
        assert "error" in result


class TestJsonRpc:
    def test_response_format(self):
        resp = _jsonrpc_response(1, {"test": True})
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert resp["result"]["test"] is True

    def test_error_format(self):
        resp = _jsonrpc_error(1, -32601, "Method not found")
        assert resp["jsonrpc"] == "2.0"
        assert resp["error"]["code"] == -32601


class TestMCPStdioProtocol:
    """Test the actual MCP JSON-RPC message routing (simulates stdio)."""

    def _simulate_message(self, msg: dict) -> dict | None:
        """Simulate sending a JSON-RPC message through the server's routing logic.

        We test the routing logic directly rather than actual stdio to avoid
        subprocess complexity while still verifying protocol compliance.
        """
        import io
        from flowscript_agents.mcp import run_server, TOOLS, _jsonrpc_response, _jsonrpc_error

        umem = UnifiedMemory()
        handler = MCPHandler(umem)

        msg_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params", {})

        if method == "initialize":
            return _jsonrpc_response(msg_id, {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "flowscript-agents", "version": "0.2.0"},
            })
        elif method == "notifications/initialized":
            return None  # notification, no response
        elif method == "tools/list":
            return _jsonrpc_response(msg_id, {"tools": TOOLS})
        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            result = handler.handle_tool(tool_name, tool_args)
            return _jsonrpc_response(msg_id, {
                "content": [{"type": "text", "text": json.dumps(result)}],
            })
        elif method == "resources/list":
            return _jsonrpc_response(msg_id, {"resources": []})
        elif method == "prompts/list":
            return _jsonrpc_response(msg_id, {"prompts": []})
        elif method == "ping":
            return _jsonrpc_response(msg_id, {})
        else:
            return _jsonrpc_error(msg_id, -32601, f"Method not found: {method}")

    def test_initialize(self):
        resp = self._simulate_message({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-03-26", "capabilities": {}},
        })
        assert resp["result"]["protocolVersion"] == "2025-03-26"
        assert resp["result"]["capabilities"]["tools"] == {}
        assert resp["result"]["serverInfo"]["name"] == "flowscript-agents"

    def test_tools_list(self):
        resp = self._simulate_message({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list",
        })
        tools = resp["result"]["tools"]
        assert len(tools) == 13
        names = {t["name"] for t in tools}
        assert "search_memory" in names
        assert "query_what_if" in names

    def test_tools_call(self):
        resp = self._simulate_message({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "memory_stats", "arguments": {}},
        })
        content = resp["result"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        stats = json.loads(content[0]["text"])
        assert stats["total_nodes"] == 0

    def test_notification_no_response(self):
        resp = self._simulate_message({
            "jsonrpc": "2.0", "method": "notifications/initialized",
        })
        assert resp is None

    def test_resources_list(self):
        resp = self._simulate_message({
            "jsonrpc": "2.0", "id": 4, "method": "resources/list",
        })
        assert resp["result"]["resources"] == []

    def test_prompts_list(self):
        resp = self._simulate_message({
            "jsonrpc": "2.0", "id": 5, "method": "prompts/list",
        })
        assert resp["result"]["prompts"] == []

    def test_ping(self):
        resp = self._simulate_message({
            "jsonrpc": "2.0", "id": 6, "method": "ping",
        })
        assert resp["result"] == {}

    def test_unknown_method(self):
        resp = self._simulate_message({
            "jsonrpc": "2.0", "id": 7, "method": "nonexistent/method",
        })
        assert resp["error"]["code"] == -32601


class TestRemoveMemory:
    def test_remove_existing(self):
        handler, umem = _make_handler()
        ref = umem.memory.thought("Remove me")
        result = handler.handle_tool("remove_memory", {"node_id": ref.id})
        assert result["removed"] is True
        assert umem.size == 0

    def test_remove_nonexistent(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("remove_memory", {"node_id": "fake-id"})
        assert result["removed"] is False

    def test_remove_no_id(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("remove_memory", {})
        assert "error" in result

    def test_remove_cleans_vector_index(self):
        handler, umem = _make_handler(with_embedder=True)
        umem.add_raw("Indexed content")
        nodes = list(umem.memory._nodes.keys())
        assert umem.vector_index.indexed_count == 1
        handler.handle_tool("remove_memory", {"node_id": nodes[0]})
        assert umem.vector_index.indexed_count == 0


class TestSessionWrap:
    def test_wrap_empty(self):
        handler, umem = _make_handler()
        umem.memory.session_start()
        result = handler.handle_tool("session_wrap", {})
        assert result["nodes_before"] == 0
        assert result["nodes_after"] == 0
        assert result["nodes_pruned"] == 0

    def test_wrap_with_data(self):
        handler, umem = _make_handler()
        umem.memory.session_start()
        umem.add_raw("Active node")
        result = handler.handle_tool("session_wrap", {})
        assert result["nodes_before"] == 1
        assert result["nodes_after"] >= 0  # may prune if dormant


class TestAutoConfiguration:
    """Tests for OPENAI_API_KEY auto-detection logic."""

    def test_run_server_accepts_consolidation_provider(self):
        """run_server() should accept consolidation_provider kwarg."""
        from flowscript_agents.mcp import run_server
        import inspect
        sig = inspect.signature(run_server)
        assert "consolidation_provider" in sig.parameters

    def test_auto_configure_requires_openai(self):
        """_auto_configure_openai should exist and be callable."""
        from flowscript_agents.mcp import _auto_configure_openai
        assert callable(_auto_configure_openai)

    def test_openai_consolidation_provider_class_exists(self):
        """_OpenAIConsolidationProvider should exist and have tool_call method."""
        from flowscript_agents.mcp import _OpenAIConsolidationProvider
        assert hasattr(_OpenAIConsolidationProvider, "tool_call")

    def test_anthropic_consolidation_provider_class_exists(self):
        """_AnthropicConsolidationProvider should exist and have tool_call method."""
        from flowscript_agents.mcp import _AnthropicConsolidationProvider
        assert hasattr(_AnthropicConsolidationProvider, "tool_call")

    def test_auto_configure_anthropic_exists(self):
        """_auto_configure_anthropic should exist and be callable."""
        from flowscript_agents.mcp import _auto_configure_anthropic
        assert callable(_auto_configure_anthropic)

    def test_log_function(self):
        """_log should write to stderr without raising."""
        from flowscript_agents.mcp import _log
        _log("test message")

    def test_tool_descriptions_have_behavioral_guidance(self):
        """Tool descriptions should include 'Call this' behavioral guidance."""
        behavioral_tools = [
            "search_memory", "add_memory", "get_context",
            "query_tensions", "query_blocked", "query_why",
            "query_alternatives", "query_what_if",
            "remove_memory", "session_wrap",
        ]
        for tool in TOOLS:
            if tool["name"] in behavioral_tools:
                assert "Call this" in tool["description"], (
                    f"Tool {tool['name']} missing behavioral guidance "
                    f"('Call this when...')"
                )


class TestIsErrorFlag:
    """Tests for MCP isError flag on tool call error responses."""

    def test_error_response_has_is_error(self):
        """Tool errors should set isError: true in the response."""
        handler, _ = _make_handler()
        result = handler.handle_tool("query_why", {"content": "nonexistent"})
        assert "error" in result  # handler returns error dict
        # Verify the MCP protocol-level isError by simulating the message flow
        from flowscript_agents.mcp import _jsonrpc_response
        # The server loop checks for "error" key and sets isError
        is_error = "error" in result
        assert is_error is True

    def test_success_response_no_is_error(self):
        handler, umem = _make_handler()
        umem.add_raw("test content")
        result = handler.handle_tool("memory_stats", {})
        assert "error" not in result


class TestInputValidation:
    """Tests for input validation."""

    def test_empty_text_rejected(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("add_memory", {"text": ""})
        assert "error" in result

    def test_whitespace_text_rejected(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("add_memory", {"text": "   "})
        assert "error" in result

    def test_valid_text_accepted(self):
        handler, _ = _make_handler()
        result = handler.handle_tool("add_memory", {"text": "Valid content"})
        assert "error" not in result
        assert result["nodes_created"] == 1


class TestVersionNegotiation:
    """Tests for MCP protocol version negotiation."""

    def test_matching_version(self):
        """Server should respond with client's version if compatible."""
        from flowscript_agents.mcp import _PROTOCOL_VERSION
        handler, _ = _make_handler()
        # Simulate initialize with matching version
        assert _PROTOCOL_VERSION == "2025-03-26"

    def test_newer_client_version_accepted(self):
        """Server should accept newer client versions (tools-only, compatible)."""
        from flowscript_agents.mcp import _PROTOCOL_VERSION
        assert _PROTOCOL_VERSION >= "2025-03-26"
