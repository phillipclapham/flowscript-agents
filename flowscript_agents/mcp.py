"""
FlowScript Unified Memory MCP Server.

Minimal MCP server (JSON-RPC over stdio) that wraps UnifiedMemory.
No external MCP SDK required — implements the protocol directly.

Zero-config quick start (auto-detects OPENAI_API_KEY):
    python -m flowscript_agents.mcp --memory ./agent.json
    # or, if installed via pip:
    flowscript-mcp --memory ./agent.json

Full config:
    python -m flowscript_agents.mcp --memory ./agent.json \\
        --embedder openai --llm-model gpt-4o-mini

Configure in your project's .mcp.json (project-level, shareable):
{
  "mcpServers": {
    "flowscript": {
      "type": "stdio",
      "command": "python3",
      "args": ["-m", "flowscript_agents.mcp", "--memory", "./agent-memory.json"],
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}

Or in ~/.claude.json for global (all projects) configuration.

When OPENAI_API_KEY is set, the server auto-configures:
- OpenAI embeddings (text-embedding-3-small) for vector search
- LLM extraction (gpt-4o-mini) for typed reasoning extraction
- Consolidation (gpt-4o-mini) for memory management (UPDATE/RELATE/RESOLVE)

Tools exposed:
- search_memory: Unified search (vector + keyword + temporal)
- add_memory: Auto-extract reasoning from text with consolidation
- get_context: Get formatted memory for prompt injection
- query_tensions: Find all tensions/tradeoffs in memory
- query_blocked: Find all blocked items with impact analysis
- query_why: Trace causal chain for a node
- query_alternatives: Reconstruct decision from options
- memory_stats: Get memory statistics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

from .memory import Memory
from .unified import UnifiedMemory
from .embeddings.providers import EmbeddingProvider
from .embeddings.consolidate import ConsolidationProvider


# =============================================================================
# MCP Protocol (JSON-RPC over stdio)
# =============================================================================

def _log(msg: str) -> None:
    """Log to stderr (stdout is reserved for JSON-RPC protocol)."""
    sys.stderr.write(f"[flowscript] {msg}\n")
    sys.stderr.flush()


_PROTOCOL_VERSION = "2025-03-26"
_SERVER_NAME = "flowscript-agents"
_SERVER_VERSION = "0.2.0"


def _jsonrpc_response(id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _jsonrpc_error(id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


# =============================================================================
# Tool definitions
# =============================================================================

TOOLS = [
    {
        "name": "search_memory",
        "description": (
            "Search agent memory using unified ranking (vector similarity + keyword "
            "matching + temporal intelligence). Call this to recall prior context before "
            "making decisions, or whenever the conversation touches topics that may "
            "have prior reasoning context. "
            "Use mode='vector' for pure semantic search, 'keyword' for exact matching, "
            "or 'unified' (default) for combined ranking."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "top_k": {"type": "integer", "description": "Max results (default 10)", "default": 10},
                "mode": {
                    "type": "string",
                    "enum": ["unified", "vector", "keyword"],
                    "description": "Search mode: unified (default), vector (semantic only), keyword (exact only)",
                    "default": "unified",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "add_memory",
        "description": (
            "Add information to agent memory. Call this when important decisions are "
            "made, architectural tradeoffs are discussed, blockers are identified, "
            "causal relationships are established, or any reasoning worth preserving "
            "occurs in conversation. Include full context — the extraction layer "
            "automatically identifies and types the reasoning structures (decisions "
            "with rationale, tensions with axes, causal chains, blockers with reasons). "
            "Do NOT store routine code changes, transient debugging steps, or "
            "information already tracked in git. "
            "Returns extraction results including node count and dedup info."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to add to memory — include full reasoning context"},
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata to attach to created nodes",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "get_context",
        "description": (
            "Get formatted memory content for prompt context. Call this at the start "
            "of sessions to load relevant memory, or periodically during long sessions "
            "to check what reasoning has been preserved. Returns nodes sorted by "
            "tier and frequency, with tier labels. Use max_tokens to control size."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens for context (default 4000)",
                    "default": 4000,
                },
            },
        },
    },
    {
        "name": "query_tensions",
        "description": (
            "Find all tensions and tradeoffs in memory. Call this when evaluating "
            "tradeoffs, before making decisions that might conflict with prior choices, "
            "or when the user asks about competing concerns. Returns tension pairs "
            "grouped by axis."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "string",
                    "enum": ["axis", "node", "flat"],
                    "description": "How to group tensions (default: axis)",
                    "default": "axis",
                },
            },
        },
    },
    {
        "name": "query_blocked",
        "description": (
            "Find all blocked items in memory with impact analysis. Call this when "
            "planning work, when progress stalls, or to check what's waiting on "
            "external dependencies. Returns blockers sorted by impact score "
            "(downstream effects), with reason, duration, and transitive causes."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "query_why",
        "description": (
            "Trace the causal chain for a specific memory node. Call this when "
            "the user asks 'why did we decide X' or when you need to understand "
            "the reasoning behind a prior decision. Returns root cause, intermediate "
            "steps, and the full chain. Search by content to find the node, or "
            "provide a node_id if known from a prior search."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Node ID to trace"},
                "content": {"type": "string", "description": "Search for node by content (alternative to node_id)"},
            },
        },
    },
    {
        "name": "query_alternatives",
        "description": (
            "Reconstruct a decision from its alternatives. Call this when revisiting "
            "decisions or when the user asks what options were considered. Shows all "
            "options, which was chosen, rejection rationale, and consequences. "
            "Search by content to find the question, or provide a question_id if known."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question_id": {"type": "string", "description": "Question node ID"},
                "content": {"type": "string", "description": "Search for question by content (alternative to question_id)"},
            },
        },
    },
    {
        "name": "query_what_if",
        "description": (
            "Forward impact analysis: what happens if a node changes? Call this "
            "when considering changes to understand downstream consequences before "
            "committing. Traces direct and indirect effects, finds tensions in "
            "the impact zone."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Node ID to analyze"},
                "content": {"type": "string", "description": "Search for node by content (alternative to node_id)"},
            },
        },
    },
    {
        "name": "remove_memory",
        "description": (
            "Remove a specific memory node by ID. Call this to correct mistakes — "
            "if something was stored incorrectly, a decision was reversed, or "
            "information is no longer relevant. Also removes associated relationships "
            "and states. Use search_memory first to find the node_id to remove."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "ID of the node to remove"},
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "session_wrap",
        "description": (
            "Run memory lifecycle maintenance: prune dormant nodes to audit trail, "
            "save to disk. Call this at the end of a work session to keep memory "
            "healthy. Dormant nodes (not accessed recently) are archived, not deleted."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "memory_stats",
        "description": (
            "Get memory statistics: node count, tier distribution, garden health, "
            "embedding status. Call this to understand the current state of memory."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
]


# =============================================================================
# Tool handlers
# =============================================================================


class MCPHandler:
    """Handles MCP tool calls against a UnifiedMemory instance."""

    def __init__(self, umem: UnifiedMemory) -> None:
        self._umem = umem

    def handle_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "search_memory": self._search_memory,
            "add_memory": self._add_memory,
            "get_context": self._get_context,
            "query_tensions": self._query_tensions,
            "query_blocked": self._query_blocked,
            "query_why": self._query_why,
            "query_what_if": self._query_what_if,
            "query_alternatives": self._query_alternatives,
            "remove_memory": self._remove_memory,
            "session_wrap": self._session_wrap,
            "memory_stats": self._memory_stats,
        }
        handler = handlers.get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return handler(args)
        except (KeyError, ValueError) as e:
            return {"error": f"{type(e).__name__}: {e}"}
        except Exception as e:
            _log(f"Tool error in {name}: {type(e).__name__}: {e}")
            return {"error": f"Internal error in {name} — check server logs"}

    def _search_memory(self, args: dict) -> dict:
        query = args.get("query", "")
        top_k = args.get("top_k", 10)
        mode = args.get("mode", "unified")

        if mode == "vector":
            results = self._umem.vector_search(query, top_k=top_k)
            return {
                "results": [
                    {
                        "content": r.content,
                        "score": round(r.score, 4),
                        "node_id": r.node_id,
                        "type": r.node_type,
                        "tier": r.tier,
                        "frequency": r.frequency,
                    }
                    for r in results
                ],
                "mode": "vector",
                "count": len(results),
            }
        elif mode == "keyword":
            results = self._umem.search(
                query, top_k=top_k,
                vector_weight=0.0, keyword_weight=0.8, temporal_weight=0.2,
            )
        else:  # unified
            results = self._umem.search(query, top_k=top_k)

        return {
            "results": [
                {
                    "content": r.content,
                    "score": round(r.combined_score, 4),
                    "node_id": r.node_id,
                    "type": r.node_type,
                    "tier": r.tier,
                    "frequency": r.frequency,
                    "sources": r.sources,
                }
                for r in results
            ],
            "mode": mode,
            "count": len(results),
        }

    def _add_memory(self, args: dict) -> dict:
        # Accept both "text" and "content" — LLMs frequently use "content"
        text = args.get("text", "") or args.get("content", "")
        if not text or not text.strip():
            return {"error": "text is required and must not be empty"}
        metadata = args.get("metadata")
        result = self._umem.add(text, metadata=metadata, actor="agent")
        resp: dict[str, Any] = {
            "nodes_created": result.nodes_created,
            "nodes_deduplicated": result.nodes_deduplicated,
            "relationships_created": result.relationships_created,
            "states_created": result.states_created,
            "node_ids": result.node_ids,
        }
        # Hint when running without LLM (raw storage only)
        if self._umem.extractor is None and result.nodes_created == 1:
            resp["note"] = (
                "Running without LLM — stored as raw text (no typed extraction). "
                "Install openai and set OPENAI_API_KEY for automatic reasoning extraction."
            )
        return resp

    def _get_context(self, args: dict) -> dict:
        max_tokens = args.get("max_tokens", 4000)
        context = self._umem.get_context(max_tokens=max_tokens)
        return {"context": context, "nodes": self._umem.size}

    def _query_tensions(self, args: dict) -> dict:
        group_by = args.get("group_by", "axis")
        result = self._umem.memory.query.tensions(group_by=group_by)
        # Serialize the result
        return _serialize_query_result(result)

    def _query_blocked(self, args: dict) -> dict:
        result = self._umem.memory.query.blocked()
        return _serialize_query_result(result)

    def _query_why(self, args: dict) -> dict:
        node_id = args.get("node_id")
        content = args.get("content")
        if not node_id and content:
            refs = self._umem.memory.find_nodes(content)
            if refs:
                node_id = refs[0].id
        if not node_id:
            return {"error": "No node found. Provide node_id or searchable content."}
        result = self._umem.memory.query.why(node_id)
        return _serialize_query_result(result)

    def _query_what_if(self, args: dict) -> dict:
        node_id = args.get("node_id")
        content = args.get("content")
        if not node_id and content:
            refs = self._umem.memory.find_nodes(content)
            if refs:
                node_id = refs[0].id
        if not node_id:
            return {"error": "No node found. Provide node_id or searchable content."}
        result = self._umem.memory.query.what_if(node_id)
        return _serialize_query_result(result)

    def _query_alternatives(self, args: dict) -> dict:
        question_id = args.get("question_id")
        content = args.get("content")
        if not question_id and content:
            refs = self._umem.memory.find_nodes(content)
            if refs:
                question_id = refs[0].id
        if not question_id:
            return {"error": "No question found. Provide question_id or searchable content."}
        result = self._umem.memory.query.alternatives(question_id)
        return _serialize_query_result(result)

    def _remove_memory(self, args: dict) -> dict:
        node_id = args.get("node_id", "")
        if not node_id:
            return {"error": "node_id is required"}
        # Remove from graph first, then clean up vector index
        removed = self._umem.memory.remove_node(node_id)
        if removed and self._umem.vector_index:
            self._umem.vector_index.remove_node(node_id)
        return {"removed": removed, "node_id": node_id}

    def _session_wrap(self, args: dict) -> dict:
        result = self._umem.memory.session_wrap()
        return {
            "nodes_before": result.nodes_before,
            "nodes_after": result.nodes_after,
            "nodes_pruned": result.pruned.count,
            "tiers_after": result.tiers_after,
        }

    def _memory_stats(self, args: dict) -> dict:
        mem = self._umem.memory
        tiers = mem.count_tiers()
        garden = mem.garden()
        stats = {
            "total_nodes": mem.size,
            "tiers": tiers,
            "garden": {
                "growing": len(garden.growing),
                "resting": len(garden.resting),
                "dormant": len(garden.dormant),
            },
            "relationships": mem.relationship_count,
            "states": mem.state_count,
        }
        if self._umem.vector_index:
            stats["embeddings"] = {
                "indexed": self._umem.vector_index.indexed_count,
                "provider": repr(self._umem.vector_index._provider),
            }
        return stats


def _serialize_query_result(result: Any, _seen: set | None = None) -> dict:
    """Best-effort serialization of query result dataclasses."""
    if _seen is None:
        _seen = set()
    obj_id = id(result)
    if obj_id in _seen:
        return {"_circular": str(type(result).__name__)}
    _seen.add(obj_id)
    if hasattr(result, "__dict__"):
        d = {}
        for k, v in result.__dict__.items():
            if k.startswith("_"):
                continue
            d[k] = _serialize_value(v, _seen)
        return d
    return {"result": str(result)}


def _serialize_value(v: Any, _seen: set | None = None) -> Any:
    if _seen is None:
        _seen = set()
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_serialize_value(x, _seen) for x in v]
    if isinstance(v, dict):
        return {str(k): _serialize_value(val, _seen) for k, val in v.items()}
    if hasattr(v, "__dict__"):
        return _serialize_query_result(v, _seen)
    return str(v)


# =============================================================================
# MCP Server loop
# =============================================================================


def _create_embedder(provider: str, **kwargs: Any) -> Optional[EmbeddingProvider]:
    """Create an embedding provider by name."""
    if provider == "openai":
        from .embeddings.providers import OpenAIEmbeddings
        return OpenAIEmbeddings(**kwargs)
    elif provider == "sentence-transformers":
        from .embeddings.providers import SentenceTransformerEmbeddings
        return SentenceTransformerEmbeddings(**kwargs)
    elif provider == "ollama":
        from .embeddings.providers import OllamaEmbeddings
        return OllamaEmbeddings(**kwargs)
    return None


class _OpenAIConsolidationProvider:
    """OpenAI consolidation provider using tool calling for memory management."""

    def __init__(self, model: str = "gpt-4o-mini", client: Any = None) -> None:
        if client is not None:
            self._client = client
        else:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "Auto-configuration requires the openai package. "
                    "Install with: pip install openai"
                )
            self._client = openai.OpenAI()
        self._model = model

    def tool_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice="required",
            temperature=0.1,
        )
        results = []
        for tc in resp.choices[0].message.tool_calls or []:
            results.append({
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            })
        return results


class _AnthropicConsolidationProvider:
    """Anthropic consolidation provider using Claude tool calling.

    Translates between the OpenAI-format ConsolidationProvider protocol
    and Anthropic's native tool calling API.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", client: Any = None) -> None:
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "Anthropic auto-configuration requires the anthropic package. "
                    "Install with: pip install anthropic"
                )
            self._client = anthropic.Anthropic()
        self._model = model

    def tool_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Translate OpenAI-format tools to Anthropic format
        anthropic_tools = []
        for t in tools:
            fn = t.get("function", t)  # handle both wrapped and flat formats
            anthropic_tools.append({
                "name": fn.get("name", t.get("name")),
                "description": fn.get("description", t.get("description", "")),
                "input_schema": fn.get("parameters", t.get("inputSchema", t.get("parameters", {}))),
            })

        # Separate system message from user messages
        system_msg = None
        user_messages = []
        for m in messages:
            if m.get("role") == "system":
                system_msg = m["content"]
            else:
                user_messages.append({"role": m["role"], "content": m["content"]})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "tools": anthropic_tools,
            "tool_choice": {"type": "any"},
            "messages": user_messages,
        }
        if system_msg:
            kwargs["system"] = system_msg

        resp = self._client.messages.create(**kwargs)

        results = []
        for block in resp.content:
            if hasattr(block, "type") and block.type == "tool_use":
                results.append({
                    "name": block.name,
                    "arguments": block.input,
                })
        return results


def _auto_configure_anthropic(
    model: str = "claude-haiku-4-5-20251001",
) -> tuple[None, Any, _AnthropicConsolidationProvider]:
    """Auto-configure Anthropic extraction LLM + consolidation provider.

    Returns (None, llm_fn, consolidation_provider).
    Embedder is None — Anthropic has no embeddings API.
    Uses keyword-only search (still functional, just no vector similarity).
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "Anthropic auto-configuration requires the anthropic package. "
            "Install with: pip install anthropic"
        )

    client = anthropic.Anthropic()

    def llm_extract(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text if resp.content else ""

    consolidation = _AnthropicConsolidationProvider(model=model, client=client)

    return None, llm_extract, consolidation


def _auto_configure_openai(
    model: str = "gpt-4o-mini",
    embedding_model: str | None = None,
) -> tuple[EmbeddingProvider, Any, _OpenAIConsolidationProvider]:
    """Auto-configure OpenAI embedder + extraction LLM + consolidation provider.

    Returns (embedder, llm_fn, consolidation_provider).
    Requires OPENAI_API_KEY in environment and the openai package installed.
    Uses a single shared OpenAI client for extraction and consolidation.
    """
    try:
        import openai
    except ImportError:
        raise ImportError(
            "Auto-configuration requires the openai package. "
            "Install with: pip install openai"
        )

    from .embeddings.providers import OpenAIEmbeddings

    client = openai.OpenAI()

    emb_kwargs: dict[str, Any] = {}
    if embedding_model:
        emb_kwargs["model"] = embedding_model
    embedder = OpenAIEmbeddings(**emb_kwargs)

    def llm_extract(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""

    consolidation = _OpenAIConsolidationProvider(model=model, client=client)

    return embedder, llm_extract, consolidation


def run_server(
    memory_path: str,
    embedder: Optional[EmbeddingProvider] = None,
    llm: Optional[Any] = None,
    consolidation_provider: Optional[Any] = None,
) -> None:
    """Run the MCP server over stdio."""
    umem = UnifiedMemory(
        file_path=memory_path,
        embedder=embedder,
        llm=llm,
        consolidation_provider=consolidation_provider,
    )
    # Start session tracking — enables touch deduplication and temporal
    # intelligence across the lifetime of this MCP server instance.
    umem.memory.session_start()
    handler = MCPHandler(umem)

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_id = msg.get("id")
            method = msg.get("method", "")
            params = msg.get("params", {})

            if method == "initialize":
                # Version negotiation: respond with client's version if we
                # support it, otherwise our latest. Our tools-only server is
                # compatible with all versions from 2024-11-05 onward.
                client_version = params.get("protocolVersion", _PROTOCOL_VERSION)
                resp = _jsonrpc_response(msg_id, {
                    "protocolVersion": client_version if client_version >= _PROTOCOL_VERSION else _PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": _SERVER_NAME,
                        "version": _SERVER_VERSION,
                    },
                })
            elif method == "notifications/initialized":
                continue  # notification, no response
            elif method == "tools/list":
                resp = _jsonrpc_response(msg_id, {"tools": TOOLS})
            elif method == "resources/list":
                resp = _jsonrpc_response(msg_id, {"resources": []})
            elif method == "prompts/list":
                resp = _jsonrpc_response(msg_id, {"prompts": []})
            elif method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments", {})
                result = handler.handle_tool(tool_name, tool_args)
                # Save after modifications
                if tool_name in ("add_memory", "remove_memory", "session_wrap"):
                    try:
                        umem.save()
                    except ValueError:
                        pass  # in-memory mode, no file path — expected
                    except OSError as e:
                        _log(f"Warning: save failed: {e}")
                is_error = "error" in result
                call_result: dict[str, Any] = {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                }
                if is_error:
                    call_result["isError"] = True
                resp = _jsonrpc_response(msg_id, call_result)
            elif method == "ping":
                resp = _jsonrpc_response(msg_id, {})
            else:
                resp = _jsonrpc_error(msg_id, -32601, f"Method not found: {method}")

            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    finally:
        # Safety net: save state when stdin closes (Claude Code exits).
        # Use save() not close() — don't prune on unclean shutdown.
        try:
            umem.save()
        except Exception:
            pass


# =============================================================================
# CLI entry point
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlowScript Unified Memory MCP Server",
        epilog=(
            "Zero-config: if OPENAI_API_KEY is set and no --embedder is specified, "
            "the server auto-configures OpenAI embeddings, extraction, and consolidation. "
            "Just run: python -m flowscript_agents.mcp --memory ./agent.json"
        ),
    )
    parser.add_argument(
        "--memory", required=True,
        help="Path to memory JSON file (created if doesn't exist)",
    )
    parser.add_argument(
        "--embedder", choices=["openai", "sentence-transformers", "ollama"],
        help="Embedding provider (overrides auto-detection)",
    )
    parser.add_argument(
        "--embedding-model",
        help="Embedding model name (provider-specific, default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="LLM model for extraction and consolidation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="Disable auto-configuration from OPENAI_API_KEY",
    )
    args = parser.parse_args()

    embedder = None
    llm = None
    consolidation = None

    if args.embedder:
        # Explicit embedder specified — use it, no auto-config
        kwargs = {}
        if args.embedding_model:
            if args.embedder == "openai":
                kwargs["model"] = args.embedding_model
            elif args.embedder == "sentence-transformers":
                kwargs["model_name"] = args.embedding_model
            elif args.embedder == "ollama":
                kwargs["model"] = args.embedding_model
        embedder = _create_embedder(args.embedder, **kwargs)
    elif not args.no_auto and os.environ.get("OPENAI_API_KEY"):
        # Auto-configure from OPENAI_API_KEY (full stack: embeddings + extraction + consolidation)
        try:
            embedder, llm, consolidation = _auto_configure_openai(
                model=args.llm_model,
                embedding_model=args.embedding_model,
            )
            _log("Auto-configured: OpenAI embeddings + extraction + consolidation "
                 f"(model: {args.llm_model})")
        except ImportError as e:
            _log(f"OpenAI auto-configuration skipped: {e}")
        except Exception as e:
            _log(f"OpenAI auto-configuration failed: {e}")
    elif not args.no_auto and os.environ.get("ANTHROPIC_API_KEY"):
        # Auto-configure from ANTHROPIC_API_KEY (extraction + consolidation, no embeddings)
        try:
            embedder, llm, consolidation = _auto_configure_anthropic(
                model=args.llm_model if args.llm_model != "gpt-4o-mini" else "claude-haiku-4-5-20251001",
            )
            _log("Auto-configured: Anthropic extraction + consolidation "
                 f"(no embeddings — Anthropic has no embedding API, using keyword search)")
        except ImportError as e:
            _log(f"Anthropic auto-configuration skipped: {e}")
        except Exception as e:
            _log(f"Anthropic auto-configuration failed: {e}")

    # Validate API key at startup with a lightweight probe
    if embedder is not None:
        try:
            embedder.embed(["startup validation"])
            _log("Embedding provider validated successfully")
        except Exception as e:
            _log(f"ERROR: Embedding provider failed validation: {e}")
            _log("Check your API key and network connection. "
                 "Falling back to keyword-only search.")
            embedder = None

    # Validate LLM at startup (lightweight probe — extraction, not full call)
    if llm is not None:
        try:
            test_result = llm("Respond with OK.")
            if test_result:
                _log("LLM extraction provider validated successfully")
        except Exception as e:
            _log(f"ERROR: LLM extraction provider failed validation: {e}")
            _log("Check your API key and network connection. "
                 "Falling back to raw text storage (no typed extraction).")
            llm = None
            consolidation = None

    # Warn about degraded mode so developers know what they're getting
    if embedder is None:
        _log("Warning: No embedding provider configured — vector search disabled. "
             "Set OPENAI_API_KEY for full auto-configuration, or use --embedder.")
    if llm is None:
        _log("Warning: No LLM configured — add_memory stores raw text only "
             "(no typed extraction). Set OPENAI_API_KEY or ANTHROPIC_API_KEY "
             "for auto-configuration.")

    run_server(
        memory_path=args.memory,
        embedder=embedder,
        llm=llm,
        consolidation_provider=consolidation,
    )


if __name__ == "__main__":
    main()
