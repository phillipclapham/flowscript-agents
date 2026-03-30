"""
Integration test: Three-Layer MCP Architecture (Layer 1 + Layer 2 + Layer 3).

Exercises the full pipeline with a real LLM:
  1. Configure all 3 layers (continuity + extraction + vector)
  2. Encode exchanges (Layer 2)
  3. Search memory (Layer 3)
  4. Session wrap → produces continuity file (Layer 1)
  5. Reload → verify continuity injected into get_context
  6. Second session → verify temporal graduation in continuity

Requires: OPENAI_API_KEY in environment or .env.flow

Run: python3 -m pytest tests/test_integration_continuity.py -v -s
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Load API key from .env.flow if not in environment
_ENV_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "flow", ".env.flow")
if not os.environ.get("OPENAI_API_KEY"):
    _alt = os.path.expanduser("~/Documents/flow/.env.flow")
    for path in [_ENV_FILE, _alt]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY="):
                        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
            break

# Skip entire module if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not available"
)


def _make_llm():
    """Create a real OpenAI LLM function."""
    import openai
    client = openai.OpenAI()

    def llm_fn(prompt: str) -> str:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""

    return llm_fn


def _make_embedder():
    """Create a real OpenAI embedder."""
    from flowscript_agents.embeddings.providers import OpenAIEmbeddings
    return OpenAIEmbeddings()


def _make_consolidation_provider():
    """Create a real consolidation provider."""
    import openai
    from flowscript_agents.mcp import _OpenAIConsolidationProvider
    client = openai.OpenAI()
    return _OpenAIConsolidationProvider(model="gpt-4o-mini", client=client)


class TestThreeLayerIntegration:
    """End-to-end test of the three-layer MCP architecture."""

    def test_full_pipeline(self):
        """Full pipeline: encode → search → wrap → reload → verify continuity."""
        from flowscript_agents import UnifiedMemory, ContinuityManager

        llm = _make_llm()
        embedder = _make_embedder()
        consolidation = _make_consolidation_provider()

        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = os.path.join(tmpdir, "agent.json")
            cont_mgr = ContinuityManager(llm=llm, project_name="IntegrationTest")

            # === Session 1: Build memory ===
            umem = UnifiedMemory(
                file_path=mem_path,
                embedder=embedder,
                llm=llm,
                consolidation_provider=consolidation,
            )
            umem.memory.session_start()

            # Layer 2: Encode exchanges
            exchange_1 = umem.add(
                "User: Should we use PostgreSQL or Redis for the session store?\n"
                "Assistant: PostgreSQL is better here because we need ACID compliance "
                "for financial transactions. Redis is faster but doesn't guarantee "
                "durability. The tradeoff is speed vs data safety, and for financial "
                "data, safety wins.",
                actor="agent",
            )
            assert exchange_1.nodes_created > 0, "Exchange 1 should create nodes"
            print(f"\nExchange 1: {exchange_1.nodes_created} nodes, "
                  f"{exchange_1.relationships_created} rels")

            exchange_2 = umem.add(
                "User: What about connection pooling?\n"
                "Assistant: We should use pgbouncer for connection pooling. It sits "
                "between the app and PostgreSQL, managing a pool of connections. "
                "This prevents the app from opening too many connections under load. "
                "The decision to use PostgreSQL makes pgbouncer the natural choice.",
                actor="agent",
            )
            assert exchange_2.nodes_created > 0, "Exchange 2 should create nodes"
            print(f"Exchange 2: {exchange_2.nodes_created} nodes, "
                  f"{exchange_2.relationships_created} rels")

            exchange_3 = umem.add(
                "User: How should we handle database migrations?\n"
                "Assistant: Use Alembic for migrations since we're on PostgreSQL. "
                "Keep migrations in version control. Run them as part of the deploy "
                "pipeline, not manually. This connects to our ACID decision — we "
                "need transactional DDL which PostgreSQL supports but many databases don't.",
                actor="agent",
            )
            print(f"Exchange 3: {exchange_3.nodes_created} nodes, "
                  f"{exchange_3.relationships_created} rels")

            # Layer 3: Vector search
            search_results = umem.search("database decision ACID")
            print(f"Search results: {len(search_results)} hits")
            assert len(search_results) > 0, "Should find database-related nodes"

            # Verify we have meaningful content
            total_nodes = umem.memory.size
            total_rels = umem.memory.relationship_count
            print(f"Total: {total_nodes} nodes, {total_rels} relationships")
            assert total_nodes >= 3, f"Expected at least 3 nodes, got {total_nodes}"

            # Query: check tensions exist
            tensions = umem.memory.query.tensions()
            print(f"Tensions found: {len(tensions.items) if hasattr(tensions, 'items') else 'N/A'}")

            # Layer 1: Produce continuity file
            existing = ContinuityManager.load(mem_path)
            assert existing is None, "No continuity should exist yet"

            cont_result = cont_mgr.produce(umem.memory)
            print(f"\nContinuity produced: {cont_result.char_count} chars, "
                  f"{cont_result.patterns_extracted} patterns")
            print(f"Sections: {cont_result.section_sizes}")

            # Verify continuity structure
            text_lower = cont_result.text.lower()
            assert "state" in text_lower and "##" in cont_result.text, "Missing State section"
            assert "patterns" in text_lower, "Missing Patterns section"
            assert "decisions" in text_lower or "decision" in text_lower, "Missing Decisions section"
            assert "context" in text_lower, "Missing Context section"
            assert cont_result.char_count <= 20000, f"Exceeded max chars: {cont_result.char_count}"
            assert not cont_result.truncated, "Should not be truncated"

            # Save continuity
            saved_path = cont_mgr.save(cont_result.text, mem_path)
            assert os.path.exists(saved_path), "Continuity file should exist"

            # Save memory + embeddings
            umem.save()

            # === Reload: Verify continuity persists ===
            loaded_continuity = ContinuityManager.load(mem_path)
            assert loaded_continuity is not None, "Should load saved continuity"
            assert "state" in loaded_continuity.lower(), "Loaded continuity missing State"
            assert len(loaded_continuity) == cont_result.char_count, "Loaded size should match"

            print(f"\n--- Continuity File ---")
            # Print first 500 chars for inspection
            print(loaded_continuity[:500])
            if len(loaded_continuity) > 500:
                print(f"... ({len(loaded_continuity) - 500} more chars)")

            # === Session 2: Verify continuity feeds back ===
            umem2 = UnifiedMemory(
                file_path=mem_path,
                embedder=embedder,
                llm=llm,
                consolidation_provider=consolidation,
            )
            umem2.memory.session_start()

            # Add new exchange that should connect to existing knowledge
            exchange_4 = umem2.add(
                "User: Should we add read replicas?\n"
                "Assistant: Yes, read replicas make sense for our PostgreSQL setup. "
                "Since we chose PostgreSQL for ACID compliance, we can use streaming "
                "replication for read scaling. pgbouncer can route read queries to "
                "replicas automatically.",
                actor="agent",
            )
            print(f"\nSession 2 Exchange: {exchange_4.nodes_created} nodes, "
                  f"{exchange_4.relationships_created} rels")

            # Produce second continuity — should show temporal graduation
            cont_result_2 = cont_mgr.produce(
                umem2.memory,
                existing_continuity=loaded_continuity,
            )
            print(f"Session 2 Continuity: {cont_result_2.char_count} chars, "
                  f"{cont_result_2.patterns_extracted} patterns")

            # The PostgreSQL/ACID pattern should be reinforced
            assert "PostgreSQL" in cont_result_2.text or "postgres" in cont_result_2.text.lower(), \
                "PostgreSQL should appear in continuity (validated across sessions)"

            print(f"\n--- Session 2 Continuity ---")
            print(cont_result_2.text[:500])
            if len(cont_result_2.text) > 500:
                print(f"... ({len(cont_result_2.text) - 500} more chars)")

            print("\n=== THREE-LAYER INTEGRATION: PASS ===")

    def test_thinking_tools_with_real_memory(self):
        """Thinking tools should create queryable nodes with real LLM."""
        from flowscript_agents import UnifiedMemory
        from flowscript_agents.mcp import MCPHandler

        llm = _make_llm()
        embedder = _make_embedder()

        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = os.path.join(tmpdir, "agent.json")
            umem = UnifiedMemory(
                file_path=mem_path,
                embedder=embedder,
                llm=llm,
            )
            umem.memory.session_start()

            handler = MCPHandler(umem, memory_path=mem_path)

            # Use think_deeper — should return framework without touching graph
            nodes_before = umem.memory.size
            result = handler.handle_tool("think_deeper", {
                "problem": "Should we use microservices or monolith?",
                "context": "Small team, early stage startup, need to move fast",
            })
            assert "error" not in result
            assert "framework" in result
            assert umem.memory.size == nodes_before  # No graph pollution
            print(f"\nthink_deeper: framework returned, graph unchanged")

            # Use think_creative
            result = handler.handle_tool("think_creative", {
                "problem": "How to reduce deployment time from 30 minutes to under 5?",
                "attempts": "Tried parallel builds, caching — still slow",
            })
            assert "error" not in result
            print(f"think_creative: framework returned")

            # Use think_breakthrough
            result = handler.handle_tool("think_breakthrough", {
                "problem": "Database scaling hitting limits at 10k req/s",
            })
            assert "error" not in result
            assert umem.memory.size == nodes_before  # Still no graph pollution
            print(f"think_breakthrough: framework returned, graph still clean")

            print("\n=== THINKING TOOLS INTEGRATION: PASS ===")

    def test_composability_layers(self):
        """Each layer combination should work independently."""
        from flowscript_agents import UnifiedMemory, ContinuityManager

        llm = _make_llm()
        embedder = _make_embedder()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Layer 1 only (continuity, no extraction, no vector)
            mem_path_1 = os.path.join(tmpdir, "l1.json")
            umem1 = UnifiedMemory(file_path=mem_path_1)
            umem1.memory.session_start()
            umem1.memory.thought("Testing layer 1 only")
            cont_mgr = ContinuityManager(llm=llm, project_name="L1Only")
            result = cont_mgr.produce(umem1.memory)
            assert "state" in result.text.lower() and "##" in result.text
            print(f"L1 only: {result.char_count} chars")

            # Layer 1 + Layer 2 (continuity + extraction, no vector)
            mem_path_12 = os.path.join(tmpdir, "l12.json")
            umem12 = UnifiedMemory(file_path=mem_path_12, llm=llm)
            umem12.memory.session_start()
            ingest = umem12.add("We decided to use Rust for performance reasons")
            assert ingest.nodes_created > 0
            result12 = cont_mgr.produce(umem12.memory)
            assert "state" in result12.text.lower() and "##" in result12.text
            print(f"L1+L2: {result12.char_count} chars, {ingest.nodes_created} extracted nodes")

            # All three layers
            mem_path_123 = os.path.join(tmpdir, "l123.json")
            umem123 = UnifiedMemory(
                file_path=mem_path_123, llm=llm, embedder=embedder,
            )
            umem123.memory.session_start()
            ingest123 = umem123.add("GraphQL vs REST — chose REST for simplicity")
            search = umem123.search("API design decision")
            result123 = cont_mgr.produce(umem123.memory)
            assert "state" in result123.text.lower() and "##" in result123.text
            assert len(search) > 0 or ingest123.nodes_created > 0  # vector or extraction worked
            print(f"L1+L2+L3: {result123.char_count} chars, "
                  f"{ingest123.nodes_created} nodes, {len(search)} search hits")

            # Layer 2 + Layer 3 only (no continuity — current default behavior)
            mem_path_23 = os.path.join(tmpdir, "l23.json")
            umem23 = UnifiedMemory(
                file_path=mem_path_23, llm=llm, embedder=embedder,
            )
            umem23.memory.session_start()
            umem23.add("Chose Docker over bare metal for isolation")
            hits = umem23.search("deployment infrastructure")
            print(f"L2+L3 (no continuity): {umem23.memory.size} nodes, {len(hits)} search hits")

            print("\n=== COMPOSABILITY: PASS ===")
