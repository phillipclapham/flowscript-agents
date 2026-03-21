"""
FlowScript Embeddings — Vector similarity + auto-extraction for agent memory.

Composable modules that layer ON TOP of existing Memory class.
Memory stays unchanged. These add: embedding providers, vector search,
auto-extraction, and unified search.

Usage:
    from flowscript_agents.embeddings import (
        OpenAIEmbeddings, SentenceTransformerEmbeddings, OllamaEmbeddings,
        VectorIndex, AutoExtract, UnifiedSearch,
    )
    from flowscript_agents import Memory

    mem = Memory.load_or_create("./agent.json")
    index = VectorIndex(mem, OpenAIEmbeddings())
    index.index_all()
    results = index.search("database decisions", top_k=5)
"""

from .providers import (
    EmbeddingProvider,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
    OllamaEmbeddings,
)
from .index import VectorIndex, VectorSearchResult
from .search import UnifiedSearch, UnifiedSearchResult
from .extract import AutoExtract, IngestResult
from .consolidate import (
    ConsolidationEngine,
    ConsolidationProvider,
    ConsolidationAction,
    ConsolidationResult,
)

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddings",
    "SentenceTransformerEmbeddings",
    "OllamaEmbeddings",
    "VectorIndex",
    "VectorSearchResult",
    "UnifiedSearch",
    "UnifiedSearchResult",
    "AutoExtract",
    "IngestResult",
    "ConsolidationEngine",
    "ConsolidationProvider",
    "ConsolidationAction",
    "ConsolidationResult",
]
