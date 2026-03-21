"""Shared test fixtures for FlowScript agents tests."""

import math

import pytest


class MockEmbeddings:
    """Deterministic mock embedder for testing.

    Generates embeddings based on word content:
    - Each unique word maps to a dimension
    - Embedding is 1.0 for words present, 0.0 for absent
    - Normalized to unit length
    """

    def __init__(self, dims: int = 8) -> None:
        self._dims = dims
        self._word_map: dict[str, int] = {}
        self._next_dim = 0

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            vec = [0.0] * self._dims
            words = text.lower().split()
            for word in words:
                if word not in self._word_map:
                    if self._next_dim < self._dims:
                        self._word_map[word] = self._next_dim
                        self._next_dim += 1
                idx = self._word_map.get(word)
                if idx is not None and idx < self._dims:
                    vec[idx] = 1.0
            mag = math.sqrt(sum(x * x for x in vec))
            if mag > 0:
                vec = [x / mag for x in vec]
            results.append(vec)
        return results


@pytest.fixture
def mock_embeddings():
    """Fixture providing a MockEmbeddings instance with 16 dims."""
    return MockEmbeddings(dims=16)
