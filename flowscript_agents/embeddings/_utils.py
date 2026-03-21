"""Shared utilities for the embeddings package.

Extracted to avoid duplication between extract.py and consolidate.py
(which cannot import from each other due to circular dependency risk).
"""

from __future__ import annotations

import re


def strip_llm_wrapping(text: str) -> str:
    """Strip common LLM output wrapping that interferes with JSON parsing.

    Handles:
    - <think>...</think> tags (DeepSeek, reasoning models)
    - Unclosed <think> tags (truncated streaming responses)
    - Markdown code fences (```json ... ```)
    - Leading/trailing whitespace
    """
    text = text.strip()

    # Strip closed <think>...</think> blocks (non-greedy, handles multiple)
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

    # Strip unclosed <think> tags (truncated stream — no closing tag)
    # Only fires if a <think> survived the first regex (i.e., no matching </think>)
    if "<think>" in text:
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)

    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    return text
