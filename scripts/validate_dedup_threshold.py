#!/usr/bin/env python3
"""
Dedup Threshold Validation — empirical testing with real embeddings.

Tests semantic similarity scores for pairs of text at different relationship
levels to find the optimal dedup and candidate thresholds.

Categories tested:
1. EXACT duplicates (same meaning, different words) → should be >0.90
2. RELATED (same topic, additional info) → should be 0.60-0.85
3. UNRELATED (different topics entirely) → should be <0.40
4. CONTRADICTIONS (same topic, opposing views) → should be 0.50-0.80

The sweet spot:
- dedup_threshold = 0.80-0.85 → catches exact dupes, lets related through
- candidate_threshold = 0.45-0.55 → casts wide net for LLM consolidation

Requires: OPENAI_API_KEY in environment (uses text-embedding-3-small)
Usage: python3 scripts/validate_dedup_threshold.py
"""

import os
import sys

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowscript_agents.embeddings.providers import OpenAIEmbeddings
from flowscript_agents.embeddings.index import _cosine_similarity, _normalize


# Test pairs: (text_a, text_b, expected_category)
TEST_PAIRS = [
    # EXACT DUPLICATES — different words, same meaning
    (
        "We chose PostgreSQL for the database due to ACID compliance",
        "PostgreSQL was selected as our database because of its ACID support",
        "EXACT",
    ),
    (
        "The authentication system is blocked waiting for the security audit",
        "Auth is blocked — can't proceed until the security audit completes",
        "EXACT",
    ),
    (
        "Redis is too expensive for our use case",
        "The cost of Redis makes it impractical for what we need",
        "EXACT",
    ),

    # RELATED — same topic, new/additional info (UPDATE candidates)
    (
        "We chose PostgreSQL for the database due to ACID compliance",
        "PostgreSQL's ACID compliance has been validated in our load testing — it handles 10K TPS",
        "RELATED",
    ),
    (
        "Redis is being considered for caching",
        "We've decided to use Redis for caching with a 15-minute TTL",
        "RELATED",
    ),
    (
        "The API needs authentication",
        "JWT tokens with 1-hour expiry were chosen for API authentication",
        "RELATED",
    ),

    # CONTRADICTIONS — same topic, opposing views (RELATE candidates)
    (
        "We chose PostgreSQL for ACID compliance",
        "MySQL would be better — PostgreSQL is overkill for our scale",
        "CONTRADICTION",
    ),
    (
        "Microservices are the right architecture for this project",
        "A monolith would be simpler and more maintainable than microservices here",
        "CONTRADICTION",
    ),
    (
        "We should deploy on Kubernetes",
        "Kubernetes adds too much complexity — simple EC2 instances would suffice",
        "CONTRADICTION",
    ),

    # UNRELATED — completely different topics
    (
        "We chose PostgreSQL for the database due to ACID compliance",
        "The marketing team wants to launch the newsletter by Friday",
        "UNRELATED",
    ),
    (
        "Redis is being considered for caching",
        "The office lease expires in March and we need to negotiate renewal",
        "UNRELATED",
    ),
    (
        "JWT tokens for API authentication",
        "The team lunch is scheduled for Thursday at noon",
        "UNRELATED",
    ),
]


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. This script requires real OpenAI embeddings.")
        print("Usage: OPENAI_API_KEY=sk-... python3 scripts/validate_dedup_threshold.py")
        sys.exit(1)

    embedder = OpenAIEmbeddings(api_key=api_key)
    print("Using OpenAI text-embedding-3-small (1536 dims)\n")

    # Collect all texts for batch embedding
    all_texts = []
    for a, b, _ in TEST_PAIRS:
        all_texts.extend([a, b])

    print(f"Embedding {len(all_texts)} texts...")
    embeddings = embedder.embed(all_texts)
    print(f"Done. Computing similarities...\n")

    # Compute similarities
    results_by_category: dict[str, list[float]] = {
        "EXACT": [], "RELATED": [], "CONTRADICTION": [], "UNRELATED": [],
    }

    print(f"{'Category':<15} {'Similarity':>10}  Text A (truncated) → Text B (truncated)")
    print("-" * 100)

    for i, (text_a, text_b, category) in enumerate(TEST_PAIRS):
        vec_a = _normalize(embeddings[i * 2])
        vec_b = _normalize(embeddings[i * 2 + 1])
        sim = _cosine_similarity(vec_a, vec_b)

        results_by_category[category].append(sim)
        print(f"{category:<15} {sim:>10.4f}  {text_a[:35]}... → {text_b[:35]}...")

    print("\n" + "=" * 100)
    print("\nSummary by category:")
    print(f"{'Category':<15} {'Min':>8} {'Max':>8} {'Mean':>8}")
    print("-" * 45)

    for cat in ["EXACT", "RELATED", "CONTRADICTION", "UNRELATED"]:
        scores = results_by_category[cat]
        if scores:
            print(f"{cat:<15} {min(scores):>8.4f} {max(scores):>8.4f} {sum(scores)/len(scores):>8.4f}")

    # Threshold recommendations
    print("\n" + "=" * 100)
    print("\nThreshold recommendations:")

    exact_min = min(results_by_category["EXACT"])
    related_max = max(results_by_category["RELATED"])
    contradiction_max = max(results_by_category["CONTRADICTION"])
    unrelated_max = max(results_by_category["UNRELATED"])

    print(f"\n  dedup_threshold (skip exact duplicates):")
    print(f"    Must be BELOW {exact_min:.4f} (lowest EXACT score)")
    print(f"    Must be ABOVE {related_max:.4f} (highest RELATED score)")
    dedup_sweet = (exact_min + related_max) / 2
    print(f"    → Sweet spot: {dedup_sweet:.2f}")

    print(f"\n  candidate_threshold (find nodes worth LLM consolidation):")
    print(f"    Must be BELOW {min(min(results_by_category['RELATED']), min(results_by_category['CONTRADICTION'])):.4f} (lowest RELATED/CONTRADICTION)")
    print(f"    Must be ABOVE {unrelated_max:.4f} (highest UNRELATED score)")
    candidate_sweet = (min(min(results_by_category["RELATED"]), min(results_by_category["CONTRADICTION"])) + unrelated_max) / 2
    print(f"    → Sweet spot: {candidate_sweet:.2f}")

    print(f"\n  Current defaults: dedup=0.80, candidate=0.45")

    # Validation
    ok = True
    if exact_min < 0.80:
        print(f"\n  ⚠️  EXACT duplicates scored as low as {exact_min:.4f} — dedup_threshold=0.80 may miss some")
        ok = False
    if related_max > 0.80:
        print(f"\n  ⚠️  RELATED nodes scored as high as {related_max:.4f} — may be incorrectly deduped at 0.80")
        ok = False
    if unrelated_max > 0.50:
        print(f"\n  ⚠️  UNRELATED nodes scored as high as {unrelated_max:.4f} — candidate_threshold=0.50 may surface noise")
        ok = False

    if ok:
        print(f"\n  ✅ Current defaults look good for these test pairs.")


if __name__ == "__main__":
    main()
