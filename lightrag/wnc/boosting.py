"""
WNC-specific boosting utilities for retrieval results.

This module provides functionality to adjust similarity scores based on various criteria
(source path, metadata, etc.) to prioritize specific results before final ranking and selection.
"""

from typing import Any


def apply_source_path_boost(
    similarities: list[tuple[str, float]],
    item_id_to_file_path: dict[str, str],
    boost_rules: list[dict[str, Any]],
) -> list[tuple[str, float, float, float]]:
    """
    Apply source path boost to similarity scores.

    This function adjusts scores based on source file paths,
    allowing prioritization of items from specific directories or files.

    Args:
        similarities: List of (item_id, raw_similarity_score) tuples from vector similarity
        item_id_to_file_path: Mapping from item_id to file_path
        boost_rules: List of {"prefix": path, "boost": value} rules
                     Example: [{"prefix": "/srv/ai/LightRAG/wnc_kdb/test_json_260126/", "boost": 0.05}]

    Returns:
        List of (item_id, adjusted_score, raw_score, applied_boost) tuples
        sorted by adjusted_score (highest first), tie-break by raw_score, then stable by original order

    Algorithm:
        1. For each item, find the matching boost rule with longest prefix match
        2. Calculate adjusted_score = raw_score + boost
        3. Sort by adjusted_score desc, tie-break by raw_score desc, then stable by original index
        4. Return sorted list with boost details

    Example:
        >>> similarities = [("chunk-1", 0.40), ("chunk-2", 0.38)]
        >>> item_id_to_file_path = {
        ...     "chunk-1": "/srv/ai/LightRAG/wnc_kdb/test_json_260126/file1.json",
        ...     "chunk-2": "/srv/ai/LightRAG/wnc_kdb/other/file2.txt"
        ... }
        >>> boost_rules = [{"prefix": "/srv/ai/LightRAG/wnc_kdb/test_json_260126/", "boost": 0.05}]
        >>> result = apply_source_path_boost(similarities, item_id_to_file_path, boost_rules)
        >>> # chunk-1: adjusted=0.45 (0.40 + 0.05), chunk-2: adjusted=0.38 (0.38 + 0.0)
        >>> # Result: [("chunk-1", 0.45, 0.40, 0.05), ("chunk-2", 0.38, 0.38, 0.0)]
    """
    results = []
    for idx, (item_id, raw_score) in enumerate(similarities):
        file_path = item_id_to_file_path.get(item_id, "")

        # Find the matching boost rule with longest prefix match
        applied_boost = 0.0
        max_prefix_len = 0
        for rule in boost_rules:
            prefix = rule.get("prefix", "")
            boost_value = rule.get("boost", 0.0)
            if file_path.startswith(prefix) and len(prefix) > max_prefix_len:
                applied_boost = boost_value
                max_prefix_len = len(prefix)

        adjusted_score = raw_score + applied_boost
        # Store original index for stable sorting
        results.append((item_id, adjusted_score, raw_score, applied_boost, idx))

    # Sort by adjusted_score desc, tie-break by raw_score desc, then stable by original index
    results.sort(key=lambda x: (-x[1], -x[2], x[4]))

    # Return without original index
    return [(item_id, adjusted, raw, boost) for item_id, adjusted, raw, boost, _ in results]
