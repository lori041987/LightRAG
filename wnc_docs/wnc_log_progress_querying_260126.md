# WNC Log Progress - Query Pipeline Functions (2026-01-26)

## Summary

Adding `wnc_log` trace logging to all functions mentioned in `wnc_docs/code_analysis_query_260126.txt` for the query pipeline.

**Status:** ✅ COMPLETE (23/23 functions completed)

**Note:** Total count is 23 instead of 28 because:
- 4 functions already have WNC logging from previous work (LightRAG.initialize_storages, handle_cache, save_to_cache, LightRAG.__post_init__)
- 1 item is not a function (PROMPTS["rag_response"] is a string template)
- 2 utility functions added for completeness (compute_args_hash, openai_embed)

## Completed Functions ✅

### 1. LightRAG.query (lightrag/lightrag.py:2740)
- **Added:** Initial log + output log
- **Location:** lightrag/lightrag.py
- **Notes:** Synchronous wrapper around async query pipeline

### 2. LightRAG.aquery (lightrag/lightrag.py:2761)
- **Added:** Initial log + 2 output logs (streaming/non-streaming)
- **Location:** lightrag/lightrag.py
- **Notes:** Backward-compatible async wrapper around aquery_llm

### 3. LightRAG.aquery_llm (lightrag/lightrag.py:3042)
- **Added:** Initial log + 4 output logs (bypass streaming/non-streaming, early return, success, exception)
- **Location:** lightrag/lightrag.py
- **Notes:** Top-level async query API that dispatches by mode

### 4. kg_query (lightrag/operate.py:3885)
- **Added:** Initial log + 6 output logs (empty query, no context, only_need_context, only_need_prompt, streaming, non-streaming)
- **Location:** lightrag/operate.py
- **Notes:** Hybrid/local/global/mix query worker with multiple return paths
- **Special:** All output logs include "note" field explaining scenario

### 5. get_keywords_from_query (lightrag/operate.py:4095)
- **Added:** Initial log + 2 output logs (provided keywords, extracted keywords)
- **Location:** lightrag/operate.py
- **Notes:** Returns HL/LL keywords from QueryParam or delegates to extract_keywords_only

### 6. extract_keywords_only (lightrag/operate.py:4127)
- **Added:** Initial log + 4 output logs (cache hit, no JSON structure, JSON parse error, LLM extraction complete)
- **Location:** lightrag/operate.py
- **Notes:** Calls LLM with keyword_extraction=True, parses JSON response
- **Special:** LLM returns JSON with high_level_keywords and low_level_keywords arrays

### 7. _build_query_context (lightrag/operate.py:5072)
- **Added:** Initial log + 4 output logs (empty query, no entities/relations, mix mode no chunks, no content after merge, success)
- **Location:** lightrag/operate.py
- **Notes:** 4-stage pipeline: search → truncate → merge → build context
- **Special:** Returns QueryContextResult with context (LLM prompt string) and raw_data (structured entities/relationships/chunks)

### 8. _perform_kg_search (lightrag/operate.py:4447)
- **Added:** Initial log + 1 output log
- **Location:** lightrag/operate.py
- **Notes:** Executes mode-specific knowledge graph retrieval, round-robin merges results
- **Key Terms:**
  - KG = Knowledge Graph
  - Cosine similarity = measure of angle between embedding vectors (1.0=identical, 0.0=unrelated)

### 9. _get_node_data (lightrag/operate.py:5276)
- **Added:** Initial log + 2 output logs (no entities found, success)
- **Location:** lightrag/operate.py
- **Notes:** Local-mode retrieval via entity vector DB
- **Key Terms:**
  - Degree = number of edges connected to a node (connection count)
  - Used as rank indicator (higher degree = more important/central entity)
  - Weight = relationship strength stored during indexing stage
  - Returns entities sorted by cosine similarity, relations sorted by (rank+weight)

### 10. _get_edge_data (lightrag/operate.py:5583)
- **Added:** Initial log + 2 output logs (no relationships found, success)
- **Location:** lightrag/operate.py
- **Notes:** Global-mode retrieval via relationship vector DB
- **Special:** Relations maintain vector search order (sorted by similarity)

## Newly Completed Functions (11/11) ✅

### Core Query Functions
11. **_merge_all_chunks** (lightrag/operate.py:4813)
   - **Added:** Initial log + 1 output log
   - **Location:** lightrag/operate.py
   - **Notes:** Stage 3 of _build_query_context - fetches entity/relation chunks, round-robin merges with vector chunks

12. **_build_context_str** (lightrag/operate.py:4915)
   - **Added:** Initial log + 3 output logs (no tokenizer, no content, success)
   - **Location:** lightrag/operate.py
   - **Notes:** Stage 4 of _build_query_context - allocates token budget, reranks, truncates, formats final context

### Helper Functions - Entity/Relation Processing
13. **_find_most_related_edges_from_entities** (lightrag/operate.py:5445)
   - **Added:** Initial log + 1 output log
   - **Location:** lightrag/operate.py
   - **Notes:** Fetches connected edges from graph and ranks by (rank + weight)

14. **_find_most_related_entities_from_relationships** (lightrag/operate.py:5824)
   - **Added:** Initial log + 1 output log
   - **Location:** lightrag/operate.py
   - **Notes:** Extracts unique entity endpoints from pre-ranked relationships

15. **_find_related_text_unit_from_entities** (lightrag/operate.py:5521)
   - **Added:** Initial log + 4 output logs (no entities, no chunks, no selection, success)
   - **Location:** lightrag/operate.py
   - **Notes:** Selects entity chunks via WEIGHT polling or VECTOR similarity

16. **_find_related_text_unit_from_relations** (lightrag/operate.py:5877)
   - **Added:** Initial log + 5 output logs (no relations, no chunks, all deduplicated, no selection, success)
   - **Location:** lightrag/operate.py
   - **Notes:** Selects relation chunks, deduplicates against entity_chunks

### Chunk Processing Functions (lightrag/utils.py)
17. **process_chunks_unified** (lightrag/utils.py:3062)
   - **Added:** Initial log + 3 output logs (no chunks, filtered by rerank, success)
   - **Location:** lightrag/utils.py
   - **Notes:** Applies reranking, enforces chunk_top_k, token truncation, adds DC IDs

18. **apply_rerank_if_enabled** (lightrag/utils.py:2978)
   - **Added:** Initial log + 5 output logs (disabled, no func, empty results, exception, success with formats)
   - **Location:** lightrag/utils.py
   - **Notes:** Calls rerank_model_func and reorders chunks by relevance score

19. **generate_reference_list_from_chunks** (lightrag/utils.py:3781)
   - **Added:** Initial log + 2 output logs (no chunks, success)
   - **Location:** lightrag/utils.py
   - **Notes:** Assigns reference_ids by file_path frequency/order

20. **convert_to_user_format** (lightrag/utils.py:3657)
   - **Added:** Initial log + 1 output log
   - **Location:** lightrag/utils.py
   - **Notes:** Converts internal structures to user-friendly raw_data format

### LLM & Utility Functions
21. **openai_complete_if_cache** (lightrag/llm/openai.py:197)
   - **Added:** Initial log + 2 output logs (streaming, non-streaming)
   - **Location:** lightrag/llm/openai.py
   - **Notes:** Makes OpenAI Chat Completions API calls for keyword extraction and answer generation

22. **compute_args_hash** (lightrag/utils.py:561)
   - **Added:** Initial log + 2 output logs (normal encoding, fallback encoding)
   - **Location:** lightrag/utils.py
   - **Notes:** Computes MD5 hash for cache key generation with safe Unicode handling

23. **openai_embed** (lightrag/llm/openai.py:759)
   - **Added:** Initial log + 1 output log
   - **Location:** lightrag/llm/openai.py
   - **Notes:** Generates embeddings via OpenAI/Azure API for vector operations

## Logging Pattern Established

### Initial Log Format
```python
wnc_log(
    purpose="Brief description of what function does",
    inputs={
        "param1": param1,
        "param2": param2,
    },
    side_effects="Description of storage/network/cache operations",
    note="Additional context, edge cases, return value info",
    level="info",
)
```

### Output Log Format
```python
wnc_log(
    purpose="[OUTPUT] function_name scenario_description",
    outputs={
        "result_field": actual_value,  # Print actual values, not just lengths
        "note": "Explanation of this return scenario",
    },
    level="info",
)
```

### Key Principles
1. **Do NOT change existing code** - only add wnc_log calls
2. **Do NOT create new variables** - use existing ones or inline expressions
3. **Print actual values** in outputs, not just lengths (for debugging)
4. **Add "note" field** in output logs to explain the scenario
5. **Use level="info"** for all query pipeline logs
6. **One initial log** at function start
7. **Output logs before EACH return** statement
8. **For early returns**, add "early return" in purpose: `"[OUTPUT] function_name early return - reason"`

## Key Terms / Concepts Documented

- **KG** = Knowledge Graph (entities + relationships + text chunks)
- **Cosine similarity** = measure of angle between embedding vectors (1.0=identical, 0.0=unrelated)
- **Degree** = number of edges connected to a node (used as rank indicator)
- **Weight** = relationship strength stored during indexing stage
- **Context** = LLM prompt string (formatted text injected into system prompt)
- **Raw_data** = Structured dict with entities/relationships/chunks/references/metadata
- **QueryContextResult** = Container with both context (str) and raw_data (dict)

## Summary of Work Completed

### Total Functions Logged: 23/23 ✅

### Files Modified:
- **`lightrag/lightrag.py`** (3 functions)
  - LightRAG.query, aquery, aquery_llm

- **`lightrag/operate.py`** (10 functions)
  - kg_query, get_keywords_from_query, extract_keywords_only
  - _build_query_context, _perform_kg_search, _get_node_data, _get_edge_data
  - _merge_all_chunks, _build_context_str
  - _find_most_related_edges_from_entities, _find_most_related_entities_from_relationships
  - _find_related_text_unit_from_entities, _find_related_text_unit_from_relations

- **`lightrag/utils.py`** (5 functions)
  - process_chunks_unified, apply_rerank_if_enabled
  - generate_reference_list_from_chunks, convert_to_user_format
  - compute_args_hash

- **`lightrag/llm/openai.py`** (2 functions)
  - openai_complete_if_cache
  - openai_embed

### Total WNC Logs Added:
- **Initial logs:** 23 (one per function)
- **Output logs:** 50+ (covering all return paths and scenarios)
- **Total:** 73+ trace logs

## Next Steps

1. ✅ Test the logging by running a query and verifying trace output
2. ✅ Ensure all logs print actual data values (not just counts)
3. ✅ Verify all "note" fields provide clear scenario explanations with context
4. 📋 Document any findings or issues discovered during testing
