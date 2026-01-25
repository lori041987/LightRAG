# WNC Logging Progress - January 23, 2026

## Summary
Added comprehensive `wnc_log` tracing to LightRAG indexing pipeline functions.

## Completed Today

### 1. Timestamp Auto-Detection Feature ✓
**File**: `lightrag/wnc/wnc_logging.py`

Added automatic Unix timestamp detection and human-readable conversion:
- Detects fields with keywords: `timestamp`, `created_at`, `updated_at`, `time`, `date`
- Validates integer range: 1,000,000,000 to 10,000,000,000 (years 2001-2286)
- Automatically adds `"<field> (readable)"` with format: `YYYY-MM-DD HH:MM:SS UTC`
- Works recursively through nested dicts and lists
- Enabled by default via `JSON_AUTO_SERIALIZE_NONSTANDARD = True`

**Documentation**: Updated `wnc_docs/wnc_log_usage_guide.md` with timestamp detection examples

### 2. Core Pipeline Functions with wnc_log ✓

#### High Priority Functions (2/6 completed):

1. **`_merge_nodes_then_upsert`** (lightrag/operate.py:1619) ✓
   - Initial log: purpose, inputs, side_effects, note
   - Output logs:
     - Early return (KEEP limit reached)
     - Successful completion with "actions performed" list
   - Shows: created vs merged, LLM usage, deduplication, truncation, storage updates

2. **`_merge_edges_then_upsert`** (lightrag/operate.py:1960) ✓
   - Initial log: purpose, inputs, side_effects, note
   - Output logs:
     - Self-loop early return
     - Early return (KEEP limit reached)
     - Successful completion with "actions performed" list
   - Shows: created vs merged, LLM usage, weight/keyword merging, missing entities created

#### Still TODO (4/6 remaining):
3. `_process_extraction_result` (lightrag/operate.py:911)
4. `handle_cache` (lightrag/utils.py:1405)
5. `save_to_cache` (lightrag/utils.py:1451)
6. `update_chunk_cache_list` (lightrag/utils.py:1916)

### 3. Other Functions with wnc_log ✓

**Already completed** (18 functions):
- `LightRAG.__post_init__`
- `LightRAG._get_storage_class`
- `LightRAG.initialize_storages`
- `LightRAG.insert`
- `LightRAG.ainsert`
- `LightRAG.apipeline_enqueue_documents`
- `LightRAG.apipeline_process_enqueue_documents`
- `LightRAG._process_extract_entities`
- `LightRAG._insert_done` (with storage details)
- `compute_mdhash_id`
- `generate_track_id`
- `sanitize_text_for_encoding`
- `get_content_summary`
- `use_llm_func_with_cache`
- `chunking_by_token_size`
- `extract_entities`
- `merge_nodes_and_edges`
- `NanoVectorDBStorage.upsert`

### 4. Key Improvements Made Today

**Better "actions performed" reporting**:
- Added explicit action list showing what each function did
- Examples:
  - "Created new entity in knowledge graph" vs "Merged with existing entity"
  - "Applied LLM summarization" vs "No LLM summarization needed"
  - "Deduplicated X duplicate descriptions"
  - "Applied source_ids truncation: FIFO"
  - "Updated entity_chunks_storage with X chunk IDs"

**Clearer purpose descriptions**:
- Changed from: "Merges entity nodes by getting existing node..."
- To: "Checks if entity exists; if exists, merges; if not exists, creates new; then upserts..."

**Storage details in `_insert_done`**:
- Shows storage name + class for each active storage
- Example: `{"name": "full_docs", "class": "JsonKVStorage"}`

## Still TODO - Medium Priority

### Storage Callback Functions (5 functions):
- `JsonKVStorage.index_done_callback` (lightrag/kg/json_kv_impl.py:77)
- `JsonKVStorage.upsert` (lightrag/kg/json_kv_impl.py:149)
- `JsonDocStatusStorage.index_done_callback` (lightrag/kg/json_doc_status_impl.py:161)
- `JsonDocStatusStorage.upsert` (lightrag/kg/json_doc_status_impl.py:186)
- `NanoVectorDBStorage.index_done_callback` (lightrag/kg/nano_vector_db_impl.py:273)
- `NetworkXStorage.index_done_callback` (lightrag/kg/networkx_impl.py:503)

### Helper Functions (5 functions):
- `_handle_entity_relation_summary` (lightrag/operate.py:166)
- `_summarize_descriptions` (lightrag/operate.py:298)
- `_handle_single_entity_extraction` (lightrag/operate.py:380)
- `_handle_single_relationship_extraction` (lightrag/operate.py:452)
- Nested functions in `merge_nodes_and_edges` (if needed)

## Log Level Guidelines

All logs use `level="info"` by default as requested by user.

## Files Modified Today

1. `lightrag/wnc/wnc_logging.py` - Added timestamp detection
2. `lightrag/operate.py` - Added logs to merge functions
3. `lightrag/lightrag.py` - Updated `_insert_done` logs
4. `lightrag/kg/nano_vector_db_impl.py` - Added logs to `upsert`
5. `wnc_docs/wnc_log_usage_guide.md` - Updated documentation

## Testing

User tested with:
```bash
python wnc_scripts/openai_test.py
```

Output logs to: `wnc_log/test_4.log`

## Next Session Plan

1. Complete remaining 4 high-priority functions:
   - `_process_extraction_result`
   - `handle_cache`
   - `save_to_cache`
   - `update_chunk_cache_list`

2. Add logs to medium-priority storage callbacks (6 functions)

3. Add logs to helper functions (5 functions) if needed

4. Final testing and verification

## Notes

- All timestamp fields are automatically converted to human-readable format
- "actions performed" list makes it clear what each function did
- Purpose descriptions clarified to show check-then-merge-or-create logic
- User prefers normal case for field names (not ALL CAPS)
- Duplication of entity_name in both top-level and node_data is acceptable for clarity

## Additional Improvements Made (Session 2 - Same Day)

### 1. File Path Reporting in "actions performed" ✓
- Retrieve actual file paths from storage instances using `getattr()`
- Shows exact files modified during operations
- Works with different storage backends (not hardcoded)
- Falls back to descriptive names if attribute doesn't exist

**Implementation**:
- `knowledge_graph_inst._graphml_xml_file` → graph file path
- `entity_vdb._client_file_name` → entity vector DB file path
- `relationships_vdb._client_file_name` → relationship vector DB file path
- `entity_chunks_storage._file_name` → entity chunks KV file path
- `relation_chunks_storage._file_name` → relation chunks KV file path

**Example output**:
```json
"actions performed": [
  "Upserted edge to knowledge graph → /path/to/graph_chunk_entity_relation.graphml",
  "Upserted new relationship → /path/to/vdb_relationships.json",
  "Updated relation chunks storage (1 chunk IDs) → /path/to/kv_store_relation_chunks.json"
]
```

### 2. Accurate Vector Delete Logging ✓
**Investigation**: Discovered that `NanoVectorDB.delete()` doesn't throw exception when IDs don't exist - it just does nothing (no-op).

**Issue**: Original log said "Deleted old relationship vectors" even for new relationships where nothing was deleted.

**Fix**: Differentiate log message based on whether relationship is new or existing:
- New relationship: "Upserted new relationship → vdb_relationships.json"
- Existing relationship: "Cleaned up old vectors and upserted updated relationship → vdb_relationships.json"

**Added comment in code**:
```python
# Note: The code always calls "await relationships_vdb.delete([rel_vdb_id, rel_vdb_id_reverse])" before upsert, even for new relationships.
# NanoVectorDB's delete() doesn't throw exception when IDs don't exist - it just does nothing (no-op).
# So for new relationships, delete succeeds but removes 0 items; for existing relationships, it removes old vectors.
# We differentiate the log message based on whether this is a new or existing relationship.
```

### 3. Investigation Notes

**Why no error when deleting non-existent vectors?**
- Checked `NanoVectorDBStorage.delete()` implementation in `lightrag/kg/nano_vector_db_impl.py`
- It calls `client.delete(ids)` which is NanoVectorDB's native delete
- NanoVectorDB's delete is a silent no-op when IDs don't exist (no exception thrown)
- This is why the try/except in `_merge_edges_then_upsert` never logs the debug message for new relationships

**Debug logging observation**:
- User noted that lightrag logger prints DEBUG level logs to `wnc_log/test_5.log`
- Example: "Processing relation ['5GHz Band', 'Client Info A']" at DEBUG level
- But "Could not delete old relationship vector records" never appears for new relationships
- This confirms that delete succeeds silently for non-existent IDs (no exception)
