# [WNC] LightRAG Prompt Logging Documentation

## Purpose

When LightRAG gives a wrong answer, this logging feature helps you identify the root cause:

1. **LLM problem**: The LLM misunderstood or hallucinated
2. **Index DB problem**: The retrieved context is missing information
3. **Instruction problem**: The system prompt is poorly written

Shows you **exactly what was sent to the LLM** at each stage.

---

## Quick Start

### 1. Basic Configuration

Edit `wnc_scripts/openai_test_config.py`:

```python
# Enable prompt logging
enable_prompt_logging: bool = True

# Force re-index to see entity extraction logs
index_mode: Literal["skip", "incremental", "force"] = "force"

# Disable cache to see all LLM calls
enable_llm_cache: bool = False
enable_llm_cache_for_entity_extract: bool = False
```

### 2. Run Test

```bash
cd /srv/ai/LightRAG
python wnc_scripts/openai_test.py
```

---

## Configuration Options

### Prompt Logging Controls

```python
# Log prompts to console (shows system/user prompts, history)
enable_prompt_logging: bool = True

# Dump full prompts to JSON files
enable_prompt_file_dump: bool = False
prompt_dump_dir: str = "./prompt_logs"
```

### Trace ID Configuration

```python
# Prefix for trace IDs (set to None to disable)
trace_id_prefix: str | None = "query"

# Include timestamp in trace_id?
trace_id_include_timestamp: bool = False

# Auto-increment counter?
trace_id_auto_increment: bool = True

# Examples:
#   prefix="query", timestamp=False, increment=True  → query_001
#   prefix="query", timestamp=True,  increment=True  → query_20260113_113008_001
#   prefix="query", timestamp=True,  increment=False → query_20260113_113008
```

### Index Mode (NEW!)

```python
# "skip": Skip indexing, only query existing storage
# "incremental": Index new docs only (DEFAULT)
# "force": Delete storage and re-index everything

index_mode: Literal["skip", "incremental", "force"] = "incremental"
```

**When to use "force":**
- See entity extraction logs
- Test with cache disabled
- Verify indexing pipeline changes
- **WARNING**: Deletes all indexed data!

### Cache Configuration

```python
# Query-time LLM cache (keyword extraction, answer generation)
enable_llm_cache: bool = True

# Indexing-time LLM cache (entity extraction, summarization)
enable_llm_cache_for_entity_extract: bool = True
```

---

## What Gets Logged

### Two-Stage Logging Flow

```
Question → [QUERY_CONTEXT] → [LLM_PROMPT] → Answer
```

### Stage 1: `[WNC][QUERY_CONTEXT]` - Retrieval Context

Shows what was retrieved from the database:

```
[WNC][QUERY_CONTEXT][kg_query][trace_id=query_001] Retrieved Context:
  Entities: 15 items
  Relations: 8 items
  Chunks: 5 items
  mode: hybrid
  hl_keywords: Client's IP address, Networking
  ll_keywords: IP address, Client, Network
  context_length: 21547
```

**Debugging**: If context is missing key info → **Index problem**

---

### Stage 2: `[WNC][LLM_PROMPT]` - LLM Prompt

Shows exactly what was sent to the LLM:

```
[WNC][LLM_PROMPT][kg_query][direct_call][trace_id=query_001]
  Cache disabled - calling LLM directly

  SYSTEM PROMPT (24367 chars):
    ---Role---
    You are an expert AI assistant...

    ---Context---
    Knowledge Graph Data: [entities, relations]
    Document Chunks: [retrieved text]

  HISTORY: None

  USER PROMPT (28 chars):
    What is client's IP address?
```

**Debugging**: If context is correct but answer is wrong → **LLM or Instruction problem**

---

## Understanding SYSTEM vs USER Prompts

| Component | Purpose | Contents |
|-----------|---------|----------|
| **SYSTEM PROMPT** | Set LLM behavior | Role + Instructions + Retrieved Context |
| **USER PROMPT** | The question | User's actual question |
| **HISTORY** | Conversation | Previous user/assistant turns |

**Why separate?**
- Better LLM understanding
- Cleaner conversation history
- System context stays constant across turns

---

## Cache Behavior

### Three Levels of Caching

#### Level 1: Document-Level Deduplication
```
Ignoring document ID (already exists): doc-abc123
```
- **Scope**: Entire document by ID
- **To bypass**: Delete `working_dir` OR use `index_mode="force"`

#### Level 2: Indexing-Time LLM Cache
```
[WNC][LLM_PROMPT][entity_extraction] Cache miss - calling LLM (will cache result for extract)
```
- **Controlled by**: `enable_llm_cache_for_entity_extract`
- **Scope**: Entity extraction, summarization

#### Level 3: Query-Time LLM Cache
```
== LLM cache == Query cache hit, using cached response
[WNC][LLM_PROMPT][kg_query] CACHE HIT - using cached result
```
- **Controlled by**: `enable_llm_cache`
- **Scope**: Keyword extraction, answer generation

### Cache Status Messages

| Message | Meaning |
|---------|---------|
| `CACHE HIT - using cached result` | Using cache, no LLM call |
| `Cache miss - calling LLM (will cache result for query)` | LLM call, will be cached |
| `Cache disabled - calling LLM directly` | Cache turned off |

---

## Common Workflows

### Workflow 1: See All LLM Calls (No Cache)

**Config:**
```python
index_mode = "force"  # Force re-index
enable_llm_cache = False
enable_llm_cache_for_entity_extract = False
```

**Run:**
```bash
python wnc_scripts/openai_test.py
```

**You'll see:**
1. Entity extraction prompts during indexing
2. Keyword extraction prompt during query
3. Answer generation prompt during query

---

### Workflow 2: See Only Query Prompts (Use Existing Index)

**Config:**
```python
index_mode = "skip"  # Skip indexing
enable_llm_cache = False
```

**Run:**
```bash
python wnc_scripts/openai_test.py
```

**You'll see:**
1. Retrieved context (entities, relations, chunks)
2. Keyword extraction prompt
3. Answer generation prompt

---

### Workflow 3: Add New Documents (Incremental)

**Config:**
```python
index_mode = "incremental"  # Default
```

**Run:**
```bash
# Add new files to kdb_dir
python wnc_scripts/openai_test.py
```

**Behavior:**
- Existing documents: Skipped (no logs)
- New documents: Indexed (see entity extraction logs)

---

### Workflow 4: Production (Cache Enabled)

**Config:**
```python
index_mode = "skip"  # Don't re-index
enable_llm_cache = True
enable_llm_cache_for_entity_extract = True
enable_prompt_logging = False  # Quiet logs
```

---

## Debugging Scenarios

### Scenario 1: Wrong Answer, Correct Context

**Symptom**: Context has right info, but answer is wrong

**Debug steps:**
1. Check `[WNC][QUERY_CONTEXT]` - verify entities/chunks
2. Check `[WNC][LLM_PROMPT]` SYSTEM PROMPT - is it clear?
3. **Root cause**: LLM or instruction problem

**Solutions:**
- Improve system prompt
- Try different LLM model
- Adjust `response_type` parameter

---

### Scenario 2: Missing Information in Context

**Symptom**: Retrieved context doesn't have needed info

**Debug steps:**
1. Check `[WNC][QUERY_CONTEXT]` - what was retrieved?
2. Check keywords - are they accurate?
3. **Root cause**: Index or retrieval problem

**Solutions:**
- Verify documents contain the information
- Check if indexing actually ran (look for entity extraction logs)
- Adjust retrieval parameters (`top_k`, `max_entity_tokens`)
- Check embedding model quality

---

### Scenario 3: No Entity Extraction Logs

**Symptom**: Set `index_mode="incremental"` but no extraction logs

**Explanation**: Documents already exist in storage

**Solution:** Use `index_mode="force"` to re-index:
```python
index_mode = "force"  # Deletes working_dir
```

---

## JSON File Dumps

Enable with:
```python
enable_prompt_file_dump = True
prompt_dump_dir = "./prompt_logs"
```

Creates files like:
```
prompt_logs/
├── 20260113_113008_001_keyword_extraction_direct_call_trace_query_001.json
├── 20260113_113010_002_kg_query_direct_call_trace_query_001.json
```

Each file contains:
```json
{
  "timestamp": "20260113_113008_001",
  "stage": "kg_query",
  "cache_type": "direct_call",
  "trace_id": "query_001",
  "system_prompt": "Full system prompt...",
  "user_prompt": "What is client's IP address?",
  "history_messages": [],
  "message_sequence": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

**Use case**: Share exact prompts with team or LLM provider

---

## Environment Variables

Override config via environment:

```bash
export LOG_LEVEL=DEBUG
export WNC_ENABLE_PROMPT_LOGGING=true
export WNC_ENABLE_PROMPT_FILE_DUMP=true
export WNC_PROMPT_DUMP_DIR=./my_prompts
```

---

## Filtering Logs

```bash
# View specific trace
grep "trace_id=query_001" lightrag.log

# View only context retrieval
grep "\[QUERY_CONTEXT\]" lightrag.log

# View only LLM prompts
grep "\[LLM_PROMPT\]" lightrag.log

# View cache behavior
grep -E "CACHE HIT|Cache miss|Cache disabled" lightrag.log

# View entity extraction
grep "\[entity_extraction\]" lightrag.log
```

---

## Summary

This logging infrastructure helps you:

✅ See exactly what LightRAG sends to the LLM
✅ Debug cache behavior (hits, misses, disabled)
✅ Trace multi-query workflows (via trace_id)
✅ Distinguish LLM vs Index vs Instruction problems
✅ Export prompts for sharing (JSON dumps)
✅ Control indexing behavior (skip/incremental/force)

---

## Files Modified

- `lightrag/wnc_prompt_logger.py` - Logging implementation
- `lightrag/base.py` - Added `trace_id` to QueryParam
- `lightrag/utils.py` - Added logging to `use_llm_func_with_cache`
- `lightrag/llm/openai.py` - Added logging to OpenAI calls
- `lightrag/operate.py` - Added context and prompt logging at call sites
- `wnc_scripts/openai_test_config.py` - Added config options
- `wnc_scripts/openai_test.py` - Integrated trace_id and index_mode
