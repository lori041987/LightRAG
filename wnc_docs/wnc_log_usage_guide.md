# WNC Log Usage Guide

This guide explains how to add `wnc_log()` calls to trace code execution in LightRAG.

## Table of Contents
- [Quick Start](#quick-start)
- [The Two-Rule Pattern](#the-two-rule-pattern)
- [How to Generate Each Field](#how-to-generate-each-field)
- [Configuration Variables](#configuration-variables)
- [Complete Examples](#complete-examples)

---

## Quick Start

Import the logging function:
```python
from lightrag.wnc.wnc_logging import wnc_log
```

Add logging to your function:
```python
def my_function(param1: str, param2: int) -> dict:
    # 1. Log at the beginning
    wnc_log(
        purpose="Brief description of what this function does",
        inputs={
            "param1": param1,
            "param2": param2,
        },
        outputs=None,  # or omit this line if function returns a value
        side_effects="Description of files/cache touched or None",
        note="Additional context, edge cases, error handling details",
        level="trace",
    )

    # ... function body ...

    result = {"status": "success", "count": 42}

    # 2. Log before return (if function returns a value)
    wnc_log(
        purpose="[OUTPUT] my_function completed",
        outputs={"status": "success", "count": 42},
        level="trace",
    )
    return result
```

---

## The Two-Rule Pattern

### Rule 1: One log at the beginning
Shows inputs, purpose, side_effects, note

**For functions that return a value:**
```python
wnc_log(
    purpose="What the function does",
    inputs={...},
    side_effects="...",
    note="...",
    level="trace",
)
```

**For functions that DON'T return a value (return None implicitly):**
```python
wnc_log(
    purpose="What the function does",
    inputs={...},
    outputs=None,  # Explicitly set to None
    side_effects="...",
    note="...",
    level="trace",
)
```

### Rule 2: Add logs before each return
Shows outputs only with `[OUTPUT]` prefix in purpose

**For normal returns:**
```python
wnc_log(
    purpose="[OUTPUT] Brief description of this return branch",
    outputs={...},  # The actual return value
    level="trace",
)
return result
```

**For early returns (unexpected/exceptional returns):**
Add "early return" in the purpose to indicate the function is returning before normal completion:
```python
wnc_log(
    purpose="[OUTPUT] function_name early return - reason",
    outputs={...},
    level="trace",
)
return result
```

Examples of early returns:
- `"[OUTPUT] handle_cache early return - no hashing_kv"`
- `"[OUTPUT] handle_cache early return - cache disabled for queries"`
- `"[OUTPUT] _locked_process_entity_name early return - cancelled"`
- `"[OUTPUT] _locked_process_edges early return - edge_data is None"`

---

## How to Generate Each Field

### 1. `purpose` (required)

**For initial log:**
- 1 sentence, present tense, user-facing
- Describe what the function does and why it exists
- Prefer docstring/comments/type hints
- Avoid vague phrasing like "handles", "does stuff", "wrapper"

**Examples:**
- Good: "Generates deterministic unique ID by computing MD5 hash of content and prepending prefix"
- Bad: "Handles ID generation stuff"

**For output log:**
- Use format: `"[OUTPUT] Brief description"`
- Examples:
  - `"[OUTPUT] Cache hit branch"`
  - `"[OUTPUT] Sanitized and stripped"`
  - `"[OUTPUT] JsonKVStorage branch"`

### 2. `inputs` (required for initial log)

Pass a dictionary with parameter names and values:

```python
inputs={
    "param1": param1,
    "param2": param2,
    "optional_param": optional_param if optional_param else "None",
}
```

**Field naming rules:**
- Use the actual parameter name as-is if it's a function parameter: `param1`, `chunk_token_size`, `llm_model_name`
- For computed/descriptive fields, use spaces: `"content length"`, `"max length"`, `"num items"`
- Only use underscores for actual variable/parameter names

**Examples:**
```python
# Good - parameter names as-is, descriptive fields with spaces
inputs={
    "content": content,
    "chunk_token_size": chunk_token_size,  # Parameter name
    "content length": len(content),        # Descriptive field - use space
}

# Bad - mixing conventions
inputs={
    "content_length": len(content),  # Should be "content length"
}
```

**Auto-formatting:**
- Dicts are automatically formatted as JSON with indentation
- No need to call `json.dumps()` manually
- Newlines and quotes are unescaped for readability

### 3. `outputs` (required for output logs, optional for initial log)

**For initial log:**
- Set to `None` if function doesn't return a value
- Omit if function returns a value (use output logs instead)

**For output logs:**
Pass the actual return value as a dict:

```python
outputs={
    "track_id": track_id,
    "status": "success",
}
```

Or for simple values:
```python
outputs={"result": result}
```

**Field naming rules:**
- Use spaces for readability: `"original length"`, `"cache hit"`, `"track id"`
- Only use underscores for pre-defined variable names: `track_id`, `arg_hash`, `cache_key`

**Examples:**
```python
# Good - spaces for descriptive fields
outputs={"summary": result, "truncated": True, "original length": len(content)}

# Good - underscores for variable names
outputs={"cache_hit": True, "arg_hash": arg_hash, "cache_key": cache_key}

# Bad - underscores for descriptive fields
outputs={"original_length": len(content)}  # Should be "original length"
```

**Auto-formatting:**
- Dicts are automatically formatted as JSON with indentation
- No need to call `json.dumps()` manually

### 4. `side_effects` (required for initial log)

List external effects in priority order:

1. Files/dirs created/written/deleted
2. Persistent stores (KV/doc/vector/graph DB)
3. Caches
4. Network calls
5. Env/global state/background tasks

**Format:**
```python
side_effects="Creates working_dir if it does not exist.\n"
            "Writes to KV_STORE_LLM_RESPONSE_CACHE (e.g. kv_store_llm_response_cache.json).\n"
            "Storage location depends on configured backend."
```

**If no side effects:**
```python
side_effects="None"
```

**Rules:**
- Be concrete (name file types/dirs/systems)
- Don't guess exact filenames unless deterministic
- Mention object initialization only if it triggers external effects
- If truly none: write `"None"`

### 5. `note` (optional but recommended for initial log)

Add when it provides value beyond purpose:

- Edge cases, validation rules, fallback behavior
- Error handling, retries, timeouts
- Assumptions (sync vs async, thread/event-loop constraints)
- Non-obvious performance implications
- Possible other branches/routes

**Format:**
```python
note="Has 3 branches: cache hit (read only), cache miss (read + write), cache disabled (no cache ops).\n"
     "Sanitizes all prompts via sanitize_text_for_encoding before hashing/LLM call.\n"
     "Wraps uncached LLM exceptions with [LLM func] prefix when cache disabled."
```

**Rules:**
- 0-3 short bullet-like sentences
- No repetition of purpose

### 6. `level` (required)

Use `"trace"` for all WNC logs:
```python
level="trace"
```

---

## Configuration Variables

Located in `lightrag/wnc/wnc_logging.py`:

```python
# JSON formatting settings for inputs/outputs
JSON_ENSURE_ASCII = False                       # Keep non-ASCII characters as-is
JSON_INDENT = 2                                 # Indentation level for readability
JSON_UNESCAPE_FOR_READABILITY = True            # Unescape \n and \" for better readability
JSON_AUTO_SERIALIZE_NONSTANDARD = True          # Auto-convert non-JSON-serializable types
```

### Basic Settings

**To disable unescaping (strict JSON):**
```python
JSON_UNESCAPE_FOR_READABILITY = False
```

**To change indentation:**
```python
JSON_INDENT = 4  # Use 4 spaces instead of 2
```

### Auto-Serialization (JSON_AUTO_SERIALIZE_NONSTANDARD)

**What it does:**
Automatically converts non-JSON-serializable types to JSON-compatible formats:
- Tuples → Lists: `("Alice", "knows", "Bob")` → `["Alice", "knows", "Bob"]`
- Sets → Lists: `{1, 2, 3}` → `[1, 2, 3]`
- Tuple keys in dicts → String keys: `{("A", "B"): value}` → `{"['A', 'B']": value}`
- Callables/Functions → Function names: `<function obj>` → `"<function func_name>"`
- **Timestamp Detection → Human-readable format**: `"timestamp": 1769154616` → adds `"timestamp (readable)": "2026-01-23 07:50:16 UTC"`

**Default: Enabled (True)**

This feature is enabled by default to handle common Python data structures like:
- Entity/relation graphs with tuple keys: `{("src", "rel", "tgt"): {...}}`
- Function objects passed as parameters
- Sets and other collection types

**To disable (strict JSON validation):**
```python
from lightrag.wnc import wnc_logging
wnc_logging.JSON_AUTO_SERIALIZE_NONSTANDARD = False
```

When disabled, `wnc_log()` will raise `TypeError` if you try to log non-JSON-serializable types.

**Use cases for disabling:**
- Debugging: want to catch non-serializable data early
- Strict validation: ensure all logged data is already JSON-compatible
- Performance: slight overhead from recursive type checking (usually negligible)

**Example with tuple keys:**
```python
# This works with JSON_AUTO_SERIALIZE_NONSTANDARD = True
edges = {
    ("Alice", "knows", "Bob"): {"weight": 0.9},
    ("Bob", "works_at", "Company"): {"weight": 0.8},
}

wnc_log(
    purpose="Process relationship edges",
    outputs={"edges": edges},  # Automatically serialized
    level="trace",
)

# Output in log:
# outputs={
#   "edges": {
#     "['Alice', 'knows', 'Bob']": {"weight": 0.9},
#     "['Bob', 'works_at', 'Company']": {"weight": 0.8}
#   }
# }
```

### Automatic Timestamp Detection and Conversion

**What it does:**
Automatically detects Unix timestamp fields and adds human-readable versions alongside the original values.

**Detection criteria:**
- Field name contains keywords: `timestamp`, `created_at`, `updated_at`, `time`, `date`
- Value is an integer in range 1,000,000,000 to 10,000,000,000 (2001-2286)

**Example:**
```python
entity_data = {
    "entity_name": "Routine Check",
    "timestamp": 1769154616,
    "created_at": 1769154616,
}

wnc_log(
    purpose="Process entity",
    outputs={"entity": entity_data},
    level="trace",
)

# Output in log automatically includes readable versions:
# outputs={
#   "entity": {
#     "entity_name": "Routine Check",
#     "timestamp": 1769154616,
#     "timestamp (readable)": "2026-01-23 07:50:16 UTC",
#     "created_at": 1769154616,
#     "created_at (readable)": "2026-01-23 07:50:16 UTC"
#   }
# }
```

**How it works:**
- When `JSON_AUTO_SERIALIZE_NONSTANDARD = True` (default), timestamp detection is enabled
- For each detected timestamp field, a new field with suffix ` (readable)` is added
- Readable format: `YYYY-MM-DD HH:MM:SS UTC`
- Original Unix timestamp is preserved for programmatic use
- Works recursively through nested dictionaries and lists

---

## Complete Examples

### Example 1: Function with return value

```python
def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Generate MD5 hash ID with prefix"""

    # Initial log (no outputs field since function returns a value)
    wnc_log(
        purpose="Generates deterministic unique ID by computing MD5 hash of content and prepending prefix",
        inputs={
            "content": content,
            "prefix": prefix if prefix else "'' (no prefix)",
        },
        side_effects="None",
        note="Used for generating IDs: doc-*, chunk-*, ent-*, rel-*.\n"
             "Deterministic: same content always produces same ID.\n"
             "Delegates to compute_args_hash for MD5 computation with safe unicode handling.",
        level="trace",
    )

    result = prefix + compute_args_hash(content)

    # Output log before return
    wnc_log(
        purpose="[OUTPUT] compute_mdhash_id completed",
        outputs={"id": result},
        level="trace",
    )
    return result
```

### Example 2: Function with multiple return branches

```python
def get_content_summary(content: str, max_length: int = 250) -> str:
    """Get summary of document content"""

    # Initial log
    wnc_log(
        purpose="Creates short preview of document content for display in document status entries",
        inputs={
            "content": content,
            "content length": len(content),  # Descriptive field - use space
            "max_length": max_length,         # Actual parameter name
        },
        side_effects="None",
        note="Truncates content to max_length and appends '...' if truncated.\n"
             "Used when creating initial DOC_STATUS entries during enqueue.\n"
             "Provides human-readable preview without storing full document content in status.",
        level="trace",
    )

    content = content.strip()

    # Branch 1: No truncation needed
    if len(content) <= max_length:
        wnc_log(
            purpose="[OUTPUT] No truncation needed",
            outputs={"summary": content, "truncated": False},
            level="trace",
        )
        return content

    # Branch 2: Content truncated
    result = content[:max_length] + "..."
    wnc_log(
        purpose="[OUTPUT] Content truncated",
        outputs={"summary": result, "truncated": True, "original length": len(content)},
        level="trace",
    )
    return result
```

### Example 3: Function with no return value

```python
def __post_init__(self):
    """Initialize LightRAG instance"""

    # Initial log with outputs=None
    wnc_log(
        purpose="Initializes LightRAG instance by validating configuration, instantiating storage backends, and wrapping embedding function with concurrency limits",
        inputs={
            "working_dir": self.working_dir,
            "kv_storage": self.kv_storage,
            "vector_storage": self.vector_storage,
            "graph_storage": self.graph_storage,
            "llm_model_name": getattr(self, "llm_model_name", None),
        },
        outputs=None,  # Function returns None implicitly
        side_effects="Creates working_dir if it does not exist; instantiates storage objects (does not load data until initialize_storages is called)",
        note="Validates storage backend compatibility and environment variables; wraps embedding_func with priority_limit_async_func_call for concurrency control; deprecated log_level/log_file_path parameters are removed after warning",
        level="trace"
    )

    # ... function body (no explicit return statement) ...
```

### Example 4: Function with early returns

```python
async def handle_cache(
    hashing_kv: BaseKVStorage,
    args_hash: str,
    cache_type: str = "default",
    mode: str = "default",
) -> tuple[str, int] | None:
    """Check LLM response cache for existing result"""

    # Initial log
    wnc_log(
        purpose="Checks llm_response_cache for existing LLM response matching the args_hash, returns cached content and timestamp if found, otherwise None",
        inputs={
            "hashing_kv": "provided" if hashing_kv else "None",
            "args_hash": args_hash,
            "cache_type": cache_type,
            "mode": mode,
        },
        side_effects="Reads from llm_response_cache via hashing_kv.get_by_id (kv_store_llm_response_cache.json).\n"
                    "No writes - this is a read-only cache lookup function.",
        note="Called by use_llm_func_with_cache to check cache before LLM call.\n"
             "Returns None in three cases: hashing_kv is None, cache disabled by config, cache miss.",
        level="trace",
    )

    # Early return 1: No storage provided
    if hashing_kv is None:
        wnc_log(
            purpose="[OUTPUT] handle_cache early return - no hashing_kv",
            outputs={"result": None, "reason": "hashing_kv is None"},
            level="trace",
        )
        return None

    # Early return 2: Cache disabled for queries
    if mode != "default":
        if not hashing_kv.global_config.get("enable_llm_cache"):
            wnc_log(
                purpose="[OUTPUT] handle_cache early return - cache disabled for queries",
                outputs={"result": None, "reason": "enable_llm_cache is False", "mode": mode},
                level="trace",
            )
            return None

    # Early return 3: Cache disabled for entity extraction
    else:
        if not hashing_kv.global_config.get("enable_llm_cache_for_entity_extract"):
            wnc_log(
                purpose="[OUTPUT] handle_cache early return - cache disabled for entity extraction",
                outputs={"result": None, "reason": "enable_llm_cache_for_entity_extract is False", "mode": mode},
                level="trace",
            )
            return None

    # Normal flow: Check cache
    cache_entry = await hashing_kv.get_by_id(args_hash)

    if cache_entry:
        content = cache_entry["return"]
        timestamp = cache_entry.get("create_time", 0)

        # Normal return: Cache hit
        wnc_log(
            purpose="[OUTPUT] handle_cache cache hit",
            outputs={
                "result": (content, timestamp),
                "cache_hit": True,
                "args_hash": args_hash,
                "content": content,
                "timestamp": timestamp,
            },
            level="trace",
        )
        return content, timestamp

    # Normal return: Cache miss
    wnc_log(
        purpose="[OUTPUT] handle_cache cache miss",
        outputs={"result": None, "cache_hit": False, "args_hash": args_hash},
        level="trace",
    )
    return None
```

### Example 5: Function with complex inputs/outputs

```python
async def use_llm_func_with_cache(
    user_prompt: str,
    use_llm_func: Callable,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    llm_response_cache: BaseKVStorage | None = None,
    cache_type: str = "default",
    chunk_id: str | None = None,
) -> tuple[str, int]:
    """Call LLM with cache support"""

    # Initial log
    wnc_log(
        purpose="Calls LLM with cache support by checking llm_response_cache for existing response matching prompt hash, returning cached result if found, otherwise calling LLM and saving response to cache",
        inputs={
            "system_prompt": system_prompt if system_prompt else "None",
            "user_prompt": user_prompt,
            "history_messages": history_messages if history_messages else "None",
            "cache_type": cache_type,
            "chunk_id": chunk_id if chunk_id else "None",
            "llm_response_cache": "provided" if llm_response_cache else "None (cache disabled)",
        },
        side_effects="Reads from KV_STORE_LLM_RESPONSE_CACHE via handle_cache (if cache enabled).\n"
                    "Writes to KV_STORE_LLM_RESPONSE_CACHE via save_to_cache (if cache miss and enable_llm_cache_for_entity_extract=True).\n"
                    "Storage location depends on configured backend (default: kv_store_llm_response_cache.json).",
        note="Has 3 branches: cache hit (read only), cache miss (read + write), cache disabled (no cache ops).\n"
             "Sanitizes all prompts via sanitize_text_for_encoding before hashing/LLM call.\n"
             "Wraps uncached LLM exceptions with [LLM func] prefix when cache disabled.",
        level="trace",
    )

    # ... cache check logic ...

    if cached_result:
        content, timestamp = cached_result

        # Cache hit branch
        wnc_log(
            purpose="[OUTPUT] Cache hit branch",
            outputs={"cache_hit": True, "arg_hash": arg_hash, "cache_key": cache_key, "content": content, "timestamp": timestamp},
            level="trace",
        )
        return content, timestamp

    # ... LLM call logic ...

    # Cache miss branch
    wnc_log(
        purpose="[OUTPUT] Cache miss branch",
        outputs={"cache_hit": False, "llm_called": True, "content": res, "timestamp": current_timestamp},
        level="trace",
    )
    return res, current_timestamp
```

---

## Quick Checklist

When adding `wnc_log` to a function:

- [ ] Import: `from lightrag.wnc.wnc_logging import wnc_log`
- [ ] Initial log at the beginning with `purpose`, `inputs`, `side_effects`, `note`
- [ ] Set `outputs=None` if function doesn't return a value
- [ ] Add `[OUTPUT]` log before EACH return statement
- [ ] For early returns, add "early return" in purpose: `"[OUTPUT] function_name early return - reason"`
- [ ] Use `level="trace"` for all logs
- [ ] Pass dicts directly (no manual `json.dumps()`)
- [ ] Use clear, readable field names with spaces: `"original length"` not `"original_length"`
- [ ] Only use underscores for pre-defined variable names: `track_id`, `arg_hash`
- [ ] Describe actual behavior, not implementation details in `purpose`
- [ ] Be concrete and specific in `side_effects`

---

## Common Mistakes to Avoid

❌ **Don't** add `outputs=` to initial log if function has return values
```python
# BAD
wnc_log(
    purpose="Process data",
    inputs={...},
    outputs="Will return processed data",  # Don't do this
    level="trace"
)
```

✅ **Do** add `outputs=None` only for functions that don't return
```python
# GOOD
wnc_log(
    purpose="Initialize storage",
    inputs={...},
    outputs=None,  # Function returns None implicitly
    level="trace"
)
```

❌ **Don't** manually call `json.dumps()`
```python
# BAD
outputs=json.dumps({"result": result}, ensure_ascii=False, indent=2)
```

✅ **Do** pass dict directly
```python
# GOOD
outputs={"result": result}
```

❌ **Don't** forget `[OUTPUT]` prefix in purpose for return logs
```python
# BAD
wnc_log(
    purpose="Function completed",  # Missing [OUTPUT] prefix
    outputs={...},
)
```

✅ **Do** use `[OUTPUT]` prefix
```python
# GOOD
wnc_log(
    purpose="[OUTPUT] Function completed",
    outputs={...},
)
```

❌ **Don't** use underscores for descriptive field names in inputs/outputs
```python
# BAD
inputs={"content_length": len(content), "max_length": 100}
outputs={"original_length": len(content), "num_chunks": 5}
```

✅ **Do** use spaces for descriptive fields, underscores only for actual variable names
```python
# GOOD
inputs={
    "content": content,           # Actual parameter name
    "content length": len(content),  # Descriptive field - use space
    "max_length": max_length,     # Actual parameter name
}
outputs={
    "original length": len(content),  # Descriptive field - use space
    "track_id": track_id,          # Actual variable name
}
```

❌ **Don't** forget "early return" wording for unexpected returns
```python
# BAD - unclear if this is expected or unexpected
wnc_log(
    purpose="[OUTPUT] handle_cache no hashing_kv",
    outputs={"result": None},
)
```

✅ **Do** add "early return" for unexpected/exceptional returns
```python
# GOOD - clearly indicates unexpected early exit
wnc_log(
    purpose="[OUTPUT] handle_cache early return - no hashing_kv",
    outputs={"result": None, "reason": "hashing_kv is None"},
)
```

---

## Summary

**Two-rule pattern:**
1. One log at beginning: `purpose`, `inputs`, `side_effects`, `note`, (`outputs=None` for void functions)
2. `[OUTPUT]` logs before each return: `purpose="[OUTPUT] ..."`, `outputs`

**Auto-formatting:**
- Both `inputs` and `outputs` dicts are auto-formatted as JSON
- Newlines and quotes are unescaped by default for readability
- All controlled by configuration variables

**Key principles:**
- Be clear and concise in descriptions
- Output actual data, not just metadata
- Be concrete about side effects
- Use consistent formatting
