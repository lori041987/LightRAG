# Ollama Response Metrics Explanation

## Overview

When you see Ollama response logs like this:

```
Response: model='qwen3:8b' created_at='2026-01-30T10:38:41.198811715Z' done=True
done_reason='stop' total_duration=15132607810 load_duration=139510288
prompt_eval_count=620 prompt_eval_duration=11255834268 eval_count=28
eval_duration=3713762703 message=Message(...)
```

This document explains what each field means.

## Duration Metrics

**IMPORTANT**: All `*_duration` values are in **nanoseconds** (billionths of a second).

To convert to seconds: `nanoseconds / 1,000,000,000 = seconds`

### Field Explanations

| Field | Unit | Description | Example |
|-------|------|-------------|---------|
| `total_duration` | nanoseconds | Total time from request start to completion. Includes loading model + processing prompt + generating response. | `15132607810` = **15.13 seconds** |
| `load_duration` | nanoseconds | Time to load the model into memory. Close to 0 if model was already loaded. Only significant on first request or after model unload. | `139510288` = **0.14 seconds** |
| `prompt_eval_duration` | nanoseconds | Time to process/understand the input prompt. This is the "thinking" phase before generating output. Depends on prompt length and complexity. | `11255834268` = **11.26 seconds** |
| `eval_duration` | nanoseconds | Time to generate the output tokens. This is the actual text generation phase. Depends on output length. | `3713762703` = **3.71 seconds** |

### Duration Relationship

```
total_duration ≈ load_duration + prompt_eval_duration + eval_duration
```

## Token Count Metrics

| Field | Unit | Description | Example |
|-------|------|-------------|---------|
| `prompt_eval_count` | tokens | Number of input tokens processed (your prompt + system prompt + history). | `620` tokens |
| `eval_count` | tokens | Number of output tokens generated (the model's response). | `28` tokens |

## Other Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `model` | string | The Ollama model name used | `qwen3:8b` |
| `created_at` | ISO 8601 timestamp | When the response was generated (UTC) | `2026-01-30T10:38:41.198811715Z` |
| `done` | boolean | Whether the response is complete | `true` |
| `done_reason` | string | Why generation stopped: `stop` (natural end), `length` (hit token limit), `load` (model loading) | `stop` |
| `message` | object | Contains the actual response content and role | `Message(role='assistant', content='...')` |

## Example Analysis

Given this response:
```
total_duration=15132607810 load_duration=139510288
prompt_eval_count=620 prompt_eval_duration=11255834268
eval_count=28 eval_duration=3713762703
```

**Converted to seconds:**
- **Total time**: 15.13 seconds
- **Model loading**: 0.14 seconds (model was already in memory)
- **Processing 620 input tokens**: 11.26 seconds
- **Generating 28 output tokens**: 3.71 seconds

**Performance insights:**
- Model was already loaded (fast load time)
- Most time spent processing the prompt (11.26s / 15.13s = 74%)
- Generated 28 tokens in 3.71s = **~7.5 tokens/second**
- Processed 620 input tokens in 11.26s = **~55 tokens/second**

## Performance Tips

1. **Keep model loaded**: If `load_duration` is high (> 1 second), the model is being loaded from disk. Use `keep_alive` parameter to keep it in memory.

2. **Monitor prompt_eval_duration**: Large prompts (high `prompt_eval_count`) take longer to process. Consider reducing context if this is too slow.

3. **Track token generation speed**: `eval_count / (eval_duration / 1_000_000_000)` gives tokens per second. Lower-end hardware will generate slower.

4. **Optimize for your use case**:
   - For long outputs: `eval_duration` dominates
   - For short outputs with long prompts: `prompt_eval_duration` dominates

## Source

These metrics come from the **Ollama Python SDK** (`ollama` package) when calling:
```python
response = await ollama_client.chat(model=model, messages=messages, **kwargs)
```

This is NOT the same as LightRAG's `OllamaGenerateResponse` class in `lightrag/api/routers/ollama_api.py`, which is used when LightRAG acts as an Ollama-compatible API server.

## References

- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Ollama Python SDK](https://github.com/ollama/ollama-python)
