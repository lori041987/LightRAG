# Understanding Embedding Vectors

## Overview

When you see embedding output in logs like this:

```
embeddings=[[-0.03207394, 0.024582954, -0.04838823, -0.012818577, 0.009258405, ...]]
Embeddings shape: (1, 1024)
```

This document explains what those numbers mean and how they work.

## What Are Embedding Vectors?

Embedding vectors are **numerical representations of text** in high-dimensional space. The embedding model (e.g., bge-m3:567m, text-embedding-3-small) converts human-readable text into arrays of floating-point numbers that capture semantic meaning.

### Example

**Input text:**
```
"What is the client's IP address? The IP is 192.168.1.105"
```

**Output vector (simplified, showing first 10 of 1024 dimensions):**
```python
[-0.03207394, 0.024582954, -0.04838823, -0.012818577, 0.009258405,
 -0.032204874, 0.010917582, 0.019531272, 0.0025869145, 0.018748332, ...]
# ... 1014 more numbers ...
```

## The Numbers Explained

### What Each Number Represents

Each number in the embedding vector is a **dimension** in high-dimensional vector space:

- **Single number**: A learned weight/feature representing one semantic aspect
- **All numbers together**: The complete semantic "fingerprint" of the text
- **Not human-readable**: Individual values cannot be interpreted directly (no "dimension 5 = IP addresses")
- **Pattern matters**: The **pattern across all dimensions** encodes the meaning

### Key Properties

1. **Dimension Count**:
   - bge-m3:567m outputs **1024 dimensions** (fixed)
   - text-embedding-3-small outputs **1536 dimensions** (default)
   - Shape `(1, 1024)` means: 1 text → 1 vector with 1024 numbers
   - Shape `(5, 1024)` means: 5 texts → 5 vectors, each with 1024 numbers

2. **Value Range**:
   - Typically between **-1.0 to +1.0** (after normalization)
   - Both positive and negative values are normal
   - Examples: `-0.03207394`, `0.024582954`, `-0.04838823`

3. **Vector Space**:
   - Each dimension is an axis in high-dimensional space
   - Semantically similar texts → vectors point in similar directions
   - Semantic distance → geometric distance in vector space

## How Embeddings Work in RAG

### 1. Indexing Phase

When documents are indexed:

```python
# Document chunks/entities/relations
texts = [
    "Client MAC: 11:22:33:44:55:66, IP: 192.168.1.105",
    "SSID: Office_Staff, Band: 5GHz, RSSI: -55dBm",
    "Router configuration for office network"
]

# Convert to embeddings
embeddings = embed_model(texts)
# Result: 3 vectors of 1024 dimensions each
# Shape: (3, 1024)

# Store in vector database
vector_db.store(texts, embeddings)
```

### 2. Query Phase

When user asks a question:

```python
# User query
query = "What is the client's IP address?"

# Convert query to embedding
query_embedding = embed_model(query)
# Result: 1 vector of 1024 dimensions
# Shape: (1, 1024)

# Find similar vectors in database
similar_vectors = vector_db.search(query_embedding, top_k=5)
# Returns: chunks/entities with highest cosine similarity
```

### 3. Similarity Calculation

**Cosine Similarity** measures how "close" two vectors are:

```python
# Two similar texts
text1 = "What is the client's IP address?"
text2 = "Tell me the client's IP"

# Embeddings (simplified)
vec1 = [-0.032, 0.024, -0.048, 0.015, ...]  # 1024 numbers
vec2 = [-0.031, 0.025, -0.047, 0.016, ...]  # 1024 numbers (very similar!)

# Cosine similarity: dot product / (magnitude1 * magnitude2)
similarity = cosine(vec1, vec2)  # ≈ 0.95 (very similar!)
```

**Similarity scores:**
- `1.0` = identical meaning (same direction)
- `0.8-0.95` = very similar meaning
- `0.5-0.8` = somewhat related
- `0.0` = unrelated (perpendicular)
- `-1.0` = opposite meaning (rare)

## Real Example from Logs

### Input Text (from wnc_logs/lightrag_test_0202_1.log:906)

```json
{
  "id": "sample_001_basic_query",
  "tags": ["Routine Check", "Device Info"],
  "input_datas": [
    {
      "filename": "client_info_A.xml",
      "content_raw": "<Client><Mac>11:22:33:44:55:66</Mac><IP>192.168.1.105</IP>...</Client>"
    }
  ],
  "ground_truths": {
    "Ask": [
      {
        "question": "What is the IP address of this client?",
        "answer": "The client's IP address is 192.168.1.105."
      }
    ]
  }
}
```

### Output Embedding (line 910)

```python
embeddings=[[-0.03207394, 0.024582954, -0.04838823, -0.012818577, 0.009258405,
-0.032204874, 0.010917582, 0.019531272, 0.0025869145, 0.018748332,
-0.008091098, -0.0151854325, -0.037061837, -0.009103098, 0.052919157, ...
# ... 1009 more numbers ...
0.05303044, 0.00274306]]

# Shape: (1, 1024)
# Model: bge-m3:567m
# 1 text input → 1 embedding vector with 1024 dimensions
```

### How It's Used

1. **Storage**: This 1024-dimensional vector is stored in the vector database (chunks/entities/relationships namespace)

2. **Query matching**: When user asks "What is the client's IP?", that question also becomes a 1024-dimensional vector

3. **Similarity search**: The system finds stored vectors with high cosine similarity to the query vector

4. **Result**: Returns the original text associated with similar vectors (the chunk containing IP information)

## Technical Details

### Embedding Models

Different models produce different dimension counts:

| Model | Provider | Dimensions | Notes |
|-------|----------|------------|-------|
| bge-m3:567m | Ollama | 1024 | Fixed output, no reduction support in LightRAG |
| text-embedding-3-small | OpenAI | 1536 | Native output (supports API reduction to smaller dims) |
| text-embedding-3-large | OpenAI | 3072 | Native output (supports API reduction to smaller dims) |

**IMPORTANT**: The `embed_dim` configuration must match the model's actual output dimension:
```python
# Correct configuration
embed_model = "bge-m3:567m"
embed_dim = 1024  # Must match bge-m3 output

# Wrong configuration = errors!
embed_model = "bge-m3:567m"
embed_dim = 1536  # WRONG - dimension mismatch error
```

### No Prompts for Embeddings

**Critical difference from LLM calls:**

- **LLM (chat) calls**: Use prompts/instructions
  ```python
  llm("You are a helpful assistant. Answer: What is 2+2?")
  # Model follows instructions and generates response
  ```

- **Embedding calls**: NO prompts, just raw text
  ```python
  embed_model("What is 2+2?")
  # Model converts text to vector, no generation
  ```

Embedding models are **encoders only** - they compress text into fixed-size vectors. They don't generate responses or follow instructions.

### Vector Database Storage

Embeddings are stored in vector databases (e.g., NanoVectorDB) for fast similarity search:

```python
# Storage structure
{
    "chunk_id": "chunk_001",
    "text": "Client IP: 192.168.1.105",
    "embedding": [-0.032, 0.024, -0.048, ...],  # 1024 numbers
    "metadata": {"source": "client_info_A.xml"}
}
```

When querying:
1. Convert query to embedding: `query_vec = embed_model(query)`
2. Calculate similarity with all stored vectors: `cosine(query_vec, stored_vec)`
3. Return top-k most similar chunks/entities/relationships

## Analogy: Semantic Fingerprints

Think of embedding vectors like **fingerprints**:

- **Fingerprint**: Unique pattern of ridges and whorls
- **Embedding**: Unique pattern of 1024 numbers

- **You can't "read" a fingerprint**: But you can match similar ones
- **You can't "read" an embedding**: But you can find semantically similar text

- **Fingerprint matching**: Compare ridge patterns
- **Embedding matching**: Compare vector patterns (cosine similarity)

- **Similar people ≠ identical fingerprints**: But family members may have some similarities
- **Similar texts ≠ identical embeddings**: But related texts have high cosine similarity

## Key Insights

1. **Semantic meaning in numbers**: The 1024-dimensional vector encodes what the text "means"

2. **Learned representations**: The neural network learns these weights during training on massive text corpora

3. **Dimensionality = expressiveness**: More dimensions → more nuanced semantic representation (but diminishing returns after ~1000-2000)

4. **Distance = similarity**: Geometric distance in vector space corresponds to semantic similarity in meaning

5. **Not interpretable individually**: You cannot say "dimension 42 represents IP addresses" - the meaning is distributed across all dimensions

6. **Deterministic**: Same text → same embedding vector (for a given model)

7. **Language-agnostic (for multilingual models)**: Models like bge-m3 can embed multiple languages into the same vector space

## Performance Metrics

From the log example (line 910):

```
total_duration=2302761449 (2.30 seconds)
load_duration=1957583950 (1.96 seconds)
prompt_eval_count=306 (tokens processed)
```

- **Total time**: 2.30 seconds to generate 1 embedding
- **Model loading**: 1.96 seconds (first call, subsequent calls much faster)
- **Token count**: 306 tokens processed from input text
- **Output**: 1 vector × 1024 dimensions = 1,024 float32 values (~4KB)

## Use Cases in LightRAG

Embeddings are generated for:

1. **Chunks**: Document segments for retrieval
   - Input: "Client MAC: 11:22:33:44:55:66, IP: 192.168.1.105"
   - Stored in: `chunks` namespace

2. **Entities**: Extracted entities from knowledge graph
   - Input: "CLIENT_A" (entity name + description)
   - Stored in: `entities` namespace

3. **Relationships**: Entity relationships from knowledge graph
   - Input: "CLIENT_A -> CONNECTED_TO -> ROUTER_B"
   - Stored in: `relationships` namespace

4. **Queries**: User questions for similarity search
   - Input: "What is the client's IP address?"
   - Used for: Finding similar chunks/entities/relationships

## Further Reading

- **Vector databases**: How embeddings are stored and searched efficiently
- **Cosine similarity**: Mathematical details of similarity calculation
- **HNSW (Hierarchical Navigable Small World)**: Approximate nearest neighbor search algorithm
- **Embedding models**: BERT, Sentence Transformers, BGE, OpenAI embeddings

## References

- BGE-M3 model: https://huggingface.co/BAAI/bge-m3
- OpenAI embeddings: https://platform.openai.com/docs/guides/embeddings
- Sentence Transformers: https://www.sbert.net/
- Vector similarity search: https://www.pinecone.io/learn/vector-similarity/
