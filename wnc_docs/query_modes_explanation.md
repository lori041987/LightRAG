# LightRAG Query Modes Explanation

This document explains the different query modes available in LightRAG and when to use each one.

## Quick Reference

| Mode | Focus | Best For | Speed |
|------|-------|----------|-------|
| **naive** | Vector search only | Simple similarity search | Fastest |
| **local** | Entities (nodes) | "What/Who is X?" questions | Fast |
| **global** | Relationships (edges) | "How are things connected?" | Fast |
| **hybrid** | Entities + Relationships | Most real-world questions | Medium (recommended) |
| **mix** | KG + Vector integration | Alternative retrieval approach | Medium |
| **bypass** | No retrieval | Direct LLM only (testing) | Very fast |

---

## Detailed Explanations with Examples

### Scenario
You have a knowledge base about a company with employees, departments, and projects indexed in LightRAG.

---

### **"naive" mode** - Basic Vector Search
- **How it works**: Simple vector similarity search on text chunks, no knowledge graph
- **Best for**: Quick similarity-based retrieval without graph traversal
- **Limitations**: Doesn't use entity/relationship extraction
- **Speed**: Fastest
- **Example queries**:
  - "Find documents about network security"
  - "What mentions IP addresses?"

**Use case**: When you want simple, fast vector search without graph reasoning.

---

### **"local" mode** - Entity-Focused Retrieval
- **How it works**:
  1. Extracts keywords from query
  2. Searches for matching **entities** (nodes) in the knowledge graph
  3. Retrieves entity properties and nearby relationships
- **Retrieves**: `top_k` entities (default: 60)
- **Best for**: Questions about **specific entities and their direct attributes**
- **Example queries**:
  - "What is John's role?" → finds entity "John" and its properties
  - "What is the Marketing department?" → finds "Marketing" entity details
  - "What is the client's IP address?" → finds "client" entity with IP property
  - "Who is Alice?" → finds "Alice" entity and her attributes

**Use case**: "What/Who is X?" questions where you need information about a specific thing or person.

---

### **"global" mode** - Relationship-Focused Retrieval
- **How it works**:
  1. Extracts high-level keywords from query
  2. Searches for matching **relationships** (edges) in the knowledge graph
  3. Retrieves relationship patterns and connected entities
- **Retrieves**: `top_k` relationships (default: 60)
- **Best for**: Questions about **connections, patterns, and organizational structure**
- **Example queries**:
  - "Who reports to whom?" → finds "reports_to" relationships
  - "What projects involve multiple departments?" → finds cross-department relationship patterns
  - "How do services communicate?" → finds "communicates_with" relationships
  - "What are the dependencies between systems?" → finds dependency relationships

**Use case**: "How are things connected?" questions about patterns and relationships.

---

### **"hybrid" mode** - Combined Entity + Relationship Retrieval (Recommended)
- **How it works**: Combines both local (entities) AND global (relationships)
- **Retrieves**: Both entities AND relationships based on query
- **Best for**: Most real-world questions that need both facts and context
- **Example queries**:
  - "What does John work on?" → needs John's entity info AND his relationships to projects
  - "How is the system architecture organized?" → needs components (entities) AND their connections (relationships)
  - "What are the client's network settings?" → needs client entity + network configuration relationships

**Use case**: General-purpose queries (recommended default). Provides most comprehensive results.

---

### **"mix" mode** - Knowledge Graph + Vector Retrieval Integration
- **How it works**: Alternative approach that integrates knowledge graph with vector retrieval
- **Best for**: Balancing graph-based and similarity-based retrieval
- **Difference from hybrid**: Different integration strategy for combining KG and vector search

**Use case**: Alternative to hybrid when you want a different balance of graph vs. vector retrieval.

---

### **"bypass" mode** - Direct LLM Call Only
- **How it works**: Skips all retrieval (no KG, no vector search), sends query directly to LLM
- **No retrieval**: LLM answers purely from its training data, without your documents
- **Best for**:
  - Testing/comparison against retrieval modes
  - Questions that don't require your specific documents
  - Debugging to isolate LLM vs. retrieval issues

**Use case**: Testing and debugging only. Not useful for actual RAG queries.

---

## Configuration Parameters

### `top_k` (default: 60)
- **In local mode**: Number of entities to retrieve
- **In global mode**: Number of relationships to retrieve
- **In hybrid/mix**: Applies to both entities and relationships

### `chunk_top_k` (default: 20)
- Number of text chunks to retrieve from vector search
- Used in all modes except bypass
- Independent of knowledge graph retrieval

### Example Configuration

```python
from lightrag import QueryParam

# Local mode - focus on entities
param = QueryParam(mode="local", top_k=60)

# Global mode - focus on relationships
param = QueryParam(mode="global", top_k=60)

# Hybrid mode - comprehensive (recommended)
param = QueryParam(mode="hybrid", top_k=60, chunk_top_k=20)
```

---

## When to Use Each Mode

| Your Question Type | Recommended Mode | Why |
|-------------------|------------------|-----|
| "What is X?" | local | Needs entity information |
| "Who is Y?" | local | Needs person/entity details |
| "How do X and Y relate?" | global | Needs relationship information |
| "What's the structure of...?" | global | Needs pattern/connection info |
| "Tell me about X's involvement in Y" | hybrid | Needs both entity + relationships |
| "General question about topic" | hybrid | Most comprehensive |
| "Quick similarity search" | naive | Fast, simple |
| "Test without retrieval" | bypass | Debugging/testing only |

---

## Performance Considerations

- **Fastest**: naive, bypass
- **Fast**: local, global
- **Medium** (recommended): hybrid, mix
- **Quality vs Speed**: Hybrid provides best quality but takes longer than local/global alone

For production use, **hybrid mode** is recommended as the default for best results.
