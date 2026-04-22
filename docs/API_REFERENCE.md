# API Reference

## Package Exports

The package exports LiteRAG, LiteRAGConfig, result/data models, and service classes from LiteRAG/__init__.py.

## LiteRAGConfig

Location: LiteRAG/config.py

Important fields:

- data_dir: graph data directory.
- api_key: required unless use_litellm is true.
- llm_model and embedding_model.
- anchor, exploration, consensus, and context tuning parameters.

Constructors:

- LiteRAGConfig(...): create programmatically.
- LiteRAGConfig.from_yaml(path): load YAML and expand ${ENV_VAR} references.

Validation behavior:

- If lancedb_uri is omitted, LiteRAG tries common paths under data_dir.
- If use_litellm is false and api_key is missing, initialization raises ValueError.

## LiteRAG

Location: LiteRAG/engine.py

### LiteRAG(config)

Creates the orchestrator with a validated LiteRAGConfig.

### initialize(force_reload=False)

Initializes all runtime components:

- Vector store
- Graph data
- Embedding service
- LLM service
- Anchor discovery and embedding availability
- Parallel explorer warmup
- Merger and context assembler

### query(query, expand_query=True) -> LiteRAGResult

Synchronous query method.

- Requires initialize to be called first.
- Runs full pipeline and returns LiteRAGResult.

### aquery(query, expand_query=True) -> LiteRAGResult

Async query method.

- Uses native async exploration path.
- Recommended inside async applications and servers.

### get_stats() -> dict

Returns initialized status and basic graph and LLM usage stats.

## LiteRAGResult

Location: LiteRAG/models.py

Primary fields:

- query, answer
- success, error
- anchors, subgraphs, ranked_entities
- context
- tokens_used, prompt_tokens, completion_tokens, llm_calls, latency_ms

Method:

- to_dict(): summary serialization of key metrics.

## Data Models

Location: LiteRAG/models.py

- Entity: graph node plus centrality and text-unit references.
- Relationship: graph edge with weight and description.
- Community: cluster metadata and summary fields.
- Anchor: discovered starting entity with source metadata.
- ExploredNode: traversal observation and path metadata.
- Subgraph: per-anchor exploration result.
- RankedEntity: final merged/ranked entity with scoring breakdown.

## Service Classes

- EmbeddingService: embeddings, cache, cosine similarity.
- LiteRAGVectorStore: LanceDB access and similarity search.
- AnchorDiscovery: multi-strategy anchor selection and score fusion.
- ParallelExplorer: parallel traversal orchestration.
- ConsensusMerger: merge and rank explored nodes.
- ContextAssembler: prompt context construction.
- LLMService: query expansion and final completion calls.
