# Troubleshooting

## LiteRAG not initialized

Symptom:

- RuntimeError from query or aquery.

Fix:

- Call engine.initialize() before query execution.

## Missing required parquet files

Symptom:

- ValueError indicating entities or relationships data could not be loaded.

Fix:

- Verify data_dir points to directory containing entities.parquet and relationships.parquet.

## api_key is required when not using LiteLLM proxy

Symptom:

- ValueError during LiteRAGConfig initialization.

Fix:

- Set api_key or GEMINI_API_KEY.
- Or set use_litellm to true and ensure proxy credentials are configured.

## LanceDB connection or table issues

Symptom:

- Similarity search returns empty results or logs table errors.

Fix:

- Confirm lancedb path exists and is readable.
- Confirm expected table names exist.
- Allow LiteRAG to compute missing embeddings on first run.

## No anchors found

Symptom:

- Result has no anchors and answer indicates no relevant information.

Fix:

- Lower min_anchor_score.
- Increase max_anchors.
- Ensure entity descriptions and titles are meaningful.
- Confirm embedding model/provider is working.

## Slow query performance

Fix options:

- Reduce max_exploration_depth.
- Reduce max_nodes_per_anchor.
- Reduce max_neighbors_per_hop.
- Increase num_exploration_workers if CPU allows.
- Disable query expansion if not needed.

## High token usage or long prompts

Fix options:

- Lower max_context_tokens.
- Lower max_ranked_entities.
- Disable or reduce safety_net_k.

## Running inside async server

Symptom:

- Runtime error related to running event loop when using sync explore path.

Fix:

- Use await engine.aquery(...) in async contexts.
