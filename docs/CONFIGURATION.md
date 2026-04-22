# Configuration Guide

LiteRAG configuration is defined in LiteRAGConfig and can be supplied via Python or YAML.

## YAML Loading

Use LiteRAGConfig.from_yaml("literag_config.yaml").

The loader supports environment variable expansion in this form:

- ${VAR_NAME}

## Core Fields

### Required

- data_dir (str, default: ./data)
- api_key (str or null)

Note: api_key is required when use_litellm is false.

### Model and Provider

- llm_model (default: gemini/gemini-flash-lite-latest)
- embedding_model (default: models/gemini-embedding-001)
- use_litellm (default: false)
- litellm_base_url (default: <http://localhost:4000>)
- lancedb_uri (optional; auto-detected if omitted)

### Query Processing

- enable_query_expansion (default: false)

### Anchor Discovery

- max_anchors (default: 8)
- min_anchor_score (default: 0.3)
- semantic_weight (default: 0.40)
- keyword_exact_weight (default: 0.30)
- keyword_fuzzy_weight (default: 0.15)
- community_weight (default: 0.15)

### Exploration

- max_exploration_depth (default: 3)
- min_relevance_threshold (default: 0.25)
- relevance_decay_factor (default: 0.7)
- max_nodes_per_anchor (default: 50)
- max_neighbors_per_hop (default: 10)
- num_exploration_workers (default: 4)
- degree_influence (default: 0.05)
- community_cohesion_weight (default: 0.8)
- signal_amplification_factor (default: 0.5)

### Safety Net

- enable_safety_net (default: true)
- safety_net_k (default: 5)

### Consensus

- max_ranked_entities (default: 50)
- intersection_weight (default: 0.45)
- consensus_semantic_weight (default: 0.40)
- structural_weight (default: 0.05)
- proximity_weight (default: 0.10)
- anchor_boost (default: 1.2)

### Context and Generation

- max_context_tokens (default: 10000)
- response_temperature (default: 0.7)
- max_response_tokens (default: 2000)
- embedding_batch_size (default: 100)

## Recommended Tuning by Graph Size

Small graph:

- max_anchors: 5
- max_exploration_depth: 2
- max_nodes_per_anchor: 30

Medium graph:

- Use defaults as starting point.

Large graph:

- max_anchors: 8 to 12
- max_neighbors_per_hop: 6 to 10
- increase num_exploration_workers based on CPU capacity

## Secret Management

- Keep api_key in environment variables.
- Do not commit .env files.
- Keep literag_config.yaml free of raw secrets.
