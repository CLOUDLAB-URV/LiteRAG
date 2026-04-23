# ⚙️ Configuration Guide

LiteRAG is highly configurable to suit different dataset sizes and query types. Configuration is managed via the `LiteRAGConfig` class and can be loaded directly from a YAML file (`literag_config.yaml`).

> **💡 Pro Tip:** You can use environment variables in your YAML file like this: `api_key: ${GEMINI_API_KEY}`.

## 1. Core & Model Settings

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data_dir` | `str` | `"./example-data/output"` | Path to the directory containing GraphRAG parquet outputs. |
| `api_key` | `str` | `null` | Your LLM API key (Required unless using LiteLLM). |
| `llm_model` | `str` | `"gemini/gemini-flash-lite-latest"` | The model used for generation and query expansion. |
| `embedding_model` | `str` | `"models/gemini-embedding-001"` | The model used for semantic vectorization. |
| `use_litellm` | `bool` | `false` | Set to true to route traffic through a LiteLLM proxy. |

## 2. Phase 1: Anchor Discovery

Controls how LiteRAG finds its starting points in the graph.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `max_anchors` | `8` | Maximum number of starting nodes to select. |
| `min_anchor_score` | `0.3` | Minimum confidence score required to become an anchor. |
| `semantic_weight` | `0.40` | Importance of LanceDB vector similarity. |
| `keyword_exact_weight` | `0.30` | Importance of exact N-gram matches (BM25). |
| `keyword_fuzzy_weight` | `0.15` | Importance of fuzzy matching (Levenshtein). |
| `community_weight` | `0.15` | Importance of community-level relevance. |

## 3. Phase 2: Graph Exploration (Zero-LLM Traversal)

These are the most critical parameters for tuning latency and accuracy.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `max_exploration_depth` | `3` | Maximum number of hops away from an anchor. |
| `min_relevance_threshold`| `0.25`| The base floor for the Dynamic Semantic Threshold. |
| `relevance_decay_factor` | `0.7` | How much semantic relevance degrades per hop. |
| `max_nodes_per_anchor` | `50` | Hard cap on nodes explored per worker thread. |
| `degree_influence` | `0.05` | The penalty applied to high-degree hubs. Higher = stricter filtering. |
| `community_cohesion_weight`| `0.8` | Protection factor for Topic Hubs. `1.0` = fully protected. |

## 4. Phase 3: Consensus & Context Assembly

Controls what actually gets sent to the LLM.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `max_ranked_entities` | `50` | Maximum number of entities allowed in the final context window. |
| `intersection_weight` | `0.45` | Reward for nodes found by multiple independent anchor paths. |
| `max_context_tokens` | `10000` | The absolute token budget for the final LLM prompt. |
| `enable_safety_net` | `true` | Injects raw text units for exact string references as a fallback. |

---

## 🛠️ Recommended Tuning Profiles

### Profile A: "Need for Speed" (Massive Graphs)
If you have millions of nodes and need sub-second responses:
* `max_anchors`: 4
* `max_exploration_depth`: 2
* `max_nodes_per_anchor`: 20
* `degree_influence`: 0.10 *(Filter hubs aggressively)*

### Profile B: "Deep Detective" (Complex Multi-Hop)
If your queries require synthesizing data across widely disjointed concepts:
* `max_anchors`: 12
* `max_exploration_depth`: 4
* `min_relevance_threshold`: 0.15 *(Open the door to wider exploration)*
* `intersection_weight`: 0.60 *(Heavily favor nodes where paths intersect)*