"""
LiteRAG Configuration Module
"""

from dataclasses import dataclass, field
from typing import Optional
import os
import re


@dataclass
class LiteRAGConfig:
    # ─────────────────────────────────────────────────────────────────────────
    # Required Configuration
    # ─────────────────────────────────────────────────────────────────────────
    data_dir: str = "./data"
    api_key: Optional[str] = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Model Configuration
    # ─────────────────────────────────────────────────────────────────────────
    llm_model: str = "gemini/gemini-flash-lite-latest"
    embedding_model: str = "models/gemini-embedding-001"
    use_litellm: bool = False
    litellm_base_url: str = "http://localhost:4000"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Vector Store
    # ─────────────────────────────────────────────────────────────────────────
    lancedb_uri: Optional[str] = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Query Processing
    # ─────────────────────────────────────────────────────────────────────────
    enable_query_expansion: bool = False
    
    # ─────────────────────────────────────────────────────────────────────────
    # Anchor Discovery
    # ─────────────────────────────────────────────────────────────────────────
    max_anchors: int = 8
    min_anchor_score: float = 0.3
    semantic_weight: float = 0.40
    keyword_exact_weight: float = 0.30
    keyword_fuzzy_weight: float = 0.15
    community_weight: float = 0.15
    
    # ─────────────────────────────────────────────────────────────────────────
    # Anchor Discovery (Phase 1)
    # ─────────────────────────────────────────────────────────────────────────
    max_anchors: int = 8
    min_anchor_score: float = 0.30           # tau_anchor
    semantic_weight: float = 0.40            # alpha
    keyword_exact_weight: float = 0.30       # beta (exact)
    keyword_fuzzy_weight: float = 0.15       # beta (fuzzy)
    community_weight: float = 0.15           # gamma
    
    # ─────────────────────────────────────────────────────────────────────────
    # Graph Exploration (Phase 2)
    # ─────────────────────────────────────────────────────────────────────────
    max_exploration_depth: int = 3           # k_max
    min_relevance_threshold: float = 0.25    # tau_base
    relevance_decay_factor: float = 0.70     # d
    max_nodes_per_anchor: int = 50           # N_max
    max_neighbors_per_hop: int = 10
    num_exploration_workers: int = 4
    
    degree_influence: float = 0.05           # delta
    community_cohesion_weight: float = 0.80  # kappa
    signal_amplification_factor: float = 0.25 # lambda
    
    # ─────────────────────────────────────────────────────────────────────────
    # Consensus Ranking (Phase 3)
    # ─────────────────────────────────────────────────────────────────────────
    max_ranked_entities: int = 50
    
    # Weights shifted to match paper's grid search (Sum = 1.0)
    intersection_weight: float = 0.30        # rho_1
    consensus_semantic_weight: float = 0.25  # rho_2
    proximity_weight: float = 0.25           # rho_3
    structural_weight: float = 0.20          # rho_4
    
    anchor_boost: float = 1.5                # Paper specifies 1.5

    # ─────────────────────────────────────────────────────────────────────────
    # Signal-Dependent Exploration
    # ─────────────────────────────────────────────────────────────────────────
    # How much to trust the community structure?
    # 1.0 = Fully protect hubs in same community. 0.0 = Treat all hubs as noise.
    community_cohesion_weight: float = 0.8
    
    # How much does the initial signal strength dictate strictness?
    # Higher = If anchor is good, we become very strict (narrow search).
    signal_amplification_factor: float = 0.5
    
    # ─────────────────────────────────────────────────────────────────────────
    # Safety Net (Hybrid Search)
    # ─────────────────────────────────────────────────────────────────────────
    enable_safety_net: bool = True          # Use direct vector search on text units
    safety_net_k: int = 5                   # Number of direct chunks to inject
    
    # ─────────────────────────────────────────────────────────────────────────
    # Context Assembly
    # ─────────────────────────────────────────────────────────────────────────
    max_context_tokens: int = 10000
    response_temperature: float = 0.7
    max_response_tokens: int = 2000
    embedding_batch_size: int = 100
    
    def __post_init__(self):
        if self.lancedb_uri is None:
            from pathlib import Path
            possible_paths = [
                Path(self.data_dir) / "lancedb",
                Path(self.data_dir) / "output" / "lancedb",
            ]
            for path in possible_paths:
                if path.exists():
                    self.lancedb_uri = str(path)
                    break
            else:
                self.lancedb_uri = os.path.join(self.data_dir, "lancedb")
        
        if not self.use_litellm and not self.api_key:
            raise ValueError("api_key is required when not using LiteLLM proxy.")

    @classmethod
    def from_yaml(cls, path: str) -> "LiteRAGConfig":
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        if data is None: data = {}
        
        # Simple env var expansion
        def expand_env_vars(value):
            if isinstance(value, str):
                pattern = r'\$\{([^}]+)\}'
                def replace(match):
                    return os.environ.get(match.group(1), '')
                return re.sub(pattern, replace, value)
            return value
            
        processed_data = {k: expand_env_vars(v) for k, v in data.items()}
        # Clean up nulls
        processed_data = {k: v for k, v in processed_data.items() if v is not None}
        
        return cls(**processed_data)