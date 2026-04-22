"""
LiteRAG - Adaptive Graph-Native Search

A unified, embedding-guided query engine for GraphRAG that uses the query's
semantic vector as a compass to guide graph traversal, with depth and breadth
emerging naturally from data relevance.

Key Features:
- Zero-LLM Navigation: LLM only used for final response
- Adaptive Depth: Exploration depth emerges from relevance decay
- Unified Engine: One system handles all query types
- Minimal LLM Usage: 1-2 LLM calls total

Usage:
    from LiteRAG import LiteRAG, LiteRAGConfig
    
    config = LiteRAGConfig(
        data_dir="./data",
        api_key="your-api-key"
    )
    
    engine = LiteRAG(config)
    engine.initialize()
    
    result = engine.query("What technologies enable edge computing for IoT?")
    print(result.answer)
"""

from .config import LiteRAGConfig
from .engine import LiteRAG
from .models import (
    LiteRAGResult,
    Entity,
    Relationship,
    Community,
    Anchor,
    AnchorSource,
    RankedEntity,
    Subgraph,
    ExploredNode
)
from .data_loader import GraphData, load_graph_data
from .embeddings import EmbeddingService
from .llm_service import LLMService
from .vector_store import LiteRAGVectorStore

__version__ = "1.0.0"

__all__ = [
    # Main classes
    "LiteRAG",
    "LiteRAGConfig",
    
    # Result types
    "LiteRAGResult",
    
    # Data models
    "Entity",
    "Relationship", 
    "Community",
    "Anchor",
    "AnchorSource",
    "RankedEntity",
    "Subgraph",
    "ExploredNode",
    
    # Data loading
    "GraphData",
    "load_graph_data",
    
    # Services
    "EmbeddingService",
    "LLMService",
    
    # Vector Store
    "LiteRAGVectorStore",
]
