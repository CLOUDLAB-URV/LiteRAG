"""
Data Models for LiteRAG

Defines the core data structures used throughout the system.
Uses dataclasses for clean, typed representations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import numpy as np


class AnchorSource(Enum):
    """How an anchor was discovered."""
    SEMANTIC = "semantic"
    KEYWORD_EXACT = "keyword_exact"
    KEYWORD_FUZZY = "keyword_fuzzy"
    COMMUNITY = "community"


@dataclass
class Entity:
    """
    Represents a node in the knowledge graph.
    """
    id: str
    title: str
    description: str
    type: str = ""
    degree: int = 0
    community_id: Optional[str] = None
    
    # Pre-computed metrics (populated during indexing)
    embedding: Optional[np.ndarray] = None
    pagerank: float = 0.0
    betweenness_centrality: float = 0.0
    
    # Text unit references
    text_unit_ids: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.id == other.id
        return False


@dataclass
class Relationship:
    """
    Represents an edge in the knowledge graph.
    """
    id: str
    source: str  # Entity title
    target: str  # Entity title
    description: str
    weight: float = 1.0
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Community:
    """
    Represents a community (cluster) of entities.
    """
    id: str
    title: str
    level: int
    summary: str
    full_content: str
    entity_ids: List[str] = field(default_factory=list)
    
    # Pre-computed
    embedding: Optional[np.ndarray] = None
    hub_entities: List[str] = field(default_factory=list)  # Most central entities
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Anchor:
    """
    A starting point for graph exploration.
    """
    entity_id: str
    entity_title: str
    score: float
    sources: Set[AnchorSource] = field(default_factory=set)
    
    def add_source(self, source: AnchorSource, score: float):
        """Add a discovery source and update score if higher."""
        self.sources.add(source)
        self.score = max(self.score, score)
    
    def __hash__(self):
        return hash(self.entity_id)
    
    def __eq__(self, other):
        if isinstance(other, Anchor):
            return self.entity_id == other.entity_id
        return False


@dataclass
class ExploredNode:
    """
    A node discovered during graph exploration.
    """
    entity: Entity
    relevance: float
    depth: int
    semantic_similarity: float
    path_from_anchor: List[str] = field(default_factory=list)
    discovered_by_anchors: Set[str] = field(default_factory=set)


@dataclass
class Subgraph:
    """
    Result of exploration from a single anchor.
    """
    anchor: Anchor
    nodes: Dict[str, ExploredNode]  # entity_id -> ExploredNode
    max_depth_reached: int = 0
    tokens_used: int = 0


@dataclass
class RankedEntity:
    """
    An entity with its final consensus score.
    """
    entity: Entity
    final_score: float
    
    # Score components for debugging/analysis
    intersection_score: float = 0.0
    semantic_score: float = 0.0
    structural_score: float = 0.0
    proximity_score: float = 0.0
    
    # Metadata
    found_by_workers: int = 0
    is_anchor: bool = False
    avg_depth: float = 0.0
    
    # Related data for context assembly
    relationships: List[Relationship] = field(default_factory=list)
    community: Optional[Community] = None


@dataclass
class LiteRAGResult:
    """
    Complete result from an LiteRAG query.
    """
    query: str
    answer: str
    
    # State tracking
    success: bool = True
    error: Optional[str] = None
    
    # Intermediate results for debugging/analysis
    expanded_query: Optional[str] = None
    anchors: List[Anchor] = field(default_factory=list)
    subgraphs: List[Subgraph] = field(default_factory=list)
    ranked_entities: List[RankedEntity] = field(default_factory=list)
    context: str = ""
    
    # Metrics
    total_entities_explored: int = 0
    entities_in_context: int = 0
    tokens_used: int = 0  # Context tokens
    prompt_tokens: int = 0  # LLM prompt tokens
    completion_tokens: int = 0  # LLM completion tokens
    llm_calls: int = 0
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "answer": self.answer,
            "success": self.success,
            "error": self.error,
            "expanded_query": self.expanded_query,
            "num_anchors": len(self.anchors),
            "total_entities_explored": self.total_entities_explored,
            "entities_in_context": self.entities_in_context,
            "tokens_used": self.tokens_used,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "llm_calls": self.llm_calls,
            "latency_ms": self.latency_ms,
        }
