"""
Search Engine Interface

Abstract base class that all search engines must implement.
This ensures GraphRAG and LiteRAG (and future engines) have a common interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class EngineType(Enum):
    """Supported engine types"""
    GRAPHRAG_BASIC = "graphrag_basic"
    GRAPHRAG_LOCAL = "graphrag_local"
    GRAPHRAG_GLOBAL = "graphrag_global"
    GRAPHRAG_DRIFT = "graphrag_drift"
    LITERAG = "literag"
    # LightRAG query modes
    LIGHTRAG_LOCAL = "lightrag_local"
    LIGHTRAG_GLOBAL = "lightrag_global"
    LIGHTRAG_HYBRID = "lightrag_hybrid"
    LIGHTRAG_MIX = "lightrag_mix"
    LIGHTRAG_NAIVE = "lightrag_naive"
    # Google FileSearch
    GOOGLE_FILESEARCH = "google_filesearch"
    # HiRAG query modes (EMNLP 2025)
    HIRAG_HI        = "hirag_hi"
    HIRAG_LOCAL     = "hirag_local"
    HIRAG_GLOBAL    = "hirag_global"
    HIRAG_BRIDGE    = "hirag_bridge"
    HIRAG_NOBRIDGE  = "hirag_nobridge"
    HIRAG_NAIVE     = "hirag_naive"


@dataclass
class QueryResult:
    """
    Standardized result from any search engine.
    
    All engines must return results in this format for fair comparison.
    """
    # Identification
    engine: str
    prompt_name: str
    prompt_text: str
    
    # Response
    answer: str
    context_text: Optional[str] = None
    
    # Performance metrics
    latency_seconds: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_eur: float = 0.0
    
    # Ground truth for evaluation
    ground_truth: Optional[str] = None
    
    # Quality metrics (populated by MetricsCalculator)
    answer_accuracy: Optional[float] = None
    semantic_similarity: Optional[float] = None
    rougeL_f1: Optional[float] = None
    context_relevance: Optional[float] = None
    llm_judged_correctness: Optional[float] = None
    llm_judged_completeness: Optional[float] = None
    llm_judged_relevance: Optional[float] = None
    llm_judged_reasoning: Optional[str] = None
    
    # Metadata
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Engine-specific metadata
    extra_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        d = asdict(self)
        d['is_success'] = self.is_success  # Add computed property
        return d
    
    @property
    def is_success(self) -> bool:
        return self.error is None and bool(self.answer)


class SearchEngineInterface(ABC):
    """
    Abstract interface for search engines.
    
    Both GraphRAG and LiteRAG must implement this interface.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the engine"""
        pass
    
    @property
    @abstractmethod
    def engine_type(self) -> EngineType:
        """Type identifier for the engine"""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the engine (load models, data, etc.)
        
        Should be called once before queries.
        """
        pass
    
    @abstractmethod
    async def search(self, query: str) -> QueryResult:
        """
        Execute a search query.
        
        Args:
            query: The question to answer
            
        Returns:
            QueryResult with answer and metrics
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics (entities loaded, etc.)"""
        pass
    
    async def close(self) -> None:
        """
        Clean up resources (close connections, delete temp stores, etc.)
        
        Optional implementation in adapters.
        """
        pass
    
    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        prompt_price_per_m: float = 0.10,
        completion_price_per_m: float = 0.40
    ) -> float:
        """Calculate cost in EUR (shared utility)"""
        cost = (
            (prompt_tokens / 1_000_000) * prompt_price_per_m +
            (completion_tokens / 1_000_000) * completion_price_per_m
        )
        return round(cost, 6)
