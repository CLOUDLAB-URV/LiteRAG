"""
LiteRAG Engine Adapter

Wraps LiteRAG in the standard SearchEngineInterface for benchmarking.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any

from .engine_interface import SearchEngineInterface, QueryResult, EngineType

LOGGER = logging.getLogger(__name__)


class LiteRAGEngine(SearchEngineInterface):
    """
    LiteRAG (Adaptive Graph-Native Search) adapter.
    
    Wraps the LiteRAG engine to conform to the benchmark interface.
    Supports loading configuration from YAML file.
    """
    
    def __init__(
        self,
        config_path: Path,
        prompt_price_per_m: float = 0.10,
        completion_price_per_m: float = 0.40,
        **literag_kwargs
    ):
        """
        Initialize LiteRAG adapter.
        
        Args:
            config_path: Path to LiteRAG YAML configuration file
            prompt_price_per_m: Price per million prompt tokens
            completion_price_per_m: Price per million completion tokens
            **literag_kwargs: Additional LiteRAG configuration overrides
        """
        self.config_path = Path(config_path)
        self.prompt_price_per_m = prompt_price_per_m
        self.completion_price_per_m = completion_price_per_m
        self.literag_kwargs = literag_kwargs
        
        self._engine = None
        self._initialized = False
    
    @property
    def name(self) -> str:
        return "LiteRAG"
    
    @property
    def engine_type(self) -> EngineType:
        return EngineType.LITERAG
    
    async def initialize(self) -> None:
        """Initialize LiteRAG engine"""
        if self._initialized:
            return
        
        # Import LiteRAG
        try:
            from literag import LiteRAG, LiteRAGConfig
        except ImportError as e:
            raise ImportError(
                "LiteRAG is not installed or not importable. Install LiteRAG in your environment "
                "(for local development, use: pip install -e /path/to/literag)."
            ) from e
        
        # Load config from YAML file
        if not self.config_path.exists():
            raise ValueError(f"LiteRAG config file not found: {self.config_path}")
        
        config = LiteRAGConfig.from_yaml(str(self.config_path))
        
        # Apply any runtime overrides from literag_kwargs
        for key, value in self.literag_kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self._engine = LiteRAG(config)
        self._engine.initialize()
        
        self._initialized = True
    
    async def search(self, query: str) -> QueryResult:
        """Execute LiteRAG search"""
        start = time.perf_counter()
        
        try:
            result = await self._engine.aquery(query)
            latency = time.perf_counter() - start
            
            # Use real token counts from LLM service (no estimation)
            prompt_tokens = result.prompt_tokens
            completion_tokens = result.completion_tokens
            total_tokens = prompt_tokens + completion_tokens
            
            cost = self.calculate_cost(
                prompt_tokens,
                completion_tokens,
                self.prompt_price_per_m,
                self.completion_price_per_m
            )
            
            return QueryResult(
                engine=self.engine_type.value,
                prompt_name="",
                prompt_text=query,
                answer=result.answer,
                context_text=result.context,
                latency_seconds=round(latency, 4),
                llm_calls=result.llm_calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_eur=cost,
                extra_info={
                    "anchors": len(result.anchors),
                    "entities_explored": result.total_entities_explored,
                    "entities_in_context": result.entities_in_context,
                    "expanded_query": result.expanded_query,
                    "context_tokens": result.tokens_used,  # Context assembly tokens
                }
            )
            
        except Exception as e:
            if LOGGER.isEnabledFor(logging.INFO):
                LOGGER.exception("LiteRAG search failed")
            latency = time.perf_counter() - start
            return QueryResult(
                engine=self.engine_type.value,
                prompt_name="",
                prompt_text=query,
                answer="",
                latency_seconds=round(latency, 4),
                error=str(e),
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LiteRAG statistics"""
        if not self._initialized or self._engine is None:
            return {"engine": self.name, "initialized": False}
        
        stats = self._engine.get_stats()
        stats["engine"] = self.name
        return stats
