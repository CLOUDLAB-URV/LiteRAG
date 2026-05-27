"""
LightRAG Engine Adapter

Wraps LightRAG in the standard SearchEngineInterface for benchmarking.
Supports all LightRAG query modes: local, global, hybrid, mix, naive.
"""

import os
import time
import yaml
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass

from .engine_interface import SearchEngineInterface, QueryResult, EngineType

LOGGER = logging.getLogger(__name__)


@dataclass
class LightRAGConfig:
    """Configuration for LightRAG engine"""
    # Required paths
    working_dir: str
    
    # Model configuration
    llm_model: str = "openai/gpt-4o-mini"
    embedding_model: str = "openai/text-embedding-3-small"
    api_key: str = ""
    api_base: Optional[str] = None
    
    # Query parameters
    top_k: int = 60
    max_entity_tokens: int = 6000
    max_relation_tokens: int = 8000
    max_total_tokens: int = 30000
    
    # Storage backends
    kv_storage: str = "JsonKVStorage"
    vector_storage: str = "NanoVectorDBStorage"
    graph_storage: str = "NetworkXStorage"
    doc_status_storage: str = "JsonDocStatusStorage"
    
    # Optional features
    enable_rerank: bool = False
    
    @classmethod
    def from_yaml(cls, path: str) -> "LightRAGConfig":
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k) or k in cls.__dataclass_fields__})


def _resolve_env_ref(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return value
    if value.startswith("os.environ/"):
        return os.getenv(value.split("/", 1)[1], "")
    return value


class TokenTracker:
    """Thread-safe token usage tracker for LLM calls"""
    
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_calls = 0
        self._lock = asyncio.Lock()
    
    async def add(self, prompt_tokens: int, completion_tokens: int):
        async with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.llm_calls += 1
    
    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_calls = 0
    
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def create_litellm_complete_func(
    model: str,
    api_base: Optional[str],
    api_key: str,
    token_tracker: TokenTracker
):
    """
    Create a LiteLLM-compatible completion function for LightRAG.
    
    This wrapper intercepts LLM calls to track token usage and routes
    requests through the LiteLLM proxy if configured.
    """
    import litellm
    
    # Valid kwargs to pass to litellm.acompletion
    VALID_LLM_KWARGS = {
        'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 
        'presence_penalty', 'stop', 'n', 'stream', 'logprobs',
        'timeout', 'user', 'response_format', 'seed', 'tools',
        'tool_choice', 'logit_bias', 'top_logprobs'
    }
    
    async def llm_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: list = [],
        **kwargs
    ) -> str:
        """Async completion function compatible with LightRAG"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        
        # Filter kwargs to only include valid LLM parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in VALID_LLM_KWARGS}
        
        # Configure LiteLLM
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            api_base=api_base,
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            **filtered_kwargs
        )
        
        # Track tokens
        usage = response.usage
        await token_tracker.add(
            prompt_tokens=usage.prompt_tokens or 0,
            completion_tokens=usage.completion_tokens or 0
        )
        
        return response.choices[0].message.content
    
    return llm_complete


def create_litellm_embed_func(
    model: str,
    api_base: Optional[str],
    api_key: str,
    token_tracker: TokenTracker
):
    """
    Create a LiteLLM-compatible embedding function for LightRAG.
    Returns an async function that produces numpy arrays.
    
    LightRAG requires exactly one embedding per input text.
    """
    import litellm
    import numpy as np
    import os
    
    async def embed_func(texts: list[str]) -> np.ndarray:
        """Async embedding function compatible with LightRAG"""
        # Process texts one at a time to ensure 1:1 mapping
        # Some APIs (like Gemini) may return multiple embeddings per text
        all_embeddings = []
        
        for text in texts:
            response = await litellm.aembedding(
                model=model,
                input=[text],  # Single text at a time
                api_base=api_base,
                api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            )
            
            # Track tokens (embeddings use prompt tokens only)
            usage = response.usage
            if usage:
                await token_tracker.add(
                    prompt_tokens=usage.prompt_tokens or 0,
                    completion_tokens=0
                )
            
            # Extract the embedding - take only the first one if multiple are returned
            if response.data:
                embedding = response.data[0]["embedding"]
                all_embeddings.append(embedding)
            else:
                raise ValueError(f"No embedding returned for text: {text[:50]}...")
        
        return np.array(all_embeddings)
    
    return embed_func


# Map query modes to EngineType
MODE_TO_ENGINE_TYPE = {
    "local": EngineType.LIGHTRAG_LOCAL,
    "global": EngineType.LIGHTRAG_GLOBAL,
    "hybrid": EngineType.LIGHTRAG_HYBRID,
    "mix": EngineType.LIGHTRAG_MIX,
    "naive": EngineType.LIGHTRAG_NAIVE,
}


class LightRAGEngine(SearchEngineInterface):
    """
    LightRAG adapter for the benchmark interface.
    
    Supports multiple query modes:
    - local: Entity-focused retrieval from knowledge graph
    - global: Broader topic/relationship retrieval  
    - hybrid: Combines local and global strategies
    - mix: Knowledge graph + vector retrieval with reranking
    - naive: Simple vector search without graph structure
    """
    
    def __init__(
        self,
        config_path: Path,
        mode: Literal["local", "global", "hybrid", "mix", "naive"] = "hybrid",
        prompt_price_per_m: float = 0.10,
        completion_price_per_m: float = 0.40,
        **kwargs
    ):
        """
        Initialize LightRAG adapter.
        
        Args:
            config_path: Path to LightRAG YAML configuration file
            mode: Query mode (local, global, hybrid, mix, naive)
            prompt_price_per_m: Price per million prompt tokens
            completion_price_per_m: Price per million completion tokens
            **kwargs: Additional configuration overrides
        """
        self.config_path = Path(config_path)
        self.mode = mode
        self.prompt_price_per_m = prompt_price_per_m
        self.completion_price_per_m = completion_price_per_m
        self.config_overrides = kwargs
        
        self._rag = None
        self._config: Optional[LightRAGConfig] = None
        self._token_tracker = TokenTracker()
        self._initialized = False
    
    @property
    def name(self) -> str:
        return f"LightRAG ({self.mode})"
    
    @property
    def engine_type(self) -> EngineType:
        return MODE_TO_ENGINE_TYPE[self.mode]
    
    async def initialize(self) -> None:
        """Initialize LightRAG engine"""
        if self._initialized:
            return
        
        # Load configuration
        if not self.config_path.exists():
            raise ValueError(f"LightRAG config file not found: {self.config_path}")
        
        self._config = LightRAGConfig.from_yaml(str(self.config_path))
        
        # Apply runtime overrides
        for key, value in self.config_overrides.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        self._config.api_key = _resolve_env_ref(self._config.api_key) or ""
        self._config.api_base = _resolve_env_ref(self._config.api_base) or None
        
        # Ensure working directory exists
        working_dir = Path(self._config.working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)
        
        # Import LightRAG
        try:
            from lightrag import LightRAG, QueryParam
            from lightrag.utils import EmbeddingFunc
        except ImportError:
            raise ImportError(
                "LightRAG not installed. Install with: pip install lightrag-hku"
            )
        
        # Create LLM and embedding functions with token tracking
        llm_func = create_litellm_complete_func(
            model=self._config.llm_model,
            api_base=self._config.api_base,
            api_key=self._config.api_key,
            token_tracker=self._token_tracker
        )
        
        # Create raw embedding function
        raw_embed_func = create_litellm_embed_func(
            model=self._config.embedding_model,
            api_base=self._config.api_base,
            api_key=self._config.api_key,
            token_tracker=self._token_tracker
        )
        
        # Get embedding dimensions for the model
        embedding_dim = self._get_embedding_dim(self._config.embedding_model)
        
        # Wrap embedding function with EmbeddingFunc (required by LightRAG)
        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=raw_embed_func
        )
        
        # Create LightRAG instance
        self._rag = LightRAG(
            working_dir=str(working_dir),
            llm_model_func=llm_func,
            embedding_func=embedding_func,
            # Storage configuration
            kv_storage=self._config.kv_storage,
            vector_storage=self._config.vector_storage,
            graph_storage=self._config.graph_storage,
            doc_status_storage=self._config.doc_status_storage,
        )
        
        # Initialize storages (required before any operations)
        await self._rag.initialize_storages()
        
        self._initialized = True
    
    def _get_embedding_dim(self, model: str) -> int:
        """Get embedding dimensions for common models"""
        dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "gemini-embedding": 3072,
            "gemini-embedding-proxy": 3072,
            "embedding-001": 768,
        }
        
        # Check if any key is contained in model name
        for key, dim in dim_map.items():
            if key in model.lower():
                return dim
        
        # Default fallback
        return 3072
    
    async def search(self, query: str) -> QueryResult:
        """Execute LightRAG search"""
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        from lightrag import QueryParam
        
        # Reset token tracker for this query
        self._token_tracker.reset()
        
        start = time.perf_counter()
        
        try:
            # Build query parameters
            query_param = QueryParam(
                mode=self.mode,
                top_k=self._config.top_k,
                max_token_for_local_context=self._config.max_entity_tokens,
                max_token_for_global_context=self._config.max_relation_tokens,
                max_token_for_text_unit=self._config.max_total_tokens,
            )
            
            # Execute query
            result = await self._rag.aquery(query, param=query_param)
            
            latency = time.perf_counter() - start
            
            # Get token counts from tracker
            prompt_tokens = self._token_tracker.prompt_tokens
            completion_tokens = self._token_tracker.completion_tokens
            total_tokens = self._token_tracker.total_tokens
            llm_calls = self._token_tracker.llm_calls
            
            # Calculate cost
            cost = self.calculate_cost(
                prompt_tokens,
                completion_tokens,
                self.prompt_price_per_m,
                self.completion_price_per_m
            )
            
            # Extract answer (LightRAG returns string directly)
            answer = result if isinstance(result, str) else str(result)
            
            return QueryResult(
                engine=self.engine_type.value,
                prompt_name="",
                prompt_text=query,
                answer=answer,
                context_text=None,  # LightRAG doesn't expose context separately
                latency_seconds=round(latency, 4),
                llm_calls=llm_calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_eur=cost,
                extra_info={
                    "mode": self.mode,
                    "top_k": self._config.top_k,
                    "max_entity_tokens": self._config.max_entity_tokens,
                    "max_relation_tokens": self._config.max_relation_tokens,
                    "max_total_tokens": self._config.max_total_tokens,
                }
            )
            
        except Exception as e:
            if LOGGER.isEnabledFor(logging.INFO):
                LOGGER.exception("LightRAG query failed for mode=%s", self.mode)
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
        """Get LightRAG statistics"""
        if not self._initialized or self._rag is None:
            return {"engine": self.name, "initialized": False}
        
        return {
            "engine": self.name,
            "mode": self.mode,
            "working_dir": str(self._config.working_dir),
            "llm_model": self._config.llm_model,
            "embedding_model": self._config.embedding_model,
            "initialized": True,
        }
    
    async def close(self) -> None:
        """Clean up LightRAG storage resources"""
        if self._initialized and self._rag is not None:
            try:
                print(f"    Closing LightRAG storage for {self.mode}...")
                await self._rag.finalize_storages()
            except Exception as e:
                print(f"    ⚠️ Failed to finalize LightRAG storage: {e}")


# Convenience classes for each query mode
class LightRAGLocalEngine(LightRAGEngine):
    """LightRAG with local (entity-focused) query mode"""
    def __init__(self, config_path: Path, **kwargs):
        super().__init__(config_path, mode="local", **kwargs)


class LightRAGGlobalEngine(LightRAGEngine):
    """LightRAG with global (topic/relationship) query mode"""
    def __init__(self, config_path: Path, **kwargs):
        super().__init__(config_path, mode="global", **kwargs)


class LightRAGHybridEngine(LightRAGEngine):
    """LightRAG with hybrid (local + global) query mode"""
    def __init__(self, config_path: Path, **kwargs):
        super().__init__(config_path, mode="hybrid", **kwargs)


class LightRAGMixEngine(LightRAGEngine):
    """LightRAG with mix (KG + vector + reranking) query mode"""
    def __init__(self, config_path: Path, **kwargs):
        super().__init__(config_path, mode="mix", **kwargs)


class LightRAGNaiveEngine(LightRAGEngine):
    """LightRAG with naive (simple vector search) query mode"""
    def __init__(self, config_path: Path, **kwargs):
        super().__init__(config_path, mode="naive", **kwargs)
