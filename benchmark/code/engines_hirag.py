"""
HiRAG Engine Adapter

Wraps HiRAG in the standard SearchEngineInterface for benchmarking.
Supports all HiRAG query modes: hi, hi_local, hi_global, hi_bridge, hi_nobridge, naive.

HiRAG paper: https://arxiv.org/abs/2503.10150 (EMNLP 2025 Findings)
"""

import os
import time
import yaml
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass

import litellm
import numpy as np

from .engine_interface import SearchEngineInterface, QueryResult, EngineType

LOGGER = logging.getLogger(__name__)

# Retry settings for API calls
MAX_RETRIES = 20
RETRY_BASE_WAIT = 30      # seconds
RETRY_MAX_WAIT = 120       # max seconds between retries


async def _retry_api_call(func, *args, call_name="API call", **kwargs):
    """Retry wrapper for async API calls with exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await func(*args, **kwargs)
        except Exception:
            if attempt == MAX_RETRIES:
                raise
            wait = min(RETRY_BASE_WAIT * (2 ** (attempt - 1)), RETRY_MAX_WAIT)
            await asyncio.sleep(wait)


@dataclass
class HiRAGConfig:
    """Configuration for HiRAG engine"""
    # Required paths
    working_dir: str

    # Model configuration
    llm_model: str = "openai/gemini-flash-proxy"
    embedding_model: str = "openai/gemini-embedding-proxy"
    api_key: str = ""
    api_base: Optional[str] = None

    # Query parameters
    top_k: int = 20         # retrieve top-k entities
    top_m: int = 10         # retrieve top-m entities per community

    # Embedding settings
    embedding_dim: int = 3072
    embedding_batch_num: int = 6
    embedding_func_max_async: int = 8

    # LLM settings
    best_model_max_async: int = 8
    cheap_model_max_async: int = 8

    # Cache (HiRAG official config.yaml defaults to false)
    enable_llm_cache: bool = False

    # Token limits for query
    max_token_for_text_unit: int = 20000
    max_token_for_local_context: int = 20000
    max_token_for_bridge_knowledge: int = 12500
    max_token_for_community_report: int = 12500
    
    # Custom limit setting
    limit_tokens: bool = False
    max_total_tokens: int = 2500

    @classmethod
    def from_yaml(cls, path: str) -> "HiRAGConfig":
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


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


# Map query modes to EngineType
MODE_TO_ENGINE_TYPE = {
    "hi":          EngineType.HIRAG_HI,
    "hi_local":    EngineType.HIRAG_LOCAL,
    "hi_global":   EngineType.HIRAG_GLOBAL,
    "hi_bridge":   EngineType.HIRAG_BRIDGE,
    "hi_nobridge": EngineType.HIRAG_NOBRIDGE,
    "naive":       EngineType.HIRAG_NAIVE,
}


def _create_hirag_llm_func(
    model: str,
    api_base: Optional[str],
    api_key: str,
    token_tracker: TokenTracker,
):
    """
    Create a LiteLLM-backed LLM function compatible with HiRAG's expected signature:
        async (prompt, system_prompt=None, history_messages=[], **kwargs) -> str

    HiRAG internally passes a `hashing_kv` kwarg for its own LLM response cache.
    This implements the same caching logic as HiRAG's official examples
    (hi_Search_openai.py), which is critical for correct behavior:
    - During indexing: caches entity extraction LLM calls
    - During query: caches generation LLM calls (if enable_llm_cache=True)
    """
    from hirag._utils import compute_args_hash

    async def llm_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: list = [],
        **kwargs
    ) -> str:
        # Extract HiRAG's cache KV store (used when enable_llm_cache=True)
        hashing_kv = kwargs.pop("hashing_kv", None)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # Check HiRAG's LLM response cache before calling the API
        if hashing_kv is not None:
            args_hash = compute_args_hash(model, messages)
            if_cache_return = await hashing_kv.get_by_id(args_hash)
            if if_cache_return is not None:
                return if_cache_return["return"]

        # Only pass safe LLM kwargs to avoid litellm errors
        VALID_KWARGS = {
            'temperature', 'max_tokens', 'top_p', 'frequency_penalty',
            'presence_penalty', 'stop', 'n', 'stream', 'timeout',
            'seed', 'response_format',
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in VALID_KWARGS}

        response = await _retry_api_call(
            litellm.acompletion,
            model=model,
            messages=messages,
            api_base=api_base,
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            call_name="LLM completion",
            **filtered_kwargs
        )

        usage = response.usage
        await token_tracker.add(
            prompt_tokens=usage.prompt_tokens or 0,
            completion_tokens=usage.completion_tokens or 0,
        )

        result_content = response.choices[0].message.content or ""

        # Store in HiRAG's LLM response cache
        if hashing_kv is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": result_content, "model": model}}
            )

        return result_content

    return llm_func


def _create_hirag_embed_func(
    model: str,
    api_base: Optional[str],
    api_key: str,
    token_tracker: TokenTracker,
    embedding_dim: int,
):
    """
    Create a LiteLLM-backed embedding function wrapped in HiRAG's EmbeddingFunc dataclass.
    HiRAG's EmbeddingFunc expects: async (texts: list[str]) -> np.ndarray
    """
    from hirag._utils import EmbeddingFunc

    async def raw_embed(texts: list) -> np.ndarray:
        all_embeddings = []
        for text in texts:
            response = await _retry_api_call(
                litellm.aembedding,
                model=model,
                input=[text],
                api_base=api_base,
                api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
                call_name="Embedding",
            )
            usage = response.usage
            if usage:
                await token_tracker.add(
                    prompt_tokens=usage.prompt_tokens or 0,
                    completion_tokens=0,
                )
            if response.data:
                all_embeddings.append(response.data[0]["embedding"])
            else:
                raise ValueError(f"No embedding returned for text: {text[:50]}...")

        return np.array(all_embeddings)

    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=raw_embed,
    )


class HiRAGEngine(SearchEngineInterface):
    """
    HiRAG adapter for the benchmark interface.

    Supports multiple query modes:
    - hi:          Full hierarchical retrieval (local + global + bridge)
    - hi_local:    Only local (entity-level) knowledge
    - hi_global:   Only global (community-level) knowledge
    - hi_bridge:   Only bridge (inter-community) knowledge
    - hi_nobridge: Hierarchical without bridge knowledge
    - naive:       Simple vector-chunk retrieval (no graph)
    """

    def __init__(
        self,
        config_path: Path,
        mode: Literal["hi", "hi_local", "hi_global", "hi_bridge", "hi_nobridge", "naive"] = "hi",
        prompt_price_per_m: float = 0.10,
        completion_price_per_m: float = 0.40,
        **kwargs
    ):
        """
        Initialize HiRAG adapter.

        Args:
            config_path: Path to HiRAG YAML configuration file
            mode: Query mode
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
        self._config: Optional[HiRAGConfig] = None
        self._token_tracker = TokenTracker()
        self._initialized = False

    @property
    def name(self) -> str:
        return f"HiRAG ({self.mode})"

    @property
    def engine_type(self) -> EngineType:
        return MODE_TO_ENGINE_TYPE[self.mode]

    async def initialize(self) -> None:
        """Initialize HiRAG engine"""
        if self._initialized:
            return

        if not self.config_path.exists():
            raise ValueError(f"HiRAG config file not found: {self.config_path}")

        self._config = HiRAGConfig.from_yaml(str(self.config_path))

        # Apply runtime overrides
        for key, value in self.config_overrides.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        self._config.api_key = _resolve_env_ref(self._config.api_key) or ""
        self._config.api_base = _resolve_env_ref(self._config.api_base) or None

        # Ensure working directory exists
        working_dir = Path(self._config.working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)

        try:
            from hirag import HiRAG
        except ImportError:
            raise ImportError(
                "HiRAG not installed. Install with: pip install -e /path/to/HiRAG"
            )

        # Create LLM functions
        best_model_func = _create_hirag_llm_func(
            model=self._config.llm_model,
            api_base=self._config.api_base,
            api_key=self._config.api_key,
            token_tracker=self._token_tracker,
        )
        cheap_model_func = _create_hirag_llm_func(
            model=self._config.llm_model,
            api_base=self._config.api_base,
            api_key=self._config.api_key,
            token_tracker=self._token_tracker,
        )

        # Create embedding function (HiRAG EmbeddingFunc dataclass)
        embedding_func = _create_hirag_embed_func(
            model=self._config.embedding_model,
            api_base=self._config.api_base,
            api_key=self._config.api_key,
            token_tracker=self._token_tracker,
            embedding_dim=self._config.embedding_dim,
        )

        # Instantiate HiRAG
        # enable_naive_rag=True always, matching HiRAG's official config.yaml.
        # This ensures the chunks VDB is initialized (needed even for non-naive modes
        # since HiRAG loads text chunks during hierarchical queries).
        self._rag = HiRAG(
            working_dir=str(working_dir),
            enable_hierachical_mode=True,
            enable_naive_rag=True,
            best_model_func=best_model_func,
            best_model_max_async=self._config.best_model_max_async,
            cheap_model_func=cheap_model_func,
            cheap_model_max_async=self._config.cheap_model_max_async,
            embedding_func=embedding_func,
            embedding_batch_num=self._config.embedding_batch_num,
            embedding_func_max_async=self._config.embedding_func_max_async,
            enable_llm_cache=self._config.enable_llm_cache,
        )

        self._initialized = True

    async def search(self, query: str) -> QueryResult:
        """Execute HiRAG search"""
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        from hirag import QueryParam

        # Reset token tracker for this query
        self._token_tracker.reset()

        start = time.perf_counter()

        try:
            if getattr(self._config, "limit_tokens", False):
                # Calculate dynamic budget for token limits
                total_limit = getattr(self._config, "max_total_tokens", 2500)
                
                # Determine active components based on mode
                if self.mode == "hi":
                    active_components = 4 # background, reasoning, local, source
                elif self.mode == "hi_nobridge":
                    active_components = 3
                elif self.mode in ["hi_bridge", "hi_local", "hi_global"]:
                    active_components = 2 
                elif self.mode == "naive":
                    active_components = 1
                else:
                    active_components = 4 # fallback
                    
                budget_per_component = total_limit // active_components
                
                query_param = QueryParam(
                    mode=self.mode,
                    top_k=self._config.top_k,
                    top_m=self._config.top_m,
                    max_token_for_text_unit=budget_per_component,
                    max_token_for_local_context=budget_per_component,
                    max_token_for_bridge_knowledge=budget_per_component,
                    max_token_for_community_report=budget_per_component,
                )
                query_param.naive_max_token_for_text_unit = total_limit
            else:
                query_param = QueryParam(
                    mode=self.mode,
                    top_k=self._config.top_k,
                    top_m=self._config.top_m,
                    max_token_for_text_unit=self._config.max_token_for_text_unit,
                    max_token_for_local_context=self._config.max_token_for_local_context,
                    max_token_for_bridge_knowledge=self._config.max_token_for_bridge_knowledge,
                    max_token_for_community_report=self._config.max_token_for_community_report,
                )

            result = await self._rag.aquery(query, param=query_param)

            latency = time.perf_counter() - start

            prompt_tokens = self._token_tracker.prompt_tokens
            completion_tokens = self._token_tracker.completion_tokens
            total_tokens = self._token_tracker.total_tokens
            llm_calls = self._token_tracker.llm_calls

            cost = self.calculate_cost(
                prompt_tokens,
                completion_tokens,
                self.prompt_price_per_m,
                self.completion_price_per_m,
            )

            answer = result if isinstance(result, str) else str(result)

            return QueryResult(
                engine=self.engine_type.value,
                prompt_name="",
                prompt_text=query,
                answer=answer,
                context_text=None,
                latency_seconds=round(latency, 4),
                llm_calls=llm_calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_eur=cost,
                extra_info={
                    "mode": self.mode,
                    "top_k": self._config.top_k,
                    "top_m": self._config.top_m,
                    "max_token_for_text_unit": self._config.max_token_for_text_unit,
                    "max_token_for_community_report": self._config.max_token_for_community_report,
                }
            )

        except Exception as e:
            if LOGGER.isEnabledFor(logging.INFO):
                LOGGER.exception("HiRAG query failed for mode=%s", self.mode)
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
        """Get HiRAG statistics"""
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


# Convenience classes for each query mode

class HiRAGHiEngine(HiRAGEngine):
    """HiRAG full hierarchical mode (local + global + bridge)"""
    def __init__(self, config_path: Path, **kwargs):
        super().__init__(config_path, mode="hi", **kwargs)


class HiRAGLocalEngine(HiRAGEngine):
    """HiRAG local (entity-level) knowledge only"""
    def __init__(self, config_path: Path, **kwargs):
        super().__init__(config_path, mode="hi_local", **kwargs)


class HiRAGGlobalEngine(HiRAGEngine):
    """HiRAG global (community-level) knowledge only"""
    def __init__(self, config_path: Path, **kwargs):
        super().__init__(config_path, mode="hi_global", **kwargs)


class HiRAGBridgeEngine(HiRAGEngine):
    """HiRAG bridge (inter-community) knowledge only"""
    def __init__(self, config_path: Path, **kwargs):
        super().__init__(config_path, mode="hi_bridge", **kwargs)


class HiRAGNoBridgeEngine(HiRAGEngine):
    """HiRAG hierarchical without bridge knowledge"""
    def __init__(self, config_path: Path, **kwargs):
        super().__init__(config_path, mode="hi_nobridge", **kwargs)


class HiRAGNaiveEngine(HiRAGEngine):
    """HiRAG naive (simple vector-chunk retrieval, no graph)"""
    def __init__(self, config_path: Path, **kwargs):
        super().__init__(config_path, mode="naive", **kwargs)
