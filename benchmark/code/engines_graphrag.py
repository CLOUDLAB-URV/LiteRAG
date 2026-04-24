"""
GraphRAG Engine Adapters

Wraps GraphRAG search methods in the standard SearchEngineInterface.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from .engine_interface import SearchEngineInterface, QueryResult, EngineType

LOGGER = logging.getLogger(__name__)


class GraphRAGEngine(SearchEngineInterface):
    """
    Base class for GraphRAG search engines.
    
    Handles common initialization and provides shared utilities.
    """
    
    def __init__(
        self,
        root_dir: Path,
        prompt_price_per_m: float = 0.10,
        completion_price_per_m: float = 0.40
    ):
        self.root_dir = Path(root_dir)
        self.output_dir = self.root_dir / "output"
        self.prompt_price_per_m = prompt_price_per_m
        self.completion_price_per_m = completion_price_per_m
        
        self._engine = None
        self._initialized = False
        
        # Shared resources (loaded once)
        self._shared_resources = None
    
    def _load_shared_resources(self):
        """Load resources shared across all GraphRAG engines"""
        if self._shared_resources is not None:
            return self._shared_resources
        
        from graphrag.config.enums import ModelType
        from graphrag.language_model.manager import ModelManager
        from graphrag.tokenizer.get_tokenizer import get_tokenizer
        from graphrag.config.load_config import load_config
        from graphrag.query.indexer_adapters import (
            read_indexer_entities,
            read_indexer_relationships,
            read_indexer_reports,
            read_indexer_text_units,
        )
        
        cfg = load_config(self.root_dir)
        chat_cfg = cfg.models["default_chat_model"]
        embed_cfg = cfg.models["default_embedding_model"]
        
        chat_model = ModelManager().get_or_create_chat_model(
            name="benchmark_chat",
            model_type=ModelType.Chat,
            config=chat_cfg,
        )
        
        tokenizer = get_tokenizer(chat_cfg)
        
        text_embedder = ModelManager().get_or_create_embedding_model(
            name="benchmark_embed",
            model_type=ModelType.Embedding,
            config=embed_cfg,
        )
        
        # Load artifacts
        entities_df = pd.read_parquet(self.output_dir / "entities.parquet")
        communities_df = pd.read_parquet(self.output_dir / "communities.parquet")
        relationships_df = pd.read_parquet(self.output_dir / "relationships.parquet")
        text_units_df = pd.read_parquet(self.output_dir / "text_units.parquet")
        reports_df = pd.read_parquet(self.output_dir / "community_reports.parquet")
        
        community_level = int(communities_df["level"].max())
        
        entities = read_indexer_entities(entities_df, communities_df, community_level)
        relationships = read_indexer_relationships(relationships_df)
        text_units = read_indexer_text_units(text_units_df)
        reports = read_indexer_reports(reports_df, communities_df, community_level)
        
        self._shared_resources = {
            "chat_model": chat_model,
            "tokenizer": tokenizer,
            "text_embedder": text_embedder,
            "entities": entities,
            "relationships": relationships,
            "text_units": text_units,
            "reports": reports,
            "entities_df": entities_df,
            "communities_df": communities_df,
        }
        
        return self._shared_resources
    
    def _extract_telemetry(self, result) -> Dict[str, int]:
        """Extract token usage from result"""
        telemetry = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        if hasattr(result, "__dict__"):
            result_dict = result.__dict__
        elif isinstance(result, dict):
            result_dict = result
        else:
            return telemetry
        
        for key in ["llm_calls", "prompt_tokens", "output_tokens", "completion_tokens"]:
            if key in result_dict:
                val = result_dict[key]
                if isinstance(val, dict):
                    val = sum(v for v in val.values() if isinstance(v, (int, float)))
                mapped_key = "completion_tokens" if key == "output_tokens" else key
                telemetry[mapped_key] = int(val or 0)
        
        telemetry["total_tokens"] = telemetry["prompt_tokens"] + telemetry["completion_tokens"]
        return telemetry
    
    def _extract_response(self, result) -> str:
        """Extract response text from result"""
        if isinstance(result, str):
            return result
        
        for attr in ["response", "text", "content", "answer"]:
            if hasattr(result, attr):
                val = getattr(result, attr)
                if val:
                    return str(val)
        
        if isinstance(result, dict):
            for key in ["response", "text", "content", "answer"]:
                if key in result and result[key]:
                    return str(result[key])
        
        if isinstance(result, tuple) and len(result) > 0:
            return self._extract_response(result[0])
        
        return str(result)
    
    def _extract_context(self, result) -> Optional[str]:
        """Extract context from result"""
        if hasattr(result, "context_text"):
            return result.context_text
        if isinstance(result, dict) and "context_text" in result:
            return result["context_text"]
        return None

    async def search(self, query: str) -> QueryResult:
        """Execute a query with shared response/telemetry extraction logic."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        start = time.perf_counter()

        try:
            result = await self._engine.search(query)
            latency = time.perf_counter() - start

            response = self._extract_response(result)
            context = self._extract_context(result)
            telemetry = self._extract_telemetry(result)
            cost = self.calculate_cost(
                telemetry["prompt_tokens"],
                telemetry["completion_tokens"],
                self.prompt_price_per_m,
                self.completion_price_per_m,
            )

            return QueryResult(
                engine=self.engine_type.value,
                prompt_name="",
                prompt_text=query,
                answer=response,
                context_text=context,
                latency_seconds=round(latency, 4),
                llm_calls=telemetry["llm_calls"],
                prompt_tokens=telemetry["prompt_tokens"],
                completion_tokens=telemetry["completion_tokens"],
                total_tokens=telemetry["total_tokens"],
                cost_eur=cost,
            )
        except Exception as e:
            if LOGGER.isEnabledFor(logging.INFO):
                LOGGER.exception("%s search failed", self.name)
            latency = time.perf_counter() - start
            return QueryResult(
                engine=self.engine_type.value,
                prompt_name="",
                prompt_text=query,
                answer="",
                latency_seconds=round(latency, 4),
                error=str(e),
            )


class GraphRAGBasicEngine(GraphRAGEngine):
    """GraphRAG Basic Search adapter"""
    
    @property
    def name(self) -> str:
        return "GraphRAG Basic"
    
    @property
    def engine_type(self) -> EngineType:
        return EngineType.GRAPHRAG_BASIC
    
    async def initialize(self) -> None:
        if self._initialized:
            return
        
        from graphrag.query.structured_search.basic_search.search import BasicSearch
        from graphrag.vector_stores.lancedb import LanceDBVectorStore
        from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
        from graphrag.config.load_config import load_config
        from types import SimpleNamespace
        
        resources = self._load_shared_resources()
        
        # Load configuration for max_context_tokens and k
        cfg = load_config(self.root_dir)
        basic_search_cfg = cfg.basic_search
        max_context_tokens = basic_search_cfg.max_context_tokens
        k = basic_search_cfg.k
        
        # Text units vector store
        textunit_store = LanceDBVectorStore(
            vector_store_schema_config=VectorStoreSchemaConfig(
                index_name="default-text_unit-text"
            )
        )
        textunit_store.connect(db_uri=str(self.output_dir / "lancedb"))
        
        # Context builder
        context_builder = MinimalBasicContextBuilder(
            text_units=resources["text_units"],
            text_unit_embeddings=textunit_store,
            text_embedder=resources["text_embedder"],
            tokenizer=resources["tokenizer"],
        )
        
        self._engine = BasicSearch(
            model=resources["chat_model"],
            context_builder=context_builder,
            context_builder_params={"k": k, "max_context_tokens": max_context_tokens},
        )
        
        self._initialized = True
    
    def get_stats(self) -> Dict[str, Any]:
        resources = self._load_shared_resources()
        return {
            "engine": self.name,
            "text_units": len(resources["text_units"]),
        }


class GraphRAGLocalEngine(GraphRAGEngine):
    """GraphRAG Local Search adapter"""
    
    @property
    def name(self) -> str:
        return "GraphRAG Local"
    
    @property
    def engine_type(self) -> EngineType:
        return EngineType.GRAPHRAG_LOCAL
    
    async def initialize(self) -> None:
        if self._initialized:
            return
        
        from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
        from graphrag.query.structured_search.local_search.search import LocalSearch
        from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
        from graphrag.vector_stores.lancedb import LanceDBVectorStore
        from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
        from graphrag.config.load_config import load_config
        
        resources = self._load_shared_resources()
        
        # Load configuration for max_context_tokens
        cfg = load_config(self.root_dir)
        local_search_cfg = cfg.local_search
        max_context_tokens = local_search_cfg.max_context_tokens
        
        description_store = LanceDBVectorStore(
            vector_store_schema_config=VectorStoreSchemaConfig(
                index_name="default-entity-description"
            )
        )
        description_store.connect(db_uri=str(self.output_dir / "lancedb"))
        
        context = LocalSearchMixedContext(
            community_reports=resources["reports"],
            text_units=resources["text_units"],
            entities=resources["entities"],
            relationships=resources["relationships"],
            entity_text_embeddings=description_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            text_embedder=resources["text_embedder"],
            tokenizer=resources["tokenizer"],
        )
        
        self._engine = LocalSearch(
            model=resources["chat_model"],
            context_builder=context,
            context_builder_params={"max_context_tokens": max_context_tokens},
        )
        
        self._initialized = True
    
    def get_stats(self) -> Dict[str, Any]:
        resources = self._load_shared_resources()
        return {
            "engine": self.name,
            "entities": len(resources["entities"]),
            "relationships": len(resources["relationships"]),
        }


class GraphRAGGlobalEngine(GraphRAGEngine):
    """GraphRAG Global Search adapter"""
    
    @property
    def name(self) -> str:
        return "GraphRAG Global"
    
    @property
    def engine_type(self) -> EngineType:
        return EngineType.GRAPHRAG_GLOBAL
    
    async def initialize(self) -> None:
        if self._initialized:
            return
        
        from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
        from graphrag.query.structured_search.global_search.search import GlobalSearch
        
        resources = self._load_shared_resources()
        
        context = GlobalCommunityContext(
            community_reports=resources["reports"],
            communities=None,
            entities=resources["entities"],
            tokenizer=resources["tokenizer"],
        )
        
        self._engine = GlobalSearch(
            model=resources["chat_model"],
            context_builder=context
        )
        
        self._initialized = True
    
    def get_stats(self) -> Dict[str, Any]:
        resources = self._load_shared_resources()
        return {
            "engine": self.name,
            "reports": len(resources["reports"]),
        }


class GraphRAGDriftEngine(GraphRAGEngine):
    """GraphRAG DRIFT Search adapter"""
    
    @property
    def name(self) -> str:
        return "GraphRAG DRIFT"
    
    @property
    def engine_type(self) -> EngineType:
        return EngineType.GRAPHRAG_DRIFT
    
    async def initialize(self) -> None:
        if self._initialized:
            return
        
        from graphrag.query.structured_search.drift_search.drift_context import DRIFTSearchContextBuilder
        from graphrag.query.structured_search.drift_search.search import DRIFTSearch
        from graphrag.vector_stores.lancedb import LanceDBVectorStore
        from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
        
        resources = self._load_shared_resources()

        description_store = LanceDBVectorStore(
            vector_store_schema_config=VectorStoreSchemaConfig(
                index_name="default-entity-description"
            )
        )
        description_store.connect(db_uri=str(self.output_dir / "lancedb"))
        
        fullcontent_store = LanceDBVectorStore(
            vector_store_schema_config=VectorStoreSchemaConfig(
                index_name="default-community-full_content"
            )
        )
        fullcontent_store.connect(db_uri=str(self.output_dir / "lancedb"))
        
        # Load report embeddings - this can be slow with large datasets
        for report in resources["reports"]:
            try:
                result = fullcontent_store.search_by_id(report.id)
                if result and hasattr(result, 'vector'):
                    report.full_content_embedding = result.vector
            except Exception:
                if LOGGER.isEnabledFor(logging.INFO):
                    LOGGER.exception(
                        "Failed loading DRIFT report embedding for report id=%s", report.id
                    )
        
        context = DRIFTSearchContextBuilder(
            model=resources["chat_model"],
            text_embedder=resources["text_embedder"],
            entities=resources["entities"],
            relationships=resources["relationships"],
            reports=resources["reports"],
            entity_text_embeddings=description_store,
            text_units=resources["text_units"],
            tokenizer=resources["tokenizer"],
        )
        
        self._engine = DRIFTSearch(
            model=resources["chat_model"],
            context_builder=context,
        )
        
        self._initialized = True
    
    def get_stats(self) -> Dict[str, Any]:
        resources = self._load_shared_resources()
        return {
            "engine": self.name,
            "entities": len(resources["entities"]),
            "reports": len(resources["reports"]),
        }


# Helper class from original benchmark
class MinimalBasicContextBuilder:
    """Context builder for Basic search - compatible with graphrag's BasicSearch"""
    
    def __init__(self, *, text_units, text_unit_embeddings, text_embedder, tokenizer):
        self.text_units = text_units
        self.store = text_unit_embeddings
        self.embedder = text_embedder
        self.tokenizer = tokenizer
        
        self._tu_by_id = {}
        for tu in text_units:
            tid = getattr(tu, "id", None)
            txt = getattr(tu, "text", None) or getattr(tu, "content", None)
            if tid is not None and txt:
                self._tu_by_id[tid] = txt
    
    def build_context(self, query: str, **params):
        from types import SimpleNamespace
        
        k = int(params.get("k", 10))
        # Support both max_context_tokens (GraphRAG config) and max_tokens
        max_tokens = int(params.get("max_context_tokens", params.get("max_tokens", 8_000)))
        
        hits = self.store.similarity_search_by_text(
            text=query,
            text_embedder=lambda t: self.embedder.embed(t),
            k=k,
        )
        
        chosen = []
        curr_tok = 0
        
        for h in hits:
            txt = getattr(getattr(h, "document", None), "text", None)
            if not txt:
                tid = getattr(getattr(h, "document", None), "id", None)
                txt = self._tu_by_id.get(tid, "")
            if not txt:
                continue
            
            if hasattr(self.tokenizer, "encode"):
                tks = len(self.tokenizer.encode(txt))
            else:
                tks = len(txt.split())
            
            if curr_tok + tks > max_tokens:
                break
            
            chosen.append(txt)
            curr_tok += tks
        
        context_text = "\n\n".join(chosen)
        
        # Return object compatible with BasicSearch expectations
        # BasicSearch expects context_chunks and context_records attributes
        return SimpleNamespace(
            context=context_text,
            context_text=context_text,
            text=context_text,
            context_chunks=chosen,  # Required by BasicSearch
            context_records=chosen,  # Also required by BasicSearch in newer versions
            context_data={"text_units": chosen},
            llm_calls=0,
            prompt_tokens=0,
            output_tokens=0,
        )
