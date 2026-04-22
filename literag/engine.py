"""
LiteRAG Engine - Orchestrator with Safety Net
"""

import time
from typing import Optional, Dict, Any, List
import numpy as np
import logging

from .config import LiteRAGConfig
from .models import LiteRAGResult, Anchor, Subgraph, RankedEntity
from .data_loader import GraphData, load_graph_data
from .embeddings import EmbeddingService
from .anchor_discovery import AnchorDiscovery
from .exploration import GraphExplorer, ParallelExplorer, ExplorationConfig
from .consensus import ConsensusMerger, filter_redundant_entities
from .context import ContextAssembler
from .llm_service import LLMService
from .vector_store import LiteRAGVectorStore, GRAPHRAG_TEXTUNIT_TABLE

logger = logging.getLogger(__name__)


class LiteRAG:
    """Main LiteRAG orchestrator for data loading, retrieval, and response generation."""

    def __init__(self, config: LiteRAGConfig):
        self.config = config
        self.graph_data: Optional[GraphData] = None
        self.vector_store: Optional[LiteRAGVectorStore] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.llm_service: Optional[LLMService] = None
        self.anchor_discovery: Optional[AnchorDiscovery] = None
        self.explorer: Optional[ParallelExplorer] = None
        self.merger: Optional[ConsensusMerger] = None
        self.context_assembler: Optional[ContextAssembler] = None
        self._initialized = False
    
    def initialize(self, force_reload: bool = False):
        """Initialize all LiteRAG services and load graph/runtime state.

        Args:
            force_reload: If True, reload cached resources and rebuild in-memory state.
        """
        if self._initialized:
            if not force_reload:
                return
            # Explicitly close old db connection to prevent memory leaks
            if getattr(self, 'vector_store', None):
                self.vector_store.close()
                
        logger.info("Initializing LiteRAG...")
        
        self.vector_store = LiteRAGVectorStore(self.config.lancedb_uri)
        self.graph_data = load_graph_data(self.config.data_dir, force_recompute=force_reload)
        
        self.embedding_service = EmbeddingService(
            api_key=self.config.api_key,
            model=self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
            use_litellm=self.config.use_litellm,
            litellm_base_url=self.config.litellm_base_url
        )
        
        self.llm_service = LLMService(
            api_key=self.config.api_key,
            model=self.config.llm_model,
            use_litellm=self.config.use_litellm,
            litellm_base_url=self.config.litellm_base_url
        )
        
        self.anchor_discovery = AnchorDiscovery(
            graph_data=self.graph_data,
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            semantic_weight=self.config.semantic_weight,
            keyword_exact_weight=self.config.keyword_exact_weight,
            keyword_fuzzy_weight=self.config.keyword_fuzzy_weight,
            community_weight=self.config.community_weight,
            min_score_threshold=self.config.min_anchor_score,
            max_anchors=self.config.max_anchors
        )
        self.anchor_discovery.ensure_embeddings()
        
        exp_config = ExplorationConfig(
            max_depth=self.config.max_exploration_depth,
            min_relevance_threshold=self.config.min_relevance_threshold,
            decay_factor=self.config.relevance_decay_factor,
            max_nodes_per_anchor=self.config.max_nodes_per_anchor,
            max_neighbors_per_hop=self.config.max_neighbors_per_hop,
            degree_influence=self.config.degree_influence,
            community_cohesion=self.config.community_cohesion_weight if hasattr(self.config, 'community_cohesion_weight') else 0.8,
            signal_amplification=self.config.signal_amplification_factor if hasattr(self.config, 'signal_amplification_factor') else 0.5,
        )
        
        self.explorer = ParallelExplorer(
            graph_data=self.graph_data,
            embedding_service=self.embedding_service,
            config=exp_config,
            num_workers=self.config.num_exploration_workers,
            vector_store=self.vector_store,
            entity_table_name=self.anchor_discovery._entity_table_name
        )

        # Force loading of embeddings into memory
        self.explorer.warmup() 
        
        self.merger = ConsensusMerger(
            graph_data=self.graph_data,
            intersection_weight=self.config.intersection_weight,
            semantic_weight=self.config.consensus_semantic_weight,
            structural_weight=self.config.structural_weight,
            proximity_weight=self.config.proximity_weight,
            anchor_boost=self.config.anchor_boost
        )
        
        self.context_assembler = ContextAssembler(
            graph_data=self.graph_data,
            max_context_tokens=self.config.max_context_tokens
        )
        
        self.llm_service.warmup()
        self._initialized = True
    
    def query(self, query: str, expand_query: bool = True) -> LiteRAGResult:
        """Run a full LiteRAG query synchronously.

        Args:
            query: User query string.
            expand_query: If True and enabled in config, run query expansion.

        Returns:
            LiteRAGResult containing answer, intermediate artifacts, and metrics.
        """
        if not self._initialized:
            raise RuntimeError("LiteRAG not initialized.")
        
        start_time = time.time()
        self.llm_service.reset_stats()
        result = LiteRAGResult(query=query, answer="")
        
        try:
            # 1. Query Expansion (optional)
            expanded_query = None
            if expand_query and self.config.enable_query_expansion:
                expanded_query = self.llm_service.expand_query(query)
                result.expanded_query = expanded_query
            
            # 2. Query Embedding
            search_query = expanded_query if expanded_query else query
            query_embedding = self.embedding_service.embed(search_query)
            
            # 3. Safety Net (Hybrid Search on Text Units)
            safety_net_chunks = []
            if self.config.enable_safety_net:
                table_name = "default-text_unit-text"
                if self.vector_store.has_table(table_name):
                    hits = self.vector_store.similarity_search(
                        table_name, 
                        query_embedding, 
                        k=self.config.safety_net_k
                    )
                    for chunk_id, score in hits:
                        text = self.graph_data.text_units.get(chunk_id)
                        if text:
                            safety_net_chunks.append(text)

            # 4. Anchor Discovery
            anchors = self.anchor_discovery.discover_anchors(query, expanded_query)
            result.anchors = anchors
            
            if not anchors and not safety_net_chunks:
                result.answer = "No relevant information found."
                return result
            
            # 5. Graph Exploration
            subgraphs = self.explorer.explore(anchors, query_embedding)
            result.subgraphs = subgraphs
            result.total_entities_explored = sum(len(sg.nodes) for sg in subgraphs)
            
            # 6. Consensus Merge & Rank
            ranked_entities = self.merger.merge_and_rank(subgraphs, anchors, self.config.max_ranked_entities)
            ranked_entities = filter_redundant_entities(ranked_entities)
            result.ranked_entities = ranked_entities
            
            # 7. Context Assembly
            context, tokens_used = self.context_assembler.assemble_context(
                ranked_entities=ranked_entities,
                query=query,
                safety_net_text_units=safety_net_chunks
            )
            result.context = context
            result.tokens_used = tokens_used
            result.entities_in_context = len(ranked_entities)
            
            # 8. Response Generation
            answer = self.llm_service.generate_response(
                context=context,
                query=query,
                temperature=self.config.response_temperature
            )
            result.answer = answer
            
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            result.success = False
            result.error = str(e)
            result.answer = "An error occurred while processing the graph context."
        
        # Stats
        elapsed = (time.time() - start_time) * 1000
        stats = self.llm_service.get_stats()
        result.prompt_tokens = stats.get('prompt_tokens', 0)
        result.completion_tokens = stats.get('completion_tokens', 0)
        result.llm_calls = stats.get('call_count', 0)
        result.latency_ms = elapsed
        
        return result

    async def aquery(self, query: str, expand_query: bool = True) -> LiteRAGResult:
        """
        Native Async version of query(). 
        Uses explorer.aexplore() instead of the ThreadPool bridge.
        """
        if not self._initialized:
            raise RuntimeError("LiteRAG not initialized.")
        
        start_time = time.time()
        self.llm_service.reset_stats()
        result = LiteRAGResult(query=query, answer="")
        
        try:
            expanded_query = None
            if expand_query and self.config.enable_query_expansion:
                # LLM calls kept sync as they are fast/managed by their own clients
                expanded_query = self.llm_service.expand_query(query)
                result.expanded_query = expanded_query
            
            search_query = expanded_query if expanded_query else query
            query_embedding = self.embedding_service.embed(search_query)
            
            safety_net_chunks = []
            if self.config.enable_safety_net:
                table_name = "default-text_unit-text"
                if self.vector_store.has_table(table_name):
                    hits = self.vector_store.similarity_search(
                        table_name, query_embedding, k=self.config.safety_net_k
                    )
                    for chunk_id, score in hits:
                        text = self.graph_data.text_units.get(chunk_id)
                        if text: safety_net_chunks.append(text)

            anchors = self.anchor_discovery.discover_anchors(query, expanded_query)
            result.anchors = anchors
            
            if not anchors and not safety_net_chunks:
                result.answer = "No relevant information found."
                return result
            
            # Use native async exploration
            subgraphs = await self.explorer.aexplore(anchors, query_embedding)
            
            result.subgraphs = subgraphs
            result.total_entities_explored = sum(len(sg.nodes) for sg in subgraphs)
            
            ranked_entities = self.merger.merge_and_rank(subgraphs, anchors, self.config.max_ranked_entities)
            ranked_entities = filter_redundant_entities(ranked_entities)
            result.ranked_entities = ranked_entities
            
            context, tokens_used = self.context_assembler.assemble_context(
                ranked_entities=ranked_entities, query=query, safety_net_text_units=safety_net_chunks
            )
            result.context = context
            result.tokens_used = tokens_used
            result.entities_in_context = len(ranked_entities)
            
            answer = self.llm_service.generate_response(
                context=context, query=query, temperature=self.config.response_temperature
            )
            result.answer = answer
            
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            result.success = False
            result.error = str(e)
            result.answer = "An error occurred while processing the graph context."
        
        elapsed = (time.time() - start_time) * 1000
        stats = self.llm_service.get_stats()
        result.prompt_tokens = stats.get('prompt_tokens', 0)
        result.completion_tokens = stats.get('completion_tokens', 0)
        result.llm_calls = stats.get('call_count', 0)
        result.latency_ms = elapsed
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Return lightweight runtime and usage statistics."""
        if not self._initialized: return {"initialized": False}
        return {
            "initialized": True,
            "entities": len(self.graph_data.entities),
            "text_units": len(self.graph_data.text_units),
            "llm_stats": self.llm_service.get_stats()
        }