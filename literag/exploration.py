"""
Graph Exploration for LiteRAG - High Performance Async Implementation
"""

import asyncio
import logging
import threading
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from .models import Entity, Anchor, ExploredNode, Subgraph
from .data_loader import GraphData, get_neighbors, get_edge_weight
from .embeddings import EmbeddingService
from .vector_store import LiteRAGVectorStore

logger = logging.getLogger(__name__)

@dataclass
class ExplorationConfig:
    max_depth: int = 3
    min_relevance_threshold: float = 0.20 
    decay_factor: float = 0.7
    max_nodes_per_anchor: int = 50
    max_neighbors_per_hop: int = 10
    edge_weight_influence: float = 0.2
    
    # Penalties and Boosts
    degree_influence: float = 0.05       
    community_cohesion: float = 0.8      
    signal_amplification: float = 0.5    

class GraphExplorer:
    """
    Handles the logic of traversing the graph from a single starting point.
    Kept synchronous to run efficiently inside a ThreadPool.
    """
    def __init__(
        self,
        graph_data: GraphData,
        embedding_service: EmbeddingService,
        config: ExplorationConfig,
        vector_store: Optional[LiteRAGVectorStore] = None,
        entity_table_name: Optional[str] = None
    ):
        self.graph_data = graph_data
        self.embedding_service = embedding_service
        self.config = config
        self.vector_store = vector_store
        self.entity_table_name = entity_table_name
        
        # Local cache for embeddings during this exploration session
        self._entity_embeddings: Dict[str, np.ndarray] = {}
        self._embeddings_loaded = False
        
        # Thread lock for safe cache mutation
        self._cache_lock = threading.Lock()

    def _ensure_embeddings_loaded(self):
        """Lazy load all embeddings into memory for fast lookup."""
        if not self._embeddings_loaded and self.vector_store and self.entity_table_name:
            logger.info(f"Loading entity embeddings from {self.entity_table_name} into memory...")
            with self._cache_lock:
                if not self._embeddings_loaded:  # Double-check pattern
                    self._entity_embeddings = self.vector_store.get_all_embeddings(self.entity_table_name)
                    self._embeddings_loaded = True

    def explore_single_anchor(
        self, 
        anchor: Anchor, 
        query_embedding: np.ndarray,
        dynamic_threshold: float
    ) -> Subgraph:
        """
        Explores the graph starting from a single anchor.
        This method is CPU-bound (graph traversal) and I/O-bound (embeddings).
        """
        self._ensure_embeddings_loaded()

        entity = self.graph_data.entities.get(anchor.entity_title)
        if not entity:
            return Subgraph(anchor=anchor, nodes={}, max_depth_reached=0)
        
        nodes: Dict[str, ExploredNode] = {}
        visited: Set[str] = set()
        max_depth_reached = 0
        
        # Priority Queue logic via deque
        queue = deque([(anchor.entity_title, 0, anchor.score, [anchor.entity_title])])
        
        while queue and len(nodes) < self.config.max_nodes_per_anchor:
            current_title, depth, current_relevance, path = queue.popleft()
            
            if current_title in visited:
                continue
            visited.add(current_title)
            
            current_entity = self.graph_data.entities.get(current_title)
            if not current_entity:
                continue
            
            # Semantic check
            semantic_sim = self._get_semantic_similarity(current_entity, query_embedding)
            
            # Relevance update formula
            adjusted_relevance = current_relevance * (0.4 + 0.6 * semantic_sim)
            
            # Dynamic Pruning
            if adjusted_relevance < dynamic_threshold:
                continue
            
            nodes[current_title] = ExploredNode(
                entity=current_entity,
                relevance=adjusted_relevance,
                depth=depth,
                semantic_similarity=semantic_sim,
                path_from_anchor=path.copy(),
                discovered_by_anchors={anchor.entity_id}
            )
            max_depth_reached = max(max_depth_reached, depth)
            
            if depth < self.config.max_depth:
                neighbors = self._get_scored_neighbors(
                    current_entity,
                    adjusted_relevance,
                    query_embedding
                )
                
                for neighbor_title, neighbor_relevance in neighbors:
                    if neighbor_title not in visited:
                        new_path = path + [neighbor_title]
                        queue.append((neighbor_title, depth + 1, neighbor_relevance, new_path))
        
        return Subgraph(anchor=anchor, nodes=nodes, max_depth_reached=max_depth_reached)
    
    def _get_scored_neighbors(
        self,
        current_entity: Entity,
        current_relevance: float,
        query_embedding: np.ndarray
    ) -> List[Tuple[str, float]]:
        """Get neighbors sorted by relevance, applying Community-Aware Hub Penalties."""
        neighbors = get_neighbors(self.graph_data, current_entity.title)
        if not neighbors:
            return []
        
        scored_neighbors = []
        
        for neighbor in neighbors:
            # 1. Base decay
            neighbor_relevance = current_relevance * self.config.decay_factor
            
            # 2. Edge weight boost
            edge_weight = get_edge_weight(self.graph_data, current_entity.title, neighbor.title)
            neighbor_relevance *= (1 + self.config.edge_weight_influence * np.log1p(edge_weight))
            
            # 3. Community-Aware Hub Filtering
            if neighbor.degree > 10:
                base_penalty = self.config.degree_influence * np.log1p(neighbor.degree)
                
                # Reduce penalty for intra-community hubs to preserve local structure
                if (current_entity.community_id and neighbor.community_id and 
                    current_entity.community_id == neighbor.community_id):
                    final_penalty = base_penalty * (1.0 - self.config.community_cohesion)
                else:
                    final_penalty = base_penalty # Generic connector -> Full penalty
                
                neighbor_relevance *= (1.0 / (1.0 + final_penalty))
            
            scored_neighbors.append((neighbor.title, neighbor_relevance))
        
        # Sort and take top K
        scored_neighbors.sort(key=lambda x: x[1], reverse=True)
        return scored_neighbors[:self.config.max_neighbors_per_hop]
    
    def _get_semantic_similarity(self, entity: Entity, query_embedding: np.ndarray) -> float:
        # Fast path lock-free read
        entity_embedding = self._entity_embeddings.get(entity.title)
        if entity_embedding is None and entity.id:
            entity_embedding = self._entity_embeddings.get(entity.id)
        
        if entity_embedding is None:
            # Slow path: generate via API on the fly
            text = entity.description if entity.description else entity.title
            entity_embedding = self.embedding_service.embed(text)
            # Update cache safely
            with self._cache_lock:
                self._entity_embeddings[entity.title] = entity_embedding
        
        return self.embedding_service.similarity(query_embedding, entity_embedding)


class ParallelExplorer:
    """
    Orchestrates the exploration process using true concurrency.
    Uses asyncio + ThreadPoolExecutor to overcome the GIL for I/O bound operations.
    """
    def __init__(
        self,
        graph_data: GraphData,
        embedding_service: EmbeddingService,
        config: Optional[ExplorationConfig] = None,
        num_workers: int = 4,
        vector_store: Optional[LiteRAGVectorStore] = None,
        entity_table_name: Optional[str] = None
    ):
        self.config = config or ExplorationConfig()
        # Initialize the logic handler
        self.graph_explorer = GraphExplorer(
            graph_data, 
            embedding_service, 
            self.config,
            vector_store,
            entity_table_name
        )
        # Executor for running graph traversals in parallel threads
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        logger.info(f"Initialized ParallelExplorer with {num_workers} workers.")

    def warmup(self):
        """Forces loading of embeddings into memory to avoid latency on first query."""
        logger.info("Warming up ParallelExplorer: Pre-loading embeddings...")
        self.graph_explorer._ensure_embeddings_loaded()

    def explore(self, anchors: List[Anchor], query_embedding: np.ndarray) -> List[Subgraph]:
        """
        Synchronous entry point.
        Raises an error if invoked inside an async event loop (like FastAPI), enforcing safe patterns.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                raise RuntimeError(
                    "explore() cannot be called from a running event loop. "
                    "Use await engine.aquery() instead to prevent thread/loop deadlocks."
                )
        except RuntimeError as e:
            if "no running event loop" not in str(e):
                raise

        return asyncio.run(self._explore_async(anchors, query_embedding))

    async def aexplore(self, anchors: List[Anchor], query_embedding: np.ndarray) -> List[Subgraph]:
        """
        Native async entry point. Use this if you are inside an ASGI server (FastAPI).
        """
        return await self._explore_async(anchors, query_embedding)

    async def _explore_async(self, anchors: List[Anchor], query_embedding: np.ndarray) -> List[Subgraph]:
        """
        The core async orchestration method.
        """
        if not anchors:
            return []

        # 1. Pre-load embeddings once (Fast & Thread-safe read)
        self.graph_explorer._ensure_embeddings_loaded()

        # 2. Calculate Dynamic Threshold
        max_anchor_score = max([a.score for a in anchors])
        dynamic_threshold = self.config.min_relevance_threshold + (
            self.config.signal_amplification * max_anchor_score
        )
        
        # 3. Create Tasks
        loop = asyncio.get_running_loop()
        tasks = []

        for anchor in anchors:
            task = loop.run_in_executor(
                self.executor,
                self.graph_explorer.explore_single_anchor,
                anchor,
                query_embedding,
                dynamic_threshold
            )
            tasks.append(task)

        # 4. Gather Results (True Parallelism)
        results = await asyncio.gather(*tasks)
        
        return list(results)