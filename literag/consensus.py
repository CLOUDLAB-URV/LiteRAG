"""
Consensus Merge and Ranking for LiteRAG
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import logging

from .models import Entity, Anchor, ExploredNode, Subgraph, RankedEntity
from .data_loader import GraphData, get_entity_relationships

logger = logging.getLogger(__name__)


class ConsensusMerger:
    def __init__(
        self,
        graph_data: GraphData,
        intersection_weight: float = 0.45,
        semantic_weight: float = 0.40,
        structural_weight: float = 0.05,
        proximity_weight: float = 0.10,
        anchor_boost: float = 1.2,
        hub_boost: float = 0.0
    ):
        self.graph_data = graph_data
        self.intersection_weight = intersection_weight
        self.semantic_weight = semantic_weight
        self.structural_weight = structural_weight
        self.proximity_weight = proximity_weight
        self.anchor_boost = anchor_boost
        self.hub_boost = hub_boost
    
    def merge_and_rank(
        self,
        subgraphs: List[Subgraph],
        anchors: List[Anchor],
        max_entities: int = 50
    ) -> List[RankedEntity]:
        if not subgraphs:
            return []
        
        anchor_ids = {a.entity_id for a in anchors}
        anchor_titles = {a.entity_title for a in anchors}
        num_workers = len(subgraphs)
        merged: Dict[str, MergedNode] = {}
        
        for subgraph in subgraphs:
            for title, node in subgraph.nodes.items():
                if title not in merged:
                    merged[title] = MergedNode(node.entity)
                merged[title].add_observation(node, subgraph.anchor.entity_id)
        
        ranked_entities: List[RankedEntity] = []
        
        for title, merged_node in merged.items():
            intersection_score = merged_node.found_by_count / num_workers
            semantic_score = merged_node.avg_semantic_similarity
            structural_score = self._compute_structural_score(merged_node.entity)
            proximity_score = self._compute_proximity_score(merged_node.avg_depth)
            
            final_score = (
                self.intersection_weight * intersection_score +
                self.semantic_weight * semantic_score +
                self.structural_weight * structural_score +
                self.proximity_weight * proximity_score
            )
            
            # Apply relevance factor from exploration
            final_score *= merged_node.avg_relevance
            
            is_anchor = (merged_node.entity.id in anchor_ids or merged_node.entity.title in anchor_titles)
            if is_anchor:
                final_score *= self.anchor_boost
            
            relationships = get_entity_relationships(self.graph_data, title)
            community = None
            if merged_node.entity.community_id:
                community = self.graph_data.communities.get(merged_node.entity.community_id)
            
            ranked_entity = RankedEntity(
                entity=merged_node.entity,
                final_score=final_score,
                intersection_score=intersection_score,
                semantic_score=semantic_score,
                structural_score=structural_score,
                proximity_score=proximity_score,
                found_by_workers=merged_node.found_by_count,
                is_anchor=is_anchor,
                avg_depth=merged_node.avg_depth,
                relationships=relationships,
                community=community
            )
            ranked_entities.append(ranked_entity)
        
        ranked_entities.sort(key=lambda x: x.final_score, reverse=True)
        return ranked_entities[:max_entities]
    
    def _compute_structural_score(self, entity: Entity) -> float:
        # Structural scores are weighted as secondary factors to prioritize 
        # semantic relevance while providing a tie-breaking mechanism based on 
        # graph centrality.
        pagerank_score = np.log1p(entity.pagerank * 10000) / 10
        betweenness_score = np.log1p(entity.betweenness_centrality * 1000) / 10
        pagerank_score = min(1.0, max(0.0, pagerank_score))
        betweenness_score = min(1.0, max(0.0, betweenness_score))
        return 0.5 * pagerank_score + 0.5 * betweenness_score
    
    def _compute_proximity_score(self, avg_depth: float) -> float:
        return np.exp(-0.5 * avg_depth)


class MergedNode:
    def __init__(self, entity: Entity):
        self.entity = entity
        self.observations: List[ExploredNode] = []
        self.discovered_by: Set[str] = set()
    
    def add_observation(self, node: ExploredNode, anchor_id: str):
        self.observations.append(node)
        self.discovered_by.add(anchor_id)
    
    @property
    def found_by_count(self) -> int:
        return len(self.discovered_by)
    
    @property
    def avg_semantic_similarity(self) -> float:
        if not self.observations: return 0.0
        return sum(o.semantic_similarity for o in self.observations) / len(self.observations)
    
    @property
    def avg_depth(self) -> float:
        if not self.observations: return 0.0
        return sum(o.depth for o in self.observations) / len(self.observations)
    
    @property
    def avg_relevance(self) -> float:
        if not self.observations: return 0.0
        return sum(o.relevance for o in self.observations) / len(self.observations)

def filter_redundant_entities(ranked_entities: List[RankedEntity], similarity_threshold: float = 0.9) -> List[RankedEntity]:
    if not ranked_entities: return []
    filtered = [ranked_entities[0]]
    
    def title_similarity(t1: str, t2: str) -> float:
        tokens1 = set(t1.lower().split())
        tokens2 = set(t2.lower().split())
        if not tokens1 or not tokens2: return 0.0
        return len(tokens1 & tokens2) / len(tokens1 | tokens2)
    
    for entity in ranked_entities[1:]:
        is_redundant = False
        for included in filtered:
            if title_similarity(entity.entity.title, included.entity.title) > similarity_threshold:
                is_redundant = True
                break
        if not is_redundant:
            filtered.append(entity)
    return filtered