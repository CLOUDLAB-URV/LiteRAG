"""
Data Loader for LiteRAG

Handles loading GraphRAG parquet files and building the graph structure.
Computes centrality metrics if not already cached.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import networkx as nx
from dataclasses import dataclass
import logging

from .models import Entity, Relationship, Community

logger = logging.getLogger(__name__)


@dataclass
class GraphData:
    """Container for all loaded graph data."""
    entities: Dict[str, Entity]  # title -> Entity
    entities_by_id: Dict[str, Entity]  # id -> Entity
    relationships: List[Relationship]
    communities: Dict[str, Community]  # id -> Community
    text_units: Dict[str, str]  # id -> text content
    graph: nx.Graph
    
    # Lookup structures
    entity_to_relationships: Dict[str, List[Relationship]]  # entity_title -> relationships
    entity_to_community: Dict[str, str]  # entity_id -> community_id


def load_parquet_safe(path: Path) -> Optional[pd.DataFrame]:
    """Load a parquet file with error handling."""
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        return None


def compute_centrality_metrics(graph: nx.Graph) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute PageRank and betweenness centrality for all nodes.
    
    Returns:
        Tuple of (pagerank_dict, betweenness_dict)
    """
    logger.info("Computing PageRank...")
    try:
        pagerank = nx.pagerank(graph, weight='weight')
    except nx.PowerIterationFailedConvergence:
        logger.warning("PageRank did not converge, using uniform values")
        pagerank = {n: 1.0 / len(graph) for n in graph.nodes()}
    
    logger.info("Computing betweenness centrality...")
    # For large graphs, use approximation
    if len(graph) > 1000:
        betweenness = nx.betweenness_centrality(graph, k=min(100, len(graph)))
    else:
        betweenness = nx.betweenness_centrality(graph)
    
    return pagerank, betweenness


def _safe_get(row, attr, default=None):
    """Safely extracts a column value from a namedtuple row handling missing NaN properly."""
    try:
        val = getattr(row, attr)
        # Use pd.isna only on scalars to avoid exceptions on embedded lists/arrays
        if isinstance(val, (int, float, str, bool, type(None))) and pd.isna(val):
            return default
        return val
    except AttributeError:
        return default


def load_graph_data(
    data_dir: str,
    cache_dir: Optional[str] = None,
    force_recompute: bool = False
) -> GraphData:
    """
    Load all GraphRAG data and build the graph structure.
    """
    data_path = Path(data_dir)
    cache_path = Path(cache_dir) if cache_dir else data_path / ".LiteRAG_cache"
    cache_file = cache_path / "graph_data.pkl"
    
    # Check for cached data
    if not force_recompute and cache_file.exists():
        logger.info(f"Loading cached graph data from {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Recomputing...")
            try:
                cache_file.unlink()
            except Exception:
                pass
    
    logger.info(f"Loading graph data from {data_dir}...")
    
    # Load parquet files
    entities_df = load_parquet_safe(data_path / "entities.parquet")
    relationships_df = load_parquet_safe(data_path / "relationships.parquet")
    communities_df = load_parquet_safe(data_path / "communities.parquet")
    community_reports_df = load_parquet_safe(data_path / "community_reports.parquet")
    text_units_df = load_parquet_safe(data_path / "text_units.parquet")
    
    if entities_df is None or relationships_df is None:
        raise ValueError("Could not load required entities or relationships data")
    
    # Build NetworkX graph
    logger.info("Building NetworkX graph...")
    graph = nx.Graph()
    
    # Add nodes with attributes
    for row in entities_df.itertuples(index=False):
        title = getattr(row, 'title', '')
        graph.add_node(title, **{
            'id': _safe_get(row, 'id', ''),
            'type': _safe_get(row, 'type', ''),
            'description': _safe_get(row, 'description', ''),
        })
    
    for row in relationships_df.itertuples(index=False):
        source = getattr(row, 'source')
        target = getattr(row, 'target')
        if source in graph.nodes and target in graph.nodes:
            graph.add_edge(
                source,
                target,
                weight=float(_safe_get(row, 'weight', 1.0)),
                description=str(_safe_get(row, 'description', ''))
            )
    
    logger.info(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Compute centrality metrics
    pagerank, betweenness = compute_centrality_metrics(graph)
    
    # Create Entity objects
    entities: Dict[str, Entity] = {}
    entities_by_id: Dict[str, Entity] = {}
    
    for row in entities_df.itertuples(index=False):
        title = getattr(row, 'title')
        eid = str(_safe_get(row, 'id', title))
        entity = Entity(
            id=eid,
            title=title,
            description=str(_safe_get(row, 'description', '')),
            type=str(_safe_get(row, 'type', '')),
            degree=graph.degree(title) if title in graph else 0,
            pagerank=pagerank.get(title, 0.0),
            betweenness_centrality=betweenness.get(title, 0.0),
            text_unit_ids=_parse_list_field(_safe_get(row, 'text_unit_ids', []))
        )
        entities[title] = entity
        entities_by_id[entity.id] = entity
    
    # Create Relationship objects
    relationships: List[Relationship] = []
    entity_to_relationships: Dict[str, List[Relationship]] = {}
    
    for row in relationships_df.itertuples(index=False):
        source = getattr(row, 'source')
        target = getattr(row, 'target')
        default_id = f"{source}_{target}"
        
        rel = Relationship(
            id=str(_safe_get(row, 'id', default_id)),
            source=source,
            target=target,
            description=str(_safe_get(row, 'description', '')),
            weight=float(_safe_get(row, 'weight', 1.0))
        )
        relationships.append(rel)
        
        # Build lookup
        if rel.source not in entity_to_relationships:
            entity_to_relationships[rel.source] = []
        entity_to_relationships[rel.source].append(rel)
        
        if rel.target not in entity_to_relationships:
            entity_to_relationships[rel.target] = []
        entity_to_relationships[rel.target].append(rel)
    
    # Create Community objects
    communities: Dict[str, Community] = {}
    entity_to_community: Dict[str, str] = {}
    
    if community_reports_df is not None and communities_df is not None:
        for row in community_reports_df.itertuples(index=False):
            comm_id_val = _safe_get(row, 'community', _safe_get(row, 'id', ''))
            comm_id = str(comm_id_val)
            
            # Find entity_ids from communities_df
            entity_ids = []
            comm_row = communities_df[communities_df['community'] == comm_id_val]
            if len(comm_row) > 0:
                entity_ids = _parse_list_field(comm_row.iloc[0].get('entity_ids', []))
            
            community = Community(
                id=comm_id,
                title=str(_safe_get(row, 'title', '')),
                level=int(_safe_get(row, 'level', 0)),
                summary=str(_safe_get(row, 'summary', '')),
                full_content=str(_safe_get(row, 'full_content', '')),
                entity_ids=entity_ids
            )
            communities[comm_id] = community
            
            # Map entities to community
            for eid in entity_ids:
                entity_to_community[eid] = comm_id
    
    # Load text units
    text_units: Dict[str, str] = {}
    if text_units_df is not None:
        for row in text_units_df.itertuples(index=False):
            text_units[str(getattr(row, 'id'))] = str(_safe_get(row, 'text', ''))
    
    # Update entity community references
    for entity_id, comm_id in entity_to_community.items():
        if entity_id in entities_by_id:
            entities_by_id[entity_id].community_id = comm_id
    
    # Create GraphData
    graph_data = GraphData(
        entities=entities,
        entities_by_id=entities_by_id,
        relationships=relationships,
        communities=communities,
        text_units=text_units,
        graph=graph,
        entity_to_relationships=entity_to_relationships,
        entity_to_community=entity_to_community
    )
    
    # Cache the data
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(graph_data, f)
        logger.info(f"Cached graph data to {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to cache data: {e}")
    
    return graph_data


def _parse_list_field(value) -> List[str]:
    """Parse a list field that might be stored as string or list."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist()]
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        # Try to parse as list
        if value.startswith('['):
            try:
                import ast
                return [str(v) for v in ast.literal_eval(value)]
            except:
                pass
        # Split by comma
        return [s.strip() for s in value.split(',') if s.strip()]
    try:
        if pd.isna(value):
            return []
    except (ValueError, TypeError):
        pass
    return [str(value)]


# Convenience functions

def get_neighbors(graph_data: GraphData, entity_title: str) -> List[Entity]:
    """Get neighboring entities for a given entity."""
    if entity_title not in graph_data.graph:
        return []
    
    neighbors = []
    for neighbor_title in graph_data.graph.neighbors(entity_title):
        if neighbor_title in graph_data.entities:
            neighbors.append(graph_data.entities[neighbor_title])
    
    return neighbors


def get_edge_weight(graph_data: GraphData, source: str, target: str) -> float:
    """Get edge weight between two entities."""
    if graph_data.graph.has_edge(source, target):
        return graph_data.graph[source][target].get('weight', 1.0)
    return 0.0


def get_entity_relationships(graph_data: GraphData, entity_title: str) -> List[Relationship]:
    """Get all relationships for an entity."""
    return graph_data.entity_to_relationships.get(entity_title, [])