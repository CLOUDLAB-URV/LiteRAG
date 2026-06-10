"""
Anchor Discovery for LiteRAG

Multi-strategy approach to find starting points for graph exploration.
Uses semantic (LanceDB), keyword, and community-based matching.
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import numpy as np
import logging

from .models import Entity, Community, Anchor, AnchorSource
from .data_loader import GraphData
from .embeddings import EmbeddingService
from .vector_store import (
    LiteRAGVectorStore, 
    GRAPHRAG_ENTITY_TABLE, 
    LiteRAG_ENTITY_TABLE,
    GRAPHRAG_COMMUNITY_TABLE,
    LiteRAG_COMMUNITY_TABLE
)

logger = logging.getLogger(__name__)


class AnchorDiscovery:
    """
    Multi-strategy anchor discovery using LanceDB for vector operations.
    
    Strategies:
    1. Semantic: Embed query, search LanceDB for similar entities
    2. Keyword Exact: N-gram matching on entity titles
    3. Keyword Fuzzy: Substring/token matching
    4. Community: Match query to community summaries via LanceDB
    """
    
    def __init__(
        self,
        graph_data: GraphData,
        embedding_service: EmbeddingService,
        vector_store: LiteRAGVectorStore,
        semantic_weight: float = 0.4,
        keyword_exact_weight: float = 0.3,
        keyword_fuzzy_weight: float = 0.15,
        community_weight: float = 0.15,
        min_score_threshold: float = 0.3,
        max_anchors: int = 8,
        community_boost_factor: float = 0.1
    ):
        """
        Initialize anchor discovery.
        
        Args:
            graph_data: Loaded graph data
            embedding_service: Service for generating embeddings
            vector_store: LanceDB vector store (required)
            semantic_weight: Weight for semantic matching
            keyword_exact_weight: Weight for exact keyword matching
            keyword_fuzzy_weight: Weight for fuzzy keyword matching
            community_weight: Weight for community matching
            min_score_threshold: Minimum score to be an anchor
            max_anchors: Maximum number of anchors to return
            community_boost_factor: Boost for entities in matched communities
        """
        self.graph_data = graph_data
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        
        self.semantic_weight = semantic_weight
        self.keyword_exact_weight = keyword_exact_weight
        self.keyword_fuzzy_weight = keyword_fuzzy_weight
        self.community_weight = community_weight
        self.min_score_threshold = min_score_threshold
        self.max_anchors = max_anchors
        self.community_boost_factor = community_boost_factor
        
        # Determine which table to use for entity embeddings
        self._entity_table_name: Optional[str] = None
        self._community_table_name: Optional[str] = None
        
        # Check for existing GraphRAG embeddings first
        if self.vector_store.has_table(GRAPHRAG_ENTITY_TABLE):
            self._entity_table_name = GRAPHRAG_ENTITY_TABLE
            logger.info(f"✓ Using existing GraphRAG entity embeddings from '{GRAPHRAG_ENTITY_TABLE}'")
        elif self.vector_store.has_table(LiteRAG_ENTITY_TABLE):
            self._entity_table_name = LiteRAG_ENTITY_TABLE
            logger.info(f"✓ Using existing LiteRAG entity embeddings from '{LiteRAG_ENTITY_TABLE}'")
        
        # Check for community embeddings
        if self.vector_store.has_table(GRAPHRAG_COMMUNITY_TABLE):
            self._community_table_name = GRAPHRAG_COMMUNITY_TABLE
            logger.info(f"✓ Using existing GraphRAG community embeddings from '{GRAPHRAG_COMMUNITY_TABLE}'")
        elif self.vector_store.has_table(LiteRAG_COMMUNITY_TABLE):
            self._community_table_name = LiteRAG_COMMUNITY_TABLE
            logger.info(f"✓ Using existing LiteRAG community embeddings from '{LiteRAG_COMMUNITY_TABLE}'")
        
        # Build lookup structures for keyword matching
        self._entity_tokens: Dict[str, Set[str]] = {}
        self._token_to_entities: Dict[str, Set[str]] = defaultdict(set)
        self._preprocess_entities()
    
    def _preprocess_entities(self):
        """Preprocess entities for keyword matching."""
        for title in self.graph_data.entities:
            tokens = self._tokenize(title)
            self._entity_tokens[title] = tokens
            for token in tokens:
                self._token_to_entities[token].add(title)
    
    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text for matching."""
        text = text.lower()
        tokens = set(re.findall(r'\b\w+\b', text))
        return tokens
    
    def _extract_ngrams(self, text: str, n: int = 2) -> Set[str]:
        """Extract character n-grams for fuzzy matching."""
        text = text.lower()
        ngrams = set()
        for i in range(len(text) - n + 1):
            ngrams.add(text[i:i+n])
        return ngrams
    
    def ensure_embeddings(self):
        """
        Ensure entity embeddings are available in LanceDB.
        
        If no embeddings exist, compute and store them in LanceDB.
        """
        if self._entity_table_name:
            count = self.vector_store.table_count(self._entity_table_name)
            logger.info(f"Using {count} pre-indexed entity embeddings from LanceDB")
            return
        
        # Need to compute embeddings and store in LanceDB
        logger.info(f"No entity embeddings found. Computing embeddings for {len(self.graph_data.entities)} entities...")
        
        entities_data = [
            (title, entity.description or title)
            for title, entity in self.graph_data.entities.items()
        ]
        
        titles = [e[0] for e in entities_data]
        texts = [e[1] for e in entities_data]
        
        embeddings = self.embedding_service.embed_batch(texts)
        
        # Store in LanceDB
        logger.info(f"Storing entity embeddings in LanceDB table '{LiteRAG_ENTITY_TABLE}'...")
        records = [
            {
                "id": title,
                "title": title,
                "text": texts[i],
                "vector": embeddings[i].tolist()
            }
            for i, title in enumerate(titles)
        ]
        self.vector_store.add_embeddings(LiteRAG_ENTITY_TABLE, records, mode="overwrite")
        self._entity_table_name = LiteRAG_ENTITY_TABLE
        logger.info(f"✓ Stored {len(records)} entity embeddings in LanceDB")
        
        # Also compute and store community embeddings if needed
        if not self._community_table_name and self.graph_data.communities:
            logger.info(f"Computing embeddings for {len(self.graph_data.communities)} communities...")
            
            comm_data = [
                (cid, comm.summary or comm.title)
                for cid, comm in self.graph_data.communities.items()
            ]
            
            cids = [c[0] for c in comm_data]
            comm_texts = [c[1] for c in comm_data]
            
            comm_embeddings = self.embedding_service.embed_batch(comm_texts)
            
            comm_records = [
                {
                    "id": cid,
                    "text": comm_texts[i],
                    "vector": comm_embeddings[i].tolist()
                }
                for i, cid in enumerate(cids)
            ]
            self.vector_store.add_embeddings(LiteRAG_COMMUNITY_TABLE, comm_records, mode="overwrite")
            self._community_table_name = LiteRAG_COMMUNITY_TABLE
            logger.info(f"✓ Stored {len(comm_records)} community embeddings in LanceDB")
    
    def discover_anchors(
        self,
        query: str,
        expanded_query: Optional[str] = None
    ) -> List[Anchor]:
        """
        Discover anchor entities using multiple strategies.
        
        Args:
            query: Original user query
            expanded_query: Optional expanded version of the query
            
        Returns:
            List of Anchor objects sorted by score
        """
        # Ensure embeddings are in LanceDB
        self.ensure_embeddings()
        
        # Use expanded query if provided
        search_text = expanded_query if expanded_query else query
        
        # Collect scores from each strategy
        scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # 1. Semantic matching (via LanceDB)
        semantic_anchors = self._semantic_match(search_text)
        for title, score in semantic_anchors:
            scores[title][AnchorSource.SEMANTIC.value] = score
        
        # 2. Keyword exact matching
        keyword_exact_anchors = self._keyword_exact_match(search_text)
        for title, score in keyword_exact_anchors:
            scores[title][AnchorSource.KEYWORD_EXACT.value] = score
        
        # 3. Keyword fuzzy matching
        keyword_fuzzy_anchors = self._keyword_fuzzy_match(search_text)
        for title, score in keyword_fuzzy_anchors:
            scores[title][AnchorSource.KEYWORD_FUZZY.value] = score
        
        # 4. Community matching (via LanceDB)
        community_entities = self._community_match(search_text)
        for title, score in community_entities:
            scores[title][AnchorSource.COMMUNITY.value] = score
        
        # Combine scores
        anchors: List[Anchor] = []
        for title, source_scores in scores.items():
            entity = self.graph_data.entities.get(title)
            if not entity:
                # LanceDB may return UUIDs instead of titles for GraphRAG tables
                entity = self.graph_data.entities_by_id.get(title)
            if not entity:
                continue
            
            # Weighted combination
            combined_score = 0.0
            sources: Set[AnchorSource] = set()
            
            if AnchorSource.SEMANTIC.value in source_scores:
                combined_score += self.semantic_weight * source_scores[AnchorSource.SEMANTIC.value]
                sources.add(AnchorSource.SEMANTIC)
            
            if AnchorSource.KEYWORD_EXACT.value in source_scores:
                combined_score += self.keyword_exact_weight * source_scores[AnchorSource.KEYWORD_EXACT.value]
                sources.add(AnchorSource.KEYWORD_EXACT)
            
            if AnchorSource.KEYWORD_FUZZY.value in source_scores:
                combined_score += self.keyword_fuzzy_weight * source_scores[AnchorSource.KEYWORD_FUZZY.value]
                sources.add(AnchorSource.KEYWORD_FUZZY)
            
            if AnchorSource.COMMUNITY.value in source_scores:
                combined_score += self.community_weight * source_scores[AnchorSource.COMMUNITY.value]
                sources.add(AnchorSource.COMMUNITY)
            
            # Bonus for appearing in multiple strategies
            if len(sources) > 1:
                combined_score *= (1 + 0.1 * (len(sources) - 1))
            
            if combined_score >= self.min_score_threshold:
                anchor = Anchor(
                    entity_id=entity.id,
                    entity_title=entity.title,
                    score=combined_score,
                    sources=sources
                )
                anchors.append(anchor)
        
        # Sort by score and limit
        anchors.sort(key=lambda a: a.score, reverse=True)
        anchors = anchors[:self.max_anchors]
        
        logger.info(f"Discovered {len(anchors)} anchors")
        for anchor in anchors:
            sources_str = ", ".join(s.value for s in anchor.sources)
            logger.debug(f"  {anchor.entity_title}: {anchor.score:.3f} ({sources_str})")
        
        return anchors
    
    def _semantic_match(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Find semantically similar entities using LanceDB."""
        if not self._entity_table_name:
            logger.warning("No entity embeddings available for semantic search")
            return []
        
        query_embedding = self.embedding_service.embed(query)
        results = self.vector_store.similarity_search(
            self._entity_table_name,
            query_embedding,
            k=top_k
        )
        return results
    
    def _keyword_exact_match(self, query: str) -> List[Tuple[str, float]]:
        """Find entities with exact keyword matches."""
        query_tokens = self._tokenize(query)
        results: Dict[str, float] = {}
        
        for token in query_tokens:
            if token in self._token_to_entities:
                for title in self._token_to_entities[token]:
                    entity_tokens = self._entity_tokens[title]
                    overlap = len(query_tokens & entity_tokens)
                    score = overlap / max(len(entity_tokens), 1)
                    if title not in results or score > results[title]:
                        results[title] = score
        
        return list(results.items())
    
    def _keyword_fuzzy_match(self, query: str) -> List[Tuple[str, float]]:
        """Optimized fuzzy matching"""
        query_lower = query.lower()
        results: Dict[str, float] = {}
        
        # Early exit for very short queries
        if len(query_lower) < 3: return []

        # Performance optimization: Token-based candidate pre-filtering
        query_tokens = self._tokenize(query_lower)
        candidate_titles = set()
        for token in query_tokens:
            if len(token) > 3:
                candidate_titles.update(self._token_to_entities.get(token, []))
        
        # Use filtered candidates if available, otherwise full sweep
        search_set = candidate_titles if candidate_titles else self.graph_data.entities
        
        for title in search_set:
            title_lower = title.lower()
            
            # Case 1: Entity title is contained IN the query
            if title_lower in query_lower:
                results[title] = 1.0
            # Case 2: Query is contained in entity title
            elif query_lower in title_lower:
                results[title] = len(query_lower) / len(title_lower)
            # Case 3: N-gram Jaccard similarity for partial matches
            else:
                query_ngrams = self._extract_ngrams(query)
                title_ngrams = self._extract_ngrams(title)
                
                if query_ngrams and title_ngrams:
                    intersection = len(query_ngrams & title_ngrams)
                    union = len(query_ngrams | title_ngrams)
                    if union > 0:
                        jaccard = intersection / union
                        if jaccard > 0.3:
                            results[title] = jaccard
            
        return list(results.items())
    
    def _community_match(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find entities in communities matching the query via LanceDB."""
        if not self._community_table_name:
            return []
        
        query_embedding = self.embedding_service.embed(query)
        
        # Find matching communities via LanceDB
        top_communities = self.vector_store.similarity_search(
            self._community_table_name,
            query_embedding,
            k=top_k
        )
        
        results: Dict[str, float] = {}
        
        for comm_id, comm_score in top_communities:
            if comm_score < 0.3:
                continue
                
            community = self.graph_data.communities.get(comm_id)
            if not community:
                # LanceDB may return UUIDs instead of integer community IDs
                community = self.graph_data.communities_by_uuid.get(comm_id)
            if not community:
                continue
            
            for entity_id in community.entity_ids:
                entity = self.graph_data.entities_by_id.get(entity_id)
                if entity:
                    entity_score = comm_score
                    if entity.pagerank > 0:
                        entity_score *= (1 + self.community_boost_factor * np.log1p(entity.pagerank * 1000))
                    
                    if entity.title not in results or entity_score > results[entity.title]:
                        results[entity.title] = entity_score
        
        return list(results.items())
