"""
Context Assembly for LiteRAG - Modified for Safety Net
"""

from typing import Dict, List, Optional, Tuple, Set
import logging

from .models import Entity, RankedEntity
from .data_loader import GraphData

logger = logging.getLogger(__name__)


class ContextAssembler:
    def __init__(
        self,
        graph_data: GraphData,
        max_context_tokens: int = 10000,
        text_unit_token_budget: float = 0.50,
        entity_token_budget: float = 0.15,
        relationship_token_budget: float = 0.15,
        community_token_budget: float = 0.15,
        source_token_budget: float = 0.05,
    ):
        self.graph_data = graph_data
        self.max_context_tokens = max_context_tokens
        
        self.text_unit_token_budget = text_unit_token_budget
        self.entity_token_budget = entity_token_budget
        self.relationship_token_budget = relationship_token_budget
        self.community_token_budget = community_token_budget
        self.source_token_budget = source_token_budget
    
    def assemble_context(
        self,
        ranked_entities: List[RankedEntity],
        query: str,
        safety_net_text_units: List[str] = None
    ) -> Tuple[str, int]:
        
        if not safety_net_text_units:
            safety_net_text_units = []

        # Budgets
        entity_budget = int(self.max_context_tokens * self.entity_token_budget)
        text_unit_budget = int(self.max_context_tokens * self.text_unit_token_budget)
        relationship_budget = int(self.max_context_tokens * self.relationship_token_budget)
        community_budget = int(self.max_context_tokens * self.community_token_budget)
        
        # 1. Safety Net Text (High Priority)
        text_unit_section, text_unit_tokens = self._build_text_unit_section(
            ranked_entities, text_unit_budget, safety_net_text_units
        )

        # 2. Reasoning Chains (Show connections between entities)
        # We build "mini-stories" based on the graph traversal paths
        chain_section, chain_tokens = self._build_reasoning_chains(
            ranked_entities, relationship_budget
        )

        # 3. Entity Details
        entity_section, entity_tokens = self._build_entity_section(
            ranked_entities, entity_budget
        )
        
        relationship_section, rel_tokens = self._build_relationship_section(
            ranked_entities, relationship_budget
        )
        
        community_section, comm_tokens = self._build_community_section(
            ranked_entities, community_budget
        )
        
        sections = []
        if text_unit_section:
            sections.append("## Direct Evidence (Source Text)\n" + text_unit_section)
        
        # Section order: Evidence -> Graph Logic -> Entity Definitions
        if chain_section:
            sections.append("## Graph Reasoning Chains (Connections)\n" + chain_section)
            
        if entity_section:
            sections.append("## Entity Definitions\n" + entity_section)
        
        if relationship_section:
            sections.append("## Relationships\n\n" + relationship_section)
        
        if community_section:
            sections.append("## Community Context\n\n" + community_section)
        
        context = "\n\n".join(sections)
        total_tokens = entity_tokens + text_unit_tokens + rel_tokens + comm_tokens
        
        return context, total_tokens

    def _build_reasoning_chains(self, ranked_entities: List[RankedEntity], budget: int) -> Tuple[str, int]:
        """
        Reconstructs 1-hop and 2-hop paths found during exploration to show causality.
        """
        chains = []
        tokens_used = 0
        seen_edges = set()
        
        # Get high-relevance entities
        top_entities = [re.entity for re in ranked_entities[:10]]
        top_titles = {e.title for e in top_entities}
        
        for re in ranked_entities[:15]: # Look at top 15 results
            entity = re.entity
            
            # Find relationships that connect to other top entities
            for rel in re.relationships:
                if rel.target in top_titles and rel.target != entity.title:
                    edge_key = tuple(sorted((entity.title, rel.target)))
                    if edge_key in seen_edges:
                        continue
                    
                    # Format: Entity A --[RELATION]--> Entity B: Description
                    chain = f"• **{entity.title}** is connected to **{rel.target}** via *{rel.description}*"
                    
                    tks = self._estimate_tokens(chain)
                    if tokens_used + tks > budget:
                        break
                        
                    chains.append(chain)
                    tokens_used += tks
                    seen_edges.add(edge_key)
        
        return "\n".join(chains), tokens_used
    
    def _build_entity_section(self, ranked_entities, token_budget) -> Tuple[str, int]:
        lines = []
        tokens_used = 0
        for ranked_entity in ranked_entities:
            entity = ranked_entity.entity
            desc = entity.description.strip() if entity.description else ""
            line = f"**{entity.title}**: {desc}"
            
            line_tokens = self._estimate_tokens(line)
            if tokens_used + line_tokens > token_budget:
                break
            lines.append(line)
            tokens_used += line_tokens
        return "\n".join(lines), tokens_used
    
    def _build_text_unit_section(
        self,
        ranked_entities: List[RankedEntity],
        token_budget: int,
        safety_net_units: List[str]
    ) -> Tuple[str, int]:
        """
        Build text units section, prioritizing Safety Net units first, 
        then units from high-ranked entities.
        """
        if token_budget <= 0:
            return "", 0
        
        chunks = []
        tokens_used = 0
        seen_texts = set()
        
        # 1. Add Safety Net Units FIRST (Top Priority)
        for text in safety_net_units:
            text = text.strip()
            if not text or text in seen_texts:
                continue
            
            text_tokens = self._estimate_tokens(text)
            if tokens_used + text_tokens > token_budget:
                break
                
            chunks.append(f"[Direct Result]: {text}")
            tokens_used += text_tokens
            seen_texts.add(text)
        
        # 2. Add Units from Ranked Entities (if budget remains)
        if tokens_used < token_budget:
            # Sort units by best entity rank
            text_unit_info = {}
            for rank, re in enumerate(ranked_entities):
                for uid in re.entity.text_unit_ids:
                    if uid not in text_unit_info:
                        text_unit_info[uid] = {'rank': rank, 'count': 0}
                    text_unit_info[uid]['rank'] = min(text_unit_info[uid]['rank'], rank)
                    text_unit_info[uid]['count'] += 1
            
            sorted_units = sorted(
                text_unit_info.items(),
                key=lambda x: (x[1]['rank'], -x[1]['count'])
            )
            
            for uid, info in sorted_units:
                text = self.graph_data.text_units.get(uid, "").strip()
                if not text or len(text) < 50 or text in seen_texts:
                    continue
                
                text_tokens = self._estimate_tokens(text)
                if tokens_used + text_tokens > token_budget:
                    break
                
                # Truncate if chunk is huge
                if text_tokens > 500:
                    text = text[:2000] + "..."
                    text_tokens = 500
                    
                chunks.append(f"[Graph Context]: {text}")
                tokens_used += text_tokens
                seen_texts.add(text)
        
        return "\n\n".join(chunks), tokens_used
    
    def _build_relationship_section(self, ranked_entities, token_budget) -> Tuple[str, int]:
        lines = []
        tokens_used = 0
        seen_rels = set()
        titles = {re.entity.title for re in ranked_entities}
        
        for re in ranked_entities:
            for rel in re.relationships:
                if rel.source not in titles or rel.target not in titles:
                    continue
                key = f"{min(rel.source, rel.target)}|{max(rel.source, rel.target)}"
                if key in seen_rels:
                    continue
                seen_rels.add(key)
                
                line = f"- {rel.source} -> {rel.target}: {rel.description}"
                tks = self._estimate_tokens(line)
                if tokens_used + tks > token_budget:
                    break
                lines.append(line)
                tokens_used += tks
        return "\n".join(lines), tokens_used

    def _build_community_section(self, ranked_entities, token_budget) -> Tuple[str, int]:
        lines = []
        tokens_used = 0
        seen_comms = set()
        
        for re in ranked_entities:
            if not re.community or re.community.id in seen_comms:
                continue
            seen_comms.add(re.community.id)
            
            summary = re.community.summary or re.community.title
            line = f"**Community {re.community.id}**: {summary}"
            tks = self._estimate_tokens(line)
            if tokens_used + tks > token_budget:
                break
            lines.append(line)
            tokens_used += tks
        return "\n".join(lines), tokens_used

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4 + 1