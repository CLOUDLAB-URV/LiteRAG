"""
LiteRAG Quick Start Example
---------------------------
This script demonstrates how to initialize LiteRAG, query an existing
Knowledge Graph, and view the cost/latency telemetry.

Prerequisite: Ensure you have GraphRAG parquet files in your data directory
(as configured in literag_config.yaml) and your API key in your environment.
"""

import asyncio
import os
import sys
from literag import LiteRAG, LiteRAGConfig


def print_anchors(anchors):
    """Pretty-print the discovered anchors with their scores and sources."""
    if not anchors:
        print("   (none)")
        return
    for i, anchor in enumerate(anchors, 1):
        sources_str = ", ".join(s.value for s in anchor.sources)
        print(f"   {i}. {anchor.entity_title}")
        print(f"      Score: {anchor.score:.3f}  |  Sources: {sources_str}")


def print_subgraphs(subgraphs):
    """Pretty-print the entities explored in each subgraph."""
    if not subgraphs:
        print("   (none)")
        return
    for sg in subgraphs:
        anchor_title = sg.anchor.entity_title
        node_count = len(sg.nodes)
        max_depth = sg.max_depth_reached
        print(f"\n   📍 Anchor: {anchor_title}")
        print(f"      Nodes: {node_count}  |  Max Depth: {max_depth}")
        if sg.nodes:
            print("      Entities:")
            for node_id, node in sg.nodes.items():
                print(f"         • {node.entity.title}")
                print(f"           Relevance: {node.relevance:.3f}  |  "
                      f"Depth: {node.depth}  |  Semantic Sim: {node.semantic_similarity:.3f}")


def print_ranked_entities(ranked_entities):
    """Pretty-print the entities that made it into the final prompt."""
    if not ranked_entities:
        print("   (none)")
        return
    for i, re in enumerate(ranked_entities, 1):
        print(f"   {i}. {re.entity.title}")
        print(f"      Final Score: {re.final_score:.3f}  |  "
              f"Semantic: {re.semantic_score:.3f}  |  "
              f"Structural: {re.structural_score:.3f}  |  "
              f"Proximity: {re.proximity_score:.3f}")


async def main() -> None:
    print("⚙️  Loading Configuration...")
    # Load settings (paths, thresholds, model configs) from YAML
    config = LiteRAGConfig.from_yaml("literag_config.yaml")

    # 1. Initialize Engine
    print("🚀 Initializing LiteRAG Engine (Loading Graph & LanceDB)...")
    engine = LiteRAG(config)

    # force_reload=False ensures we use cached graph computations if available,
    # making startup incredibly fast after the first run.
    engine.initialize(force_reload=False)

    # 2. Define the Query
    # Allow passing a query via command line, otherwise fallback to default
    query = sys.argv[1] if len(sys.argv) > 1 else "What technologies enable edge computing for IoT?"
    print(f"\n🔍 Executing Query: '{query}'")
    print("-" * 64)

    # 3. Execute Async Query
    # expand_query=True asks the LLM to generate synonyms before graph traversal
    result = await engine.aquery(query, expand_query=True)

    if not result.success:
        print(f"❌ Error occurred: {result.error}")
        return

    # 4. Output Results
    print("\n✅ ANSWER")
    print("=" * 64)
    print(result.answer)

    # 5. Show Graph Retrieval Details
    print("\n🔎 GRAPH RETRIEVAL DETAILS")
    print("=" * 64)

    print(f"\n⚓  Anchors Found ({len(result.anchors)})")
    print("-" * 64)
    print_anchors(result.anchors)

    print(f"\n🕸️  Subgraphs Explored ({len(result.subgraphs)})")
    print("-" * 64)
    print_subgraphs(result.subgraphs)

    print(f"\n🧠  Entities in Prompt ({len(result.ranked_entities)})")
    print("-" * 64)
    print_ranked_entities(result.ranked_entities)

    # 6. Telemetry
    print("\n📊 TELEMETRY & PERFORMANCE METRICS")
    print("=" * 64)
    print(f"⏱️\tLatency:\t\t{result.latency_ms / 1000:<4.2f} seconds")
    print(f"⚓\tAnchors Found:\t\t{len(result.anchors):<4} starting nodes")
    print(f"🕸️\tEntities Explored:\t{result.total_entities_explored:<4} nodes traversed")
    print(f"🧠\tEntities in Prompt:\t{result.entities_in_context:<4} highly-relevant nodes kept")
    print(f"🪙\tContext Tokens Used:\t{result.tokens_used:<4}")
    print(f"💸\tLLM Calls:\t\t{result.llm_calls:<4}")
    print(f"📈\tToken Breakdown\t{result.prompt_tokens} (Prompt) / {result.completion_tokens} (Completion)")
    print("=" * 64)


if __name__ == "__main__":
    # Standard Python asyncio entry point
    asyncio.run(main())