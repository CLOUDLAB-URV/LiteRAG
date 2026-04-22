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

async def main() -> None:
    print("⚙️ Loading Configuration...")
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