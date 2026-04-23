# 📚 LiteRAG Documentation

Welcome to the official documentation for **LiteRAG**, a cost-efficient, zero-LLM graph traversal engine for Retrieval-Augmented Generation. 

LiteRAG completely removes the LLM from the graph navigation loop, replacing it with highly optimized algorithmic search. This drops query latency to $O(1)$ and reduces API token costs by up to 99% compared to traditional GraphRAG implementations, while actually improving multi-hop reasoning accuracy.

## 🧭 Documentation Directory

Whether you are trying to understand the math behind the engine, configure it for your dataset, or deploy it to production, this guide has you covered:

| Section | Description |
| :--- | :--- |
| 🧠 **[Architecture Deep Dive](./ARCHITECTURE.md)** | Understand how LiteRAG works under the hood. Covers Anchor Discovery, Zero-LLM Traversal, and Reasoning-Chain Assembly. |
| ⚙️ **[Configuration Guide](./CONFIGURATION.md)** | A complete breakdown of `literag_config.yaml`. Learn how to tune exploration thresholds, hub penalties, and token budgets. |
| 💻 **[API Reference](./API_REFERENCE.md)** | Developer documentation for the Python API, core classes, data models, and asynchronous query methods. |
| 📊 **[Benchmarking & Evaluation](./BENCHMARKING.md)** | Learn how LiteRAG is evaluated against competitors using the DistComp dataset and our composite three-pillar accuracy metric. |
| 🚑 **[Troubleshooting](./TROUBLESHOOTING.md)** | Common errors, performance bottlenecks, LanceDB issues, and how to fix them. |

## 🚀 Quick Start Reminder

Make sure your `literag_config.yaml` points to a valid directory containing Microsoft GraphRAG parquet outputs:

```python
import asyncio
from literag import LiteRAG, LiteRAGConfig

async def main():
    config = LiteRAGConfig.from_yaml("literag_config.yaml")
    engine = LiteRAG(config)
    engine.initialize()
    
    result = await engine.aquery("What are the latency bottlenecks in IoT?")
    print(result.answer)

if __name__ == "__main__":
    asyncio.run(main())
```