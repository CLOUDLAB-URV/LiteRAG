# 📊 Benchmarking & Evaluation Methodology

Evaluating GraphRAG systems is notoriously difficult because standard QA datasets (like HotpotQA) often test superficial fact-hopping rather than deep, abstract conceptual synthesis. 

To rigorously evaluate LiteRAG against current SOTA paradigms (Microsoft GraphRAG, LightRAG, HiRAG), we built a custom evaluation framework.

## 1. The DistComp Dataset
To evaluate complex semantic drift and graph-based reasoning, we constructed the **DistComp corpus**. 
* **Provenance**: Derived from high-impact academic literature in Distributed Computing (CCGRID, EuroSys, ICDCS) from 2020-2022.
* **Why this data?**: It features high semantic complexity and abstract conceptual dependencies (e.g., *"Byzantine Fault Tolerance"* vs *"Federated Learning"*) that require deep multi-hop reasoning, not just simple keyword matching.
* **Scalability Splits**: Stratified into subsets of `{40, 80, 160, 640, 1280}` documents to test $O(N)$ vs $O(1)$ scaling behaviors.

## 2. Query Taxonomy
A set of 160 expert-curated queries was developed, categorizing graph traversal complexity:

1. **Literal Citation (LC)**: Single-hop fact retrieval (e.g., *"What is 'SparkLeBLAST'?"*).
2. **Local Reasoning (LR)**: Requires traversing explicit edges within a local subgraph (e.g., *"How does 'Kimchi' minimize costs?"*).
3. **Global Themes (G)**: High-level summarization across communities (e.g., *"What are the major trends in Serverless Computing?"*).
4. **Drift Queries (D)**: Complex exploration across completely disjoint community clusters (e.g., *"Compare the 'Interdependency Problem' with the 'Worst Parent Attack'."*).

## 3. The Three-Pillar Evaluation Metric ($S_{total}$)
We utilize a composite accuracy metric that blends generative quality with strict factual grounding to prevent LLM hallucination biases.

$$ S_{total} = 0.6 \cdot S_{LLM} + 0.25 \cdot S_{sem} + 0.15 \cdot S_{lex} $$

* 🧠 **LLM-as-a-Judge ($S_{LLM}$, 60%)**: Evaluates Correctness, Completeness, and Relevance.
* 📐 **Semantic Similarity ($S_{sem}$, 25%)**: Cosine similarity between the embedding vectors of the generated answer and the ground truth.
* 🔤 **Lexical Overlap ($S_{lex}$, 15%)**: ROUGE-L score acting as a strict penalty against hallucinations by requiring exact terminology overlap.

## 4. Benchmark Results Summary

When tested on the 1280-document DistComp corpus, LiteRAG solved the **Performance Paradox**:

| System | Mode | Accuracy ($S_{total}$) ↑ | Latency (s) ↓ | Tokens ↓ | Cost (€) ↓ |
|--------|------|-------------------------|---------------|----------|------------|
| **LiteRAG** | **Default** | **0.798** | **1.42s** | **2,291** | **€0.0096** |
| GraphRAG | Basic | 0.775 | 1.96s | 4,599 | €0.0190 |
| LightRAG | Hybrid | 0.763 | 3.32s | 20,567 | €0.0708 |
| HiRAG | Global | 0.784 | 2.95s | 20,136 | €0.0688 |
| GraphRAG | DRIFT | 0.657 | 142.04s | 6,441,013 | €25.2844 |

**Key Takeaways:**
1. **Context Pollution Causes Failure**: GraphRAG DRIFT retrieves 6.4 million tokens, saturating the context window with noise and dropping accuracy to `0.657`. LiteRAG filters this noise algorithmically, sending only `~2,200` highly relevant tokens, achieving `0.798` accuracy.
2. **O(1) Latency**: While other systems scale linearly (O(N)) with dataset size, LiteRAG's latency remains flat (~1.4s) regardless of graph size because it only explores mathematically relevant subgraphs.