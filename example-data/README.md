# 📂 LiteRAG Example Data & Benchmarks (DistComp Dataset)

This directory contains a subset of the **DistComp Dataset**, curated specifically to benchmark GraphRAG scalability, multi-hop reasoning, and semantic drift. 

To make testing LiteRAG as seamless as possible, **we have pre-indexed the documents using Microsoft GraphRAG**. You do not need to spend time or API credits running an indexer; the Parquet files and LanceDB vector stores are ready to be queried immediately.

## Directory Layout

* **`*docs.json`**: The benchmark queries and ground-truth answers (e.g., `40docs.json`). The queries are divided into Literal Citation (`lc`), Local Reasoning (`lr`), Global Themes (`g`), and Drift (`d`).
* **`doc*/input/`**: The raw academic paper CSVs.
* **`doc*/output/`**: The pre-computed Knowledge Graph and Vector Database. **Point your LiteRAG config here.**
* **`doc*/prompts/` & `settings.yaml`**: The exact configurations used during the GraphRAG indexing phase to ensure total reproducibility.

## Quick Start

To run a query against the 40-document subset, ensure your `literag_config.yaml` points to `example-data/doc40/output`:

```yaml
# literag_config.yaml
data_dir: "example-data/doc40/output"
```

Then run the provided example script from the root of the repository:

```bash
python example.py "What technologies enable edge computing for IoT?"
```

## Running the Full Benchmark

To run the unified evaluation suite (comparing LiteRAG to MS GraphRAG, LightRAG, etc.) using the provided benchmark files, navigate to the `benchmark` directory:

```bash
python run_benchmark.py \
    --prompts ../example-data/40docs.json \
    --literag-config ../literag_config.yaml \
    --graphrag-dir ../example-data/doc40/ \
    --engines literag graphrag_local \
    --concurrency 4
```

## Full Dataset Availability

Due to GitHub file size limits, the larger splits (up to 1280 documents) are hosted on Hugging Face. 

👉 **[Download the full DistComp dataset on Hugging Face here](https://huggingface.co/datasets/macarronesc/DistComp)**

## License
The DistComp dataset and these pre-computed indexes are provided under the **Apache 2.0 License**.