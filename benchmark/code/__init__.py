"""
Unified Benchmark System for GraphRAG, LiteRAG, and LightRAG

A comprehensive benchmarking framework supporting multiple search engines
with a common interface for quality, performance, and cost metrics.

Features:
- LLM-as-Judge evaluation (correctness, completeness, relevance)
- Semantic similarity via embeddings
- ROUGE-L lexical overlap
- Performance metrics (latency, tokens, cost)
- Rich visualization and HTML reporting

Based on the methodology from GraphRAG Unified Benchmark.
"""

from .engine_interface import SearchEngineInterface, QueryResult, EngineType
from .metrics import MetricsCalculator
from .runner import UnifiedBenchmark, BenchmarkConfig, BenchmarkSummary
from .plots import create_plots, generate_html_report

__all__ = [
    # Core interfaces
    "SearchEngineInterface",
    "QueryResult",
    "EngineType",
    # Metrics
    "MetricsCalculator",
    # Runner
    "UnifiedBenchmark",
    "BenchmarkConfig",
    "BenchmarkSummary",
    # Visualization
    "create_plots",
    "generate_html_report",
]
