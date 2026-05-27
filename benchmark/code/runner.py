"""
Unified Benchmark Runner

Executes benchmarks across multiple engines with standardized metrics collection.
"""

import json
import yaml
import time
import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd
import numpy as np

from .engine_interface import SearchEngineInterface, QueryResult, EngineType
from .metrics import MetricsCalculator

LOGGER = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution"""
    output_dir: Path
    
    # GraphRAG settings (if using GraphRAG engines)
    graphrag_root_dir: Optional[Path] = None
    
    # LiteRAG settings (if using LiteRAG)
    literag_config_path: Optional[Path] = None  # Path to literag_config.yaml
    literag_data_dir: Optional[Path] = None     # Override data_dir from config
    
    # LightRAG settings (if using LightRAG engines)
    lightrag_config_path: Optional[Path] = None  # Path to lightrag_config.yaml
    
    # Google FileSearch settings
    filesearch_config_path: Optional[Path] = None  # Path to filesearch_config.yaml

    # HiRAG settings (if using HiRAG engines)
    hirag_config_path: Optional[Path] = None   # Path to hirag_config.yaml
    hirag_working_dir: Optional[Path] = None   # Override working_dir from config

    # Pricing
    prompt_price_per_m: float = 0.10
    completion_price_per_m: float = 0.40
    
    # Metrics settings
    metrics_config_path: Optional[Path] = None  # Path to settings.yaml for LLM metrics
    
    # Execution settings
    concurrency: int = 1
    cache_eval: bool = True

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if self.graphrag_root_dir:
            self.graphrag_root_dir = Path(self.graphrag_root_dir)
        if self.literag_config_path:
            self.literag_config_path = Path(self.literag_config_path)
        if self.literag_data_dir:
            self.literag_data_dir = Path(self.literag_data_dir)
        if self.lightrag_config_path:
            self.lightrag_config_path = Path(self.lightrag_config_path)
        if self.filesearch_config_path:
            self.filesearch_config_path = Path(self.filesearch_config_path)
        if self.hirag_config_path:
            self.hirag_config_path = Path(self.hirag_config_path)
        if self.hirag_working_dir:
            self.hirag_working_dir = Path(self.hirag_working_dir)
        if self.metrics_config_path:
            self.metrics_config_path = Path(self.metrics_config_path)


@dataclass
class BenchmarkSummary:
    """Summary statistics for a benchmark run"""
    total_queries: int
    successful_queries: int
    failed_queries: int
    total_time_seconds: float
    total_cost_eur: float
    total_tokens: int
    
    # Per-engine stats
    engine_stats: Dict[str, Dict[str, Any]]
    
    # Quality metrics (averages)
    avg_answer_accuracy: Optional[float] = None
    avg_llm_judged_correctness: Optional[float] = None
    avg_llm_judged_completeness: Optional[float] = None
    avg_semantic_similarity: Optional[float] = None


class UnifiedBenchmark:
    """
    Unified benchmark runner supporting multiple search engines.
    
    Usage:
        config = BenchmarkConfig(
            output_dir="./results",
            graphrag_root_dir="./graphrag_workspace",
            literag_data_dir="./data",
            literag_data_dir="./data"
        )
        
        benchmark = UnifiedBenchmark(config)
        
        # Add engines to test
        benchmark.add_graphrag_engines(["basic", "local"])
        benchmark.add_literag_engine()
        
        # Run benchmark
        await benchmark.run(prompts)
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.engines: Dict[str, SearchEngineInterface] = {}
        self.results: List[QueryResult] = []
        self.initialization_errors: Dict[str, str] = {}
        api_key = None
        
        if not config.metrics_config_path and config.literag_config_path:
            # Try to get API key from LiteRAG config if available
            try:
                with open(config.literag_config_path, 'r') as f:
                    literag_data = yaml.safe_load(f)
                    api_key = literag_data.get('api_key')
            except Exception as e:
                print(f"⚠️  Could not read API key from LiteRAG config: {e}")
                if LOGGER.isEnabledFor(logging.INFO):
                    LOGGER.exception("Failed reading LiteRAG config for metrics bootstrap key")

        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(
            config_path=config.metrics_config_path,
            api_key=api_key,
            cache_dir=self.config.output_dir / ".eval_cache" if config.cache_eval else None
        )
        
        # Create output directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir = self.config.output_dir / "responses"
        self.responses_dir.mkdir(exist_ok=True)
        
        self._print_header()
    
    def _print_header(self):
        """Print benchmark header"""
        print("=" * 70)
        print("Unified Search Engine Benchmark")
        print("=" * 70)
        print(f"Output Directory: {self.config.output_dir}")
        print(f"LLM Metrics:      {'Enabled' if self.metrics_calculator.llm_available else 'Disabled'}")
        print("=" * 70)
    
    def add_graphrag_engines(self, methods: List[str]) -> None:
        """
        Add GraphRAG engines to benchmark.
        
        Args:
            methods: List of methods to test: ["basic", "local", "global", "drift"]
        """
        if not self.config.graphrag_root_dir:
            raise ValueError("graphrag_root_dir must be set to use GraphRAG engines")
        
        from .engines_graphrag import (
            GraphRAGBasicEngine,
            GraphRAGLocalEngine,
            GraphRAGGlobalEngine,
            GraphRAGDriftEngine,
        )
        
        engine_map = {
            "basic": GraphRAGBasicEngine,
            "local": GraphRAGLocalEngine,
            "global": GraphRAGGlobalEngine,
            "drift": GraphRAGDriftEngine,
        }
        
        for method in methods:
            method = method.lower()
            if method not in engine_map:
                print(f"⚠️  Unknown GraphRAG method: {method}")
                continue
            
            engine_class = engine_map[method]
            engine = engine_class(
                root_dir=self.config.graphrag_root_dir,
                prompt_price_per_m=self.config.prompt_price_per_m,
                completion_price_per_m=self.config.completion_price_per_m,
            )
            
            self.engines[engine.engine_type.value] = engine
            print(f"  ✓ Added engine: {engine.name}")
    
    def add_literag_engine(self, **kwargs) -> None:
        """
        Add LiteRAG engine to benchmark.
        
        Args:
            **kwargs: Additional LiteRAG configuration overrides
        """
        if not self.config.literag_config_path or not self.config.literag_config_path.exists():
            raise ValueError("literag_config_path must be set and exist to use LiteRAG")
        
        from .engines_literag import LiteRAGEngine
        
        # Pass data_dir override if specified
        if self.config.literag_data_dir:
            kwargs['data_dir'] = str(self.config.literag_data_dir)
        
        engine = LiteRAGEngine(
            config_path=self.config.literag_config_path,
            prompt_price_per_m=self.config.prompt_price_per_m,
            completion_price_per_m=self.config.completion_price_per_m,
            **kwargs
        )
        
        self.engines[engine.engine_type.value] = engine
        print(f"  ✓ Added engine: {engine.name}")
    
    def add_lightrag_engines(self, modes: List[str]) -> None:
        """
        Add LightRAG engines to benchmark.
        
        Args:
            modes: List of query modes to test: ["local", "global", "hybrid", "mix", "naive"]
        """
        if not self.config.lightrag_config_path or not self.config.lightrag_config_path.exists():
            raise ValueError("lightrag_config_path must be set and exist to use LightRAG engines")
        
        from .engines_lightrag import (
            LightRAGLocalEngine,
            LightRAGGlobalEngine,
            LightRAGHybridEngine,
            LightRAGMixEngine,
            LightRAGNaiveEngine,
        )
        
        engine_map = {
            "local": LightRAGLocalEngine,
            "global": LightRAGGlobalEngine,
            "hybrid": LightRAGHybridEngine,
            "mix": LightRAGMixEngine,
            "naive": LightRAGNaiveEngine,
        }
        
        for mode in modes:
            mode = mode.lower()
            if mode not in engine_map:
                print(f"⚠️  Unknown LightRAG mode: {mode}")
                continue
            
            engine_class = engine_map[mode]
            engine = engine_class(
                config_path=self.config.lightrag_config_path,
                prompt_price_per_m=self.config.prompt_price_per_m,
                completion_price_per_m=self.config.completion_price_per_m,
            )
            
            self.engines[engine.engine_type.value] = engine
            print(f"  ✓ Added engine: {engine.name}")
    
    def add_filesearch_engine(self, **kwargs) -> None:
        """
        Add Google FileSearch engine to benchmark.
        
        Args:
            **kwargs: Additional configuration overrides
        """
        if not self.config.filesearch_config_path or not self.config.filesearch_config_path.exists():
            raise ValueError("filesearch_config_path must be set and exist to use Google FileSearch")
        
        from .engines_filesearch import GoogleFileSearchEngine
        
        engine = GoogleFileSearchEngine(
            config_path=self.config.filesearch_config_path,
            prompt_price_per_m=self.config.prompt_price_per_m,
            completion_price_per_m=self.config.completion_price_per_m,
            **kwargs
        )
        
        self.engines[engine.engine_type.value] = engine
        print(f"  ✓ Added engine: {engine.name}")

    def add_hirag_engines(self, modes: List[str]) -> None:
        """
        Add HiRAG engines to benchmark.

        Args:
            modes: List of query modes to test:
                   ["hi", "hi_local", "hi_global", "hi_bridge", "hi_nobridge", "naive"]
        """
        if not self.config.hirag_config_path or not self.config.hirag_config_path.exists():
            raise ValueError("hirag_config_path must be set and exist to use HiRAG engines")

        from .engines_hirag import (
            HiRAGHiEngine,
            HiRAGLocalEngine,
            HiRAGGlobalEngine,
            HiRAGBridgeEngine,
            HiRAGNoBridgeEngine,
            HiRAGNaiveEngine,
        )

        engine_map = {
            "hi":       HiRAGHiEngine,
            "local":    HiRAGLocalEngine,
            "global":   HiRAGGlobalEngine,
            "bridge":   HiRAGBridgeEngine,
            "nobridge": HiRAGNoBridgeEngine,
            "naive":    HiRAGNaiveEngine,
        }

        for mode in modes:
            mode = mode.lower()
            if mode not in engine_map:
                print(f"⚠️  Unknown HiRAG mode: {mode}")
                continue

            engine_class = engine_map[mode]
            extra_kwargs = {}
            if self.config.hirag_working_dir:
                extra_kwargs['working_dir'] = str(self.config.hirag_working_dir)
            engine = engine_class(
                config_path=self.config.hirag_config_path,
                prompt_price_per_m=self.config.prompt_price_per_m,
                completion_price_per_m=self.config.completion_price_per_m,
                **extra_kwargs,
            )

            self.engines[engine.engine_type.value] = engine
            print(f"  ✓ Added engine: {engine.name}")

    def add_custom_engine(self, engine: SearchEngineInterface) -> None:
        """Add a custom engine implementing SearchEngineInterface"""
        self.engines[engine.engine_type.value] = engine
        print(f"  ✓ Added engine: {engine.name}")
    
    async def initialize_engines(self) -> None:
        """Initialize all registered engines"""
        print("\nInitializing engines...")

        initialized: Dict[str, SearchEngineInterface] = {}
        self.initialization_errors = {}

        for name, engine in self.engines.items():
            try:
                print(f"  Initializing {engine.name}...", end=" ", flush=True)
                await engine.initialize()
                print("✓")
                initialized[name] = engine
            except Exception as e:
                print(f"✗ {e}")
                self.initialization_errors[name] = str(e)
                if LOGGER.isEnabledFor(logging.INFO):
                    LOGGER.exception("Engine initialization failed: %s", name)

        self.engines = initialized

        if self.initialization_errors:
            failed_names = ", ".join(sorted(self.initialization_errors.keys()))
            print(f"⚠️  Skipping engines that failed to initialize: {failed_names}")

        if not self.engines:
            details = "; ".join(
                f"{name}: {err}" for name, err in self.initialization_errors.items()
            ) or "no initialization details available"
            raise RuntimeError(f"No engines initialized successfully ({details})")
    
    async def run(
        self,
        prompts: Dict[str, Dict[str, str]],
        engines: Optional[List[str]] = None
    ) -> None:
        """
        Run benchmark across all prompts and engines.
        
        Args:
            prompts: Dict of {prompt_name: {"question": ..., "ground_truth": ...}}
            engines: Optional list of engine names to test (None = all)
        """
        if not self.engines:
            raise ValueError("No engines registered. Add at least one engine first.")
        
        await self.initialize_engines()

        engines_to_test = self.engines
        if engines:
            engines_to_test = {k: v for k, v in self.engines.items() if k in engines}
        if not engines_to_test:
            raise ValueError("No initialized engines available for execution")
        
        total_queries = len(prompts) * len(engines_to_test)
        
        print(f"\nRunning benchmark: {len(engines_to_test)} engines × {len(prompts)} prompts = {total_queries} queries")
        print(f"Concurrency: {self.config.concurrency}")
        
        start_time = time.time()
        semaphore = asyncio.Semaphore(self.config.concurrency)
        
        # Prepare tasks
        tasks = []
        current = 0
        for prompt_name, prompt_data in prompts.items():
            question = prompt_data.get("question")
            ground_truth = prompt_data.get("ground_truth")
            if not question:
                continue
                
            for engine_name, engine in engines_to_test.items():
                current += 1
                tasks.append(self._execute_single_query(
                    engine, prompt_name, question, ground_truth, semaphore, current, total_queries
                ))
        
        try:
            # Run all tasks
            results = await asyncio.gather(*tasks)
            self.results.extend(results)
        finally:
            # Ensure all engines are closed even if interrupted
            print("\nClosing engines...")
            for engine in engines_to_test.values():
                try:
                    await engine.close()
                except Exception as e:
                    print(f"  ⚠️ Error closing {engine.name}: {e}")
        
        total_time = time.time() - start_time
        
        # Save results
        self._save_results()
        
        # Generate summary
        summary = self._generate_summary(total_time)
        self._save_summary(summary)
        
        # Generate visualizations
        self._generate_visualizations(summary)
        
        # Print summary
        self._print_summary(summary)

    async def _execute_single_query(
        self,
        engine: SearchEngineInterface,
        prompt_name: str,
        question: str,
        ground_truth: Optional[str],
        semaphore: asyncio.Semaphore,
        idx: int,
        total: int
    ) -> QueryResult:
        """Execute a single query with concurrency control"""
        async with semaphore:
            # Execute query
            result = await engine.search(question)
            result.prompt_name = prompt_name
            result.ground_truth = ground_truth
            
            # Calculate quality metrics
            if ground_truth and result.answer and not result.error:
                metrics = await self.metrics_calculator.calculate_all_metrics(
                    answer=result.answer,
                    ground_truth=ground_truth,
                    question=question,
                    context=result.context_text or ""
                )
                
                # Update result with metrics
                for key, value in metrics.items():
                    if hasattr(result, key):
                        setattr(result, key, value)
            
            # Print status
            if result.error:
                print(f"   [{idx}/{total}] {engine.name}: ✗ {result.error[:40]}")
            else:
                acc = f", Acc: {result.answer_accuracy:.2f}" if result.answer_accuracy else ""
                print(f"   [{idx}/{total}] {engine.name}: ✓ {result.latency_seconds:.2f}s, {result.total_tokens} tok, ${result.cost_eur:.4f}{acc}")
            
            # Save individual response
            self._save_response(result)
            return result
    
    def _save_response(self, result: QueryResult) -> None:
        """Save individual response to file"""
        safe_engine = self._safe_filename_part(result.engine)
        safe_prompt = self._safe_filename_part(result.prompt_name)
        filename = f"{safe_engine}__{safe_prompt}.txt"
        filepath = self.responses_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Engine: {result.engine}\n")
            f.write(f"Prompt: {result.prompt_name}\n")
            f.write(f"Timestamp: {result.timestamp}\n")
            
            f.write(f"\n{'='*70}\nQUESTION\n{'='*70}\n{result.prompt_text}\n\n")
            
            if result.ground_truth:
                f.write(f"{'='*70}\nGROUND TRUTH\n{'='*70}\n{result.ground_truth}\n\n")
            
            f.write(f"{'='*70}\nRESPONSE\n{'='*70}\n{result.answer}\n\n")
            
            f.write(f"{'='*70}\nPERFORMANCE\n{'='*70}\n")
            f.write(f"Latency:    {result.latency_seconds}s\n")
            f.write(f"LLM Calls:  {result.llm_calls}\n")
            f.write(f"Tokens:     {result.total_tokens} (prompt: {result.prompt_tokens}, completion: {result.completion_tokens})\n")
            f.write(f"Cost:       ${result.cost_eur}\n")
            
            if result.answer_accuracy is not None:
                f.write(f"\n{'='*70}\nQUALITY METRICS\n{'='*70}\n")
                f.write(f"Overall Accuracy:   {result.answer_accuracy:.4f}\n")
                f.write(f"LLM Correctness:    {result.llm_judged_correctness:.4f}\n")
                f.write(f"LLM Completeness:   {result.llm_judged_completeness:.4f}\n")
                f.write(f"LLM Relevance:      {result.llm_judged_relevance:.4f}\n")
                f.write(f"Semantic Sim:       {result.semantic_similarity:.4f}\n")
                f.write(f"ROUGE-L:            {result.rougeL_f1:.4f}\n")
                
                if result.llm_judged_reasoning:
                    f.write(f"\nReasoning:\n{result.llm_judged_reasoning}\n")
            
            if result.extra_info:
                f.write(f"\n{'='*70}\nENGINE-SPECIFIC INFO\n{'='*70}\n")
                for k, v in result.extra_info.items():
                    f.write(f"{k}: {v}\n")
            
            if result.error:
                f.write(f"\n{'='*70}\nERROR\n{'='*70}\n{result.error}\n")

    @staticmethod
    def _safe_filename_part(value: Optional[str]) -> str:
        """Sanitize user/config-derived filename parts to prevent path traversal."""
        raw = (value or "unknown").strip()
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
        cleaned = cleaned.strip("._-")
        if not cleaned:
            return "unknown"
        return cleaned[:120]
    
    @staticmethod
    def load_query_result_from_file(file_path: Path) -> QueryResult:
        """
        Parse a saved response file back into a QueryResult object.
        
        Args:
            file_path: Path to the .txt response file
            
        Returns:
            Reconstructed QueryResult object
        """
        import re
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Helper to extract values
        def extract_value(pattern, text, default=None, cast_type=str):
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                try:
                    return cast_type(match.group(1).strip())
                except (ValueError, IndexError):
                    return default
            return default
            
        # Basic Info
        engine = extract_value(r"^Engine:\s*(.+)$", content)
        if not engine:
            # Fallback: try to deduce from filename if header missing
            # Filename format: {engine}__{prompt}.txt
            filename = file_path.name
            if "__" in filename:
                engine = filename.split("__")[0]
            else:
                engine = "unknown"
                
        prompt_name = extract_value(r"^Prompt:\s*(.+)$", content)
        timestamp = extract_value(r"^Timestamp:\s*(.+)$", content)
        
        # Sections
        # We split by headers to get content
        sections = re.split(r"={70}\n(.+)\n={70}\n", content)
        
        section_map = {}
        for i in range(1, len(sections), 2):
            header = sections[i].strip()
            body = sections[i+1].strip()
            section_map[header] = body
            
        question = section_map.get("QUESTION", "")
        ground_truth = section_map.get("GROUND TRUTH")
        response = section_map.get("RESPONSE", "")
        error = section_map.get("ERROR")
        
        # Performance
        perf_section = section_map.get("PERFORMANCE", "")
        latency = extract_value(r"Latency:\s*([\d\.]+)s", perf_section, 0.0, float)
        llm_calls = extract_value(r"LLM Calls:\s*(\d+)", perf_section, 0, int)
        total_tokens = extract_value(r"Tokens:\s*(\d+)", perf_section, 0, int)
        
        # Token breakdown
        prompt_tokens = 0
        completion_tokens = 0
        token_breakdown = re.search(r"prompt:\s*(\d+).*completion:\s*(\d+)", perf_section)
        if token_breakdown:
            prompt_tokens = int(token_breakdown.group(1))
            completion_tokens = int(token_breakdown.group(2))
            
        # Cost parsing
        cost_str = extract_value(r"Cost:\s*[$$]?([\d\.]+)", perf_section, "0.0")
        cost = float(cost_str)
        
        # Quality Metrics
        quality_section = section_map.get("QUALITY METRICS", "")
        
        def extract_metric(name, section):
            # Matches "Name: value"
            val = extract_value(fr"{name}:\s*([\d\.]+)", section, None, float)
            return val

        answer_accuracy = extract_metric("Overall Accuracy", quality_section)
        llm_correctness = extract_metric("LLM Correctness", quality_section)
        llm_completeness = extract_metric("LLM Completeness", quality_section)
        llm_relevance = extract_metric("LLM Relevance", quality_section)
        semantic_sim = extract_metric("Semantic Sim", quality_section)
        rouge_l = extract_metric("ROUGE-L", quality_section)
        
        # Reasoning usually follows the metrics
        reasoning = None
        if "Reasoning:" in quality_section:
            _, reasoning_part = quality_section.split("Reasoning:", 1)
            reasoning = reasoning_part.strip()
            
        # Extra Info
        extra_info_section = section_map.get("ENGINE-SPECIFIC INFO", "")
        extra_info = {}
        if extra_info_section:
            for line in extra_info_section.split('\n'):
                if ':' in line:
                    k, v = line.split(':', 1)
                    k = k.strip()
                    v = v.strip()
                    # Try to convert numbers
                    if v.replace('.', '', 1).isdigit():
                        if '.' in v:
                            v = float(v)
                        else:
                            v = int(v)
                    elif v == 'None':
                        v = None
                    extra_info[k] = v

        return QueryResult(
            engine=engine,
            prompt_name=prompt_name or file_path.stem,
            prompt_text=question,
            answer=response,
            ground_truth=ground_truth,
            latency_seconds=latency,
            llm_calls=llm_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_eur=cost,
            error=error,
            timestamp=timestamp or datetime.now().isoformat(),
            answer_accuracy=answer_accuracy,
            llm_judged_correctness=llm_correctness,
            llm_judged_completeness=llm_completeness,
            llm_judged_relevance=llm_relevance,
            semantic_similarity=semantic_sim,
            rougeL_f1=rouge_l,
            llm_judged_reasoning=reasoning,
            extra_info=extra_info
        )
    
    def _save_results(self) -> None:
        """Save all results to CSV and JSON"""
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        # Save CSV
        csv_path = self.config.output_dir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ CSV saved: {csv_path}")
        
        # Save JSON
        json_path = self.config.output_dir / "benchmark_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2, ensure_ascii=False)
        print(f"✓ JSON saved: {json_path}")
    
    def _generate_summary(self, total_time: float) -> BenchmarkSummary:
        """Generate benchmark summary statistics"""
        successful = [r for r in self.results if r.is_success]
        failed = [r for r in self.results if not r.is_success]
        
        # Per-engine stats
        engine_stats = {}
        for engine_name in self.engines.keys():
            engine_results = [r for r in self.results if r.engine == engine_name]
            engine_successful = [r for r in engine_results if r.is_success]
            
            if engine_successful:
                engine_stats[engine_name] = {
                    "queries": len(engine_results),
                    "successful": len(engine_successful),
                    "avg_latency": np.mean([r.latency_seconds for r in engine_successful]),
                    "avg_tokens": np.mean([r.total_tokens for r in engine_successful]),
                    "total_cost": sum(r.cost_eur for r in engine_results),
                    "avg_llm_calls": np.mean([r.llm_calls for r in engine_successful]),
                }
                
                # Quality metrics
                accuracies = [r.answer_accuracy for r in engine_successful if r.answer_accuracy is not None]
                if accuracies:
                    engine_stats[engine_name]["avg_accuracy"] = np.mean(accuracies)
        
        # Global averages
        def safe_mean(values):
            valid = [v for v in values if v is not None]
            return np.mean(valid) if valid else None
        
        return BenchmarkSummary(
            total_queries=len(self.results),
            successful_queries=len(successful),
            failed_queries=len(failed),
            total_time_seconds=total_time,
            total_cost_eur=sum(r.cost_eur for r in self.results),
            total_tokens=sum(r.total_tokens for r in self.results),
            engine_stats=engine_stats,
            avg_answer_accuracy=safe_mean([r.answer_accuracy for r in successful]),
            avg_llm_judged_correctness=safe_mean([r.llm_judged_correctness for r in successful]),
            avg_llm_judged_completeness=safe_mean([r.llm_judged_completeness for r in successful]),
            avg_semantic_similarity=safe_mean([r.semantic_similarity for r in successful]),
        )
    
    def _generate_visualizations(self, summary: BenchmarkSummary) -> None:
        """Generate plots and HTML report"""
        try:
            from .plots import create_plots, generate_html_report
            
            # Create DataFrame from results
            df = pd.DataFrame([r.to_dict() for r in self.results])
            
            # Generate plots
            print("\nGenerating visualizations...")
            plots_dir = self.config.output_dir / "plots"
            plots = create_plots(df, plots_dir)
            
            if plots:
                print(f"✓ Generated {len(plots)} plots in {plots_dir}")
                
                # Generate HTML report
                html_path = generate_html_report(
                    df,
                    asdict(summary),
                    plots,
                    self.config.output_dir
                )
                print(f"✓ HTML report: {html_path}")
            else:
                print("⚠️  No plots generated (matplotlib may not be installed)")
                
        except ImportError as e:
            print(f"⚠️  Could not generate visualizations: {e}")
        except Exception as e:
            print(f"⚠️  Error generating visualizations: {e}")
            if LOGGER.isEnabledFor(logging.INFO):
                LOGGER.exception("Visualization generation failed")
    
    def _save_summary(self, summary: BenchmarkSummary) -> None:
        """Save summary to JSON"""
        summary_path = self.config.output_dir / "benchmark_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(summary), f, indent=2)
        print(f"✓ Summary saved: {summary_path}")
    
    def _print_summary(self, summary: BenchmarkSummary) -> None:
        """Print formatted summary"""
        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)
        
        print(f"\nTotal Queries:  {summary.total_queries}")
        print(f"Successful:     {summary.successful_queries}")
        print(f"Failed:         {summary.failed_queries}")
        print(f"Total Time:     {summary.total_time_seconds:.2f}s")
        print(f"Total Cost:     ${summary.total_cost_eur:.4f}")
        print(f"Total Tokens:   {summary.total_tokens:,}")
        
        # Per-engine comparison
        print("\n" + "-" * 70)
        print("ENGINE COMPARISON")
        print("-" * 70)
        
        header = f"{'Engine':<20} {'Latency':>10} {'Tokens':>10} {'Cost':>10} {'Accuracy':>10}"
        print(header)
        print("-" * 70)
        
        for engine_name, stats in summary.engine_stats.items():
            latency = f"{stats['avg_latency']:.2f}s"
            tokens = f"{stats['avg_tokens']:.0f}"
            cost = f"${stats['total_cost']:.4f}"
            accuracy = f"{stats.get('avg_accuracy', 0):.2f}" if stats.get('avg_accuracy') else "N/A"
            
            print(f"{engine_name:<20} {latency:>10} {tokens:>10} {cost:>10} {accuracy:>10}")
        
        # Quality summary
        if summary.avg_answer_accuracy:
            print("\n" + "-" * 70)
            print("QUALITY METRICS (Global Averages)")
            print("-" * 70)
            print(f"Overall Accuracy:   {summary.avg_answer_accuracy:.4f}")
            print(f"LLM Correctness:    {summary.avg_llm_judged_correctness:.4f}")
            print(f"LLM Completeness:   {summary.avg_llm_judged_completeness:.4f}")
            print(f"Semantic Similarity:{summary.avg_semantic_similarity:.4f}")
        
        print("\n" + "=" * 70)
