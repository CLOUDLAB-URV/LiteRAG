#!/usr/bin/env python3
"""
Unified Benchmark CLI

Run benchmarks comparing GraphRAG, LiteRAG, LightRAG, HiRAG and Google FileSearch.

Examples:
    # Compare all configured engines
    python run_benchmark.py --prompts prompts.json --graphrag-dir ./workspace --literag-config ./config/LiteRAG.yaml --lightrag-config ./config/lightrag.yaml --hirag-config ./config/hirag.yaml --filesearch-config ./config/filesearch.yaml

    # Only GraphRAG
    python run_benchmark.py --prompts prompts.json --graphrag-dir ./workspace --engines graphrag_local graphrag_global
    
    # Only LiteRAG
    python run_benchmark.py --prompts prompts.json --literag-config ./config/LiteRAG.yaml --engines literag
    
    # Quick comparison
    python run_benchmark.py --prompts prompts.json --graphrag-dir ./workspace --literag-config ./config/LiteRAG.yaml --engines graphrag_local literag
"""

import argparse
import asyncio
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

# Filter out annoying Pydantic serializer warnings from LangChain's structured output
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

async def main():
    parser = argparse.ArgumentParser(
        description='Unified Search Engine Benchmark (GraphRAG + LiteRAG + LightRAG + HiRAG + FileSearch)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required
    parser.add_argument('--prompts', type=str, required=False,
                        help='Path to JSON file with prompts ({"name": {"question": ..., "ground_truth": ...}})')
    
    # GraphRAG settings
    parser.add_argument('--graphrag-dir', type=str, default=None,
                        help='GraphRAG workspace directory (with output/ folder)')
    
    # LiteRAG settings
    parser.add_argument('--literag-config', type=str, default=None,
                        help='Path to LiteRAG YAML configuration file')
    parser.add_argument('--literag-dir', type=str, default=None,
                        help='LiteRAG data directory (overrides data_dir in config)')
    
    # LightRAG settings
    parser.add_argument('--lightrag-config', type=str, default=None,
                        help='Path to LightRAG YAML configuration file')

    # HiRAG settings
    parser.add_argument('--hirag-config', type=str, default=None,
                        help='Path to HiRAG YAML configuration file')
    parser.add_argument('--hirag-dir', type=str, default=None,
                        help='HiRAG working directory with indexed files (overrides working_dir in config)')

    # Google FileSearch settings
    parser.add_argument('--filesearch-config', type=str, default=None,
                        help='Path to Google FileSearch YAML configuration file')
    
    # Engine selection
    parser.add_argument('--engines', nargs='+', 
                        choices=['graphrag_basic', 'graphrag_local', 'graphrag_global', 'graphrag_drift', 
                                 'literag', 
                                 'lightrag_local', 'lightrag_global', 'lightrag_hybrid', 'lightrag_mix', 'lightrag_naive',
                                 'google_filesearch',
                                 'hirag_hi', 'hirag_local', 'hirag_global', 'hirag_bridge', 'hirag_nobridge', 'hirag_naive',
                                 'all'],
                        default=['all'],
                        help='Engines to benchmark')
    
    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    
    # Metrics configuration
    parser.add_argument('--metrics-config', type=str, default=None,
                        help='Path to settings.yaml for LLM metrics (defaults to graphrag-dir/settings.yaml)')
    
    # Pricing
    parser.add_argument('--prompt-price', type=float, default=0.10,
                        help='Price per million prompt tokens (EUR)')
    parser.add_argument('--completion-price', type=float, default=0.40,
                        help='Price per million completion tokens (EUR)')
    
    # Plot only option
    parser.add_argument('--plot-only', type=str, default=None,
                        help='Generate plots from existing benchmark_results.json file')
    
    # Execution settings
    parser.add_argument('--concurrency', type=int, default=1,
                        help='Number of concurrent queries to run')
    parser.add_argument('--cache-eval', action='store_true',
                        help='Enable LLM-as-judge caching to save costs')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()

    # Configure logging
    import logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Handle plot-only mode
    if args.plot_only:
        path = Path(args.plot_only)
        if not path.exists():
            print(f"Error: Path not found: {path}")
            return

        # Import necessary classes
        from code.runner import UnifiedBenchmark, BenchmarkConfig, QueryResult
        
        # Determine if input is file or directory
        results = []
        output_dir = path if path.is_dir() else path.parent
        
        if path.is_file() and path.suffix == '.json':
            print(f"Generating plots from JSON: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                if 'is_success' in item:
                    del item['is_success']
                results.append(QueryResult(**item))
                
        elif path.is_dir():
            print(f"Generating plots from response directory: {path}")
            
            # Check for "responses" subdirectory
            responses_dir = path / "responses"
            if responses_dir.exists() and responses_dir.is_dir():
                search_dir = responses_dir
                output_dir = path # Use the parent of responses as output
            else:
                search_dir = path
                
            response_files = list(search_dir.glob("*.txt"))
            if not response_files:
                print(f"Error: No .txt response files found in {search_dir}")
                return
                
            print(f"Found {len(response_files)} response files. Parsing...")
            
            # Use the new static method to parse files
            for file_path in response_files:
                try:
                    result = UnifiedBenchmark.load_query_result_from_file(file_path)
                    results.append(result)
                except Exception as e:
                    print(f"Warning: Failed to parse {file_path.name}: {e}")
            
        else:
            print(f"Error: Invalid input path {path}. Must be a JSON file or directory.")
            return

        if not results:
            print("No results loaded.")
            return

        # Create config
        config = BenchmarkConfig(
            output_dir=output_dir,
            # Other fields irrelevant for plotting
        )
        
        # Create benchmark instance
        benchmark = UnifiedBenchmark(config)
        benchmark.results = results
        
        # Populate engines keys for summary generation
        engine_names = set(r.engine for r in results)
        benchmark.engines = {name: None for name in engine_names}
        
        # Save reconstructed JSON (important if we came from directory)
        if path.is_dir():
            benchmark._save_results()
        
        # Generate summary
        total_time = sum(r.latency_seconds for r in results)
        summary = benchmark._generate_summary(total_time)
        
        # Save summary
        benchmark._save_summary(summary)
        
        # Generate visualizations
        benchmark._generate_visualizations(summary)
        
        # Print summary
        benchmark._print_summary(summary)
        
        return
    
    # If not plot-only, prompts is required
    if not args.prompts:
        parser.error("the following arguments are required: --prompts")
    
    # Validate inputs
    if 'all' in args.engines or any(e.startswith('graphrag') for e in args.engines):
        if not args.graphrag_dir:
            print("Error: --graphrag-dir required for GraphRAG engines")
            return
    
    if 'all' in args.engines or 'literag' in args.engines:
        if not args.literag_config or not Path(args.literag_config).exists():
            print("Error: --literag_config is required for LiteRAG engine")
            return
    
    if 'all' in args.engines or any(e.startswith('lightrag') for e in args.engines):
        if not args.lightrag_config or not Path(args.lightrag_config).exists():
            print("Error: --lightrag_config is required for LightRAG engines")
            return
    
    if 'all' in args.engines or 'google_filesearch' in args.engines:
        if not args.filesearch_config or not Path(args.filesearch_config).exists():
            print("Error: --filesearch-config is required for Google FileSearch engine")
            return

    if 'all' in args.engines or any(e.startswith('hirag') for e in args.engines):
        if not args.hirag_config or not Path(args.hirag_config).exists():
            print("Error: --hirag-config is required for HiRAG engines")
            return
    
    # Load prompts
    with open(args.prompts, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./benchmark_results_{timestamp}")
    
    # Set metrics config
    metrics_config = None
    if args.metrics_config:
        metrics_config = Path(args.metrics_config)
    elif args.graphrag_dir:
        settings_path = Path(args.graphrag_dir) / "settings.yaml"
        if settings_path.exists():
            metrics_config = settings_path
    
    # Import benchmark modules
    from code.runner import UnifiedBenchmark, BenchmarkConfig
    
    # Create config
    config = BenchmarkConfig(
        output_dir=output_dir,
        graphrag_root_dir=Path(args.graphrag_dir) if args.graphrag_dir else None,
        literag_config_path=Path(args.literag_config) if args.literag_config else None,
        literag_data_dir=Path(args.literag_dir) if args.literag_dir else None,
        lightrag_config_path=Path(args.lightrag_config) if args.lightrag_config else None,
        filesearch_config_path=Path(args.filesearch_config) if args.filesearch_config else None,
        hirag_config_path=Path(args.hirag_config) if args.hirag_config else None,
        hirag_working_dir=Path(args.hirag_dir) if args.hirag_dir else None,
        prompt_price_per_m=args.prompt_price,
        completion_price_per_m=args.completion_price,
        metrics_config_path=metrics_config,
        concurrency=args.concurrency,
        cache_eval=args.cache_eval,
    )
    
    # Create benchmark
    benchmark = UnifiedBenchmark(config)
    
    # Determine which engines to add
    engines_to_add = args.engines
    if 'all' in engines_to_add:
        engines_to_add = []
        if args.graphrag_dir:
            engines_to_add.extend(['graphrag_basic', 'graphrag_local', 'graphrag_global', 'graphrag_drift'])
        if args.literag_config:
            engines_to_add.append('literag')
        if args.lightrag_config:
            engines_to_add.extend(['lightrag_local', 'lightrag_global', 'lightrag_hybrid', 'lightrag_mix', 'lightrag_naive'])
        if args.filesearch_config:
            engines_to_add.append('google_filesearch')
        if args.hirag_config:
            engines_to_add.extend(['hirag_hi', 'hirag_local', 'hirag_global', 'hirag_bridge', 'hirag_nobridge', 'hirag_naive'])
    
    # Add engines
    graphrag_methods = []
    lightrag_modes = []
    hirag_modes = []
    for engine in engines_to_add:
        if engine.startswith('graphrag_'):
            method = engine.replace('graphrag_', '')
            graphrag_methods.append(method)
        elif engine.startswith('lightrag_'):
            mode = engine.replace('lightrag_', '')
            lightrag_modes.append(mode)
        elif engine.startswith('hirag_'):
            mode = engine.replace('hirag_', '')
            hirag_modes.append(mode)
        elif engine == 'literag':
            benchmark.add_literag_engine()
        elif engine == 'google_filesearch':
            benchmark.add_filesearch_engine()

    if graphrag_methods:
        benchmark.add_graphrag_engines(graphrag_methods)

    if lightrag_modes:
        benchmark.add_lightrag_engines(lightrag_modes)

    if hirag_modes:
        benchmark.add_hirag_engines(hirag_modes)
    
    # Run benchmark
    await benchmark.run(prompts)


if __name__ == "__main__":
    asyncio.run(main())
