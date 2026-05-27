"""
Benchmark Visualization Module

Generates plots and charts for benchmark results comparison.
Supports multiple search engines with distinct color coding.

Based on the visualization methodology from GraphRAG Unified Benchmark.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import base64
from datetime import datetime

LOGGER = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Plot configuration
PLOT_DPI = 300
FIGURE_SIZE_STANDARD = (10, 6)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _format_large_number(num, decimals=1):
    """Formats a number into a string with K, M suffixes."""
    if pd.isna(num) or num == 0:
        return "0"
    if abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.{decimals}f}M"
    if abs(num) >= 1_000:
        return f"{num / 1_000:.{decimals}f}K"
    # If it's an integer (no decimals), show without decimals
    if num == int(num):
        return f"{int(num)}"
    return f"{num:.2f}"

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def _plot_accuracy_metrics(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Generate comprehensive accuracy metrics visualization"""
    metrics_to_plot = ['answer_accuracy', 'llm_judged_correctness', 'semantic_similarity', 'rougeL_f1']
    available_metrics = [m for m in metrics_to_plot if m in df.columns and df[m].notna().any()]
    
    if not available_metrics:
        return None
    
    # Create figure with only 2 subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Quality Metrics Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Bar plot: Average scores by method
    ax1 = axes[0]
    avg_scores = df.groupby('method')[available_metrics].mean()
    if 'answer_accuracy' in avg_scores.columns:
        avg_scores_sorted = avg_scores.sort_values(by='answer_accuracy', ascending=False)
    else:
        avg_scores_sorted = avg_scores
    
    x_pos = np.arange(len(avg_scores_sorted))
    width = 0.2
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    metric_labels = {
        'answer_accuracy': 'Overall Accuracy',
        'llm_judged_correctness': 'Correctness (LLM)',
        'semantic_similarity': 'Semantic Similarity',
        'rougeL_f1': 'ROUGE-L F1'
    }
    
    for i, metric in enumerate(available_metrics):
        bars = ax1.bar(
            x_pos + i * width, 
            avg_scores_sorted[metric], 
            width, 
            label=metric_labels.get(metric, metric),
            color=colors[i % len(colors)],
            alpha=0.8
        )
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Search Method', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('Average Quality Scores by Method', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos + width * (len(available_metrics) - 1) / 2)
    ax1.set_xticklabels(avg_scores_sorted.index, rotation=0)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 2. Box plot: Distribution of answer accuracy by method
    ax2 = axes[1]
    if 'answer_accuracy' in df.columns:
        methods_order = avg_scores_sorted.index.tolist()
        df_sorted = df.copy()
        df_sorted['method'] = pd.Categorical(df_sorted['method'], categories=methods_order, ordered=True)
        df_sorted = df_sorted.sort_values('method')
        
        bp = ax2.boxplot(
            [df_sorted[df_sorted['method'] == method]['answer_accuracy'].dropna() 
            for method in methods_order],
            labels=methods_order,
            patch_artist=True,
            notch=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=6, markeredgecolor='darkred')
        )
        
        # Color boxes
        for patch, color in zip(bp['boxes'], plt.cm.Set3.colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Search Method', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Answer Accuracy Score', fontsize=11, fontweight='bold')
        ax2.set_title('Distribution of Answer Accuracy by Method', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1.05)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
    else:
        ax2.text(0.5, 0.5, 'Answer Accuracy not available', ha='center', va='center')
    
    plt.tight_layout()
    path = output_dir / 'accuracy_comprehensive.png'
    plt.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    return path

def _plot_latency(df: pd.DataFrame, output_dir: Path) -> Path:
    plt.figure(figsize=FIGURE_SIZE_STANDARD)
    sns.boxplot(data=df, x='method', y='latency_seconds')
    plt.title('Query Latency by Method', fontsize=14, fontweight='bold')
    plt.xlabel('Search Method')
    plt.ylabel('Latency (seconds)')
    plt.tight_layout()
    path = output_dir / 'latency_by_method.png'
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    return path

def _plot_cost(df: pd.DataFrame, output_dir: Path) -> Path:
    plt.figure(figsize=FIGURE_SIZE_STANDARD)
    cost_by_method = df.groupby('method')['cost_eur'].sum().sort_values(ascending=False)
    ax = cost_by_method.plot(kind='bar', color='#3498db')
    
    ax.bar_label(ax.containers[0], 
                 fmt=lambda x: f'${_format_large_number(x, decimals=2)}', 
                 fontsize=10, padding=3)
    
    plt.title('Total Cost by Method', fontsize=14, fontweight='bold')
    plt.xlabel('Search Method')
    plt.ylabel('Cost (EUR)')
    ax.tick_params(axis='x', rotation=0)
    ax.set_ylim(0, cost_by_method.max() * 1.15) # Add space for labels
    plt.tight_layout()
    path = output_dir / 'cost_by_method.png'
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    return path

def _plot_tokens(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    
    # Token breakdown
    token_breakdown = df.groupby('method')[['prompt_tokens', 'completion_tokens']].sum()
    token_breakdown = token_breakdown.sort_values(by='prompt_tokens', ascending=True)  # Sort by prompt_tokens
    
    token_breakdown.plot(kind='bar', ax=ax, stacked=True)
    
    totals = token_breakdown.sum(axis=1)
    for i, total in enumerate(totals):
        ax.text(i, total, _format_large_number(total, decimals=2), 
                 ha='center', va='bottom', fontsize=10)

    ax.set_title('Token Breakdown by Method', fontsize=12)
    ax.set_xlabel('Search Method')
    ax.set_ylabel('Tokens')
    ax.legend(['Prompt', 'Completion'])
    ax.tick_params(axis='x', rotation=0)
    ax.set_ylim(0, totals.max() * 1.15)  # Adjust y-axis limit
    
    # Format y-axis to avoid scientific notation
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: _format_large_number(y, decimals=1)))

    plt.tight_layout()
    path = output_dir / 'tokens_by_method.png'
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    return path

def _plot_llm_calls(df: pd.DataFrame, output_dir: Path) -> Path:
    plt.figure(figsize=FIGURE_SIZE_STANDARD)
    calls_by_method = df.groupby('method')['llm_calls'].sum().sort_values(ascending=False)
    ax = calls_by_method.plot(kind='bar', color='#e74c3c')

    ax.bar_label(ax.containers[0], 
                 fmt=lambda x: _format_large_number(x), 
                 fontsize=10, padding=3)

    plt.title('Total LLM Calls by Method', fontsize=14, fontweight='bold')
    plt.xlabel('Search Method')
    plt.ylabel('Number of Calls')
    ax.tick_params(axis='x', rotation=0)
    ax.set_ylim(0, calls_by_method.max() * 1.15)
    plt.tight_layout()
    path = output_dir / 'llm_calls_by_method.png'
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    return path

def _plot_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generate comprehensive heatmap visualizations"""
    
    # Extract prompt category from prompt name (e.g., "literal_citation_1" -> "LC")
    def extract_category(prompt_name):
        if not isinstance(prompt_name, str):
            return 'Other'
        if 'literal' in prompt_name.lower() or prompt_name.startswith('lc_'):
            return 'LC (Basic Query)'
        elif 'local' in prompt_name.lower() or prompt_name.startswith('lr_'):
            return 'LR (Local Query)'
        elif 'global' in prompt_name.lower() or prompt_name.startswith('g_'):
            return 'G (Global Query)'
        elif 'drift' in prompt_name.lower() or prompt_name.startswith('d_'):
            return 'D (Drift Query)'
        else:
            return 'Other'
    
    df = df.copy()
    df['prompt_category'] = df['prompt_name'].apply(extract_category)
    
    # Map method names to more readable labels if needed
    # (Assuming they are already readable or handled by the user's data)
    df['search_method'] = df['method']
    
    # Metrics to visualize
    metrics = {
        'latency_seconds': ('Avg. Latency per Query (s)', 'Reds'),
        'total_tokens': ('Avg. Tokens per Query', 'YlOrRd'),
        'cost_eur': ('Avg. Cost per Query ($)', 'Greens'),
        'answer_accuracy': ('Avg. Overall Accuracy Score', 'RdYlGn')
    }
    
    # Create figure with 4 subplots (2x2 grid) with more spacing
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    axes = axes.flatten()
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    
    for idx, (metric, (title, cmap)) in enumerate(metrics.items()):
        ax = axes[idx]
        
        # Skip if metric not available
        if metric not in df.columns or df[metric].isna().all():
            ax.text(0.5, 0.5, f'{metric} data not available', 
                ha='center', va='center', fontsize=12)
            ax.set_title(title)
            continue
        
        # Create pivot table
        pivot_data = df.pivot_table(
            values=metric, 
            index='prompt_category', 
            columns='search_method', 
            aggfunc='mean'
        )
        
        # Reorder columns and rows for consistency if possible
        # We try to detect standard methods and categories
        method_order = ['graphrag_basic', 'graphrag_local', 'graphrag_global', 'graphrag_drift', 'literag']
        category_order = ['LC (Basic Query)', 'LR (Local Query)', 'G (Global Query)', 'D (Drift Query)']
        
        # Filter to existing
        existing_methods = [m for m in method_order if m in pivot_data.columns]
        # If we found known methods, use that order, otherwise use default sort
        if existing_methods:
             # Add any other methods not in the list
            others = [m for m in pivot_data.columns if m not in existing_methods]
            pivot_data = pivot_data.reindex(columns=existing_methods + others)
            
        pivot_data = pivot_data.reindex(
            index=[c for c in category_order if c in pivot_data.index] + 
                  [c for c in pivot_data.index if c not in category_order]
        )
        
        # Create heatmap
        sns.heatmap(
            pivot_data, 
            annot=True, 
            fmt='.2f' if metric in ['cost_eur', 'answer_accuracy'] else ('.0f' if metric == 'total_tokens' else '.2f'),
            cmap=cmap, 
            cbar_kws={'label': title},
            linewidths=0.5,
            ax=ax,
            vmin=0 if metric == 'answer_accuracy' else None,
            vmax=1 if metric == 'answer_accuracy' else None,
            annot_kws={'size': 11, 'weight': 'bold'}
        )
        
        # Format annotations for better readability
        if metric == 'total_tokens':
            for text in ax.texts:
                try:
                    val = float(text.get_text())
                    text.set_text(_format_large_number(val, decimals=2))
                except ValueError:
                    pass
        
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
        ax.set_xlabel('Search Method', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_ylabel('Prompt Category', fontsize=13, fontweight='bold', labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.suptitle('Average Performance Metrics Heatmap by Prompt Category', 
                fontsize=18, fontweight='bold', y=0.995)
    
    path = output_dir / "performance_heatmap_by_category.png"
    plt.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    return path

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def create_plots(
    results_df: pd.DataFrame,
    output_dir: Path,
    show_plots: bool = False
) -> Dict[str, Path]:
    """
    Generate all benchmark plots using the comprehensive style.
    
    Args:
        results_df: DataFrame with benchmark results
        output_dir: Directory to save plots
        show_plots: Whether to display plots interactively (not used in this version but kept for signature compatibility)
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    if not show_plots:
        plt.switch_backend('Agg')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Prepare DataFrame: map 'engine' to 'method' for compatibility with plotting logic
    df = results_df.copy()
    if 'engine' in df.columns and 'method' not in df.columns:
        df['method'] = df['engine']
    
    plots = {}
    
    print("Generating benchmark plots...")
    
    try:
        path = _plot_latency(df, output_dir)
        plots['latency_by_method'] = path
        print(f"  ✓ Latency plot saved: {path.name}")
    except Exception as e:
        print(f"  ⚠️ Failed to generate latency plot: {e}")

    try:
        path = _plot_cost(df, output_dir)
        plots['cost_by_method'] = path
        print(f"  ✓ Cost plot saved: {path.name}")
    except Exception as e:
        print(f"  ⚠️ Failed to generate cost plot: {e}")

    try:
        path = _plot_tokens(df, output_dir)
        plots['tokens_by_method'] = path
        print(f"  ✓ Tokens plot saved: {path.name}")
    except Exception as e:
        print(f"  ⚠️ Failed to generate tokens plot: {e}")

    try:
        path = _plot_llm_calls(df, output_dir)
        plots['llm_calls_by_method'] = path
        print(f"  ✓ LLM calls plot saved: {path.name}")
    except Exception as e:
        print(f"  ⚠️ Failed to generate LLM calls plot: {e}")

    try:
        path = _plot_heatmap(df, output_dir)
        plots['performance_heatmap_by_category'] = path
        print(f"  ✓ Heatmap saved: {path.name}")
    except Exception as e:
        print(f"  ⚠️ Failed to generate heatmap: {e}")

    try:
        path = _plot_accuracy_metrics(df, output_dir)
        if path:
            plots['accuracy_comprehensive'] = path
            print(f"  ✓ Accuracy metrics plot saved: {path.name}")
    except Exception as e:
        print(f"  ⚠️ Failed to generate accuracy plot: {e}")
        if LOGGER.isEnabledFor(logging.INFO):
            LOGGER.exception("Accuracy plot generation failed")

    return plots

# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def generate_html_report(
    results_df: pd.DataFrame,
    summary: Dict[str, Any],
    plots: Dict[str, Path],
    output_dir: Path
) -> Path:
    """
    Generate a comprehensive HTML report with embedded plots.
    """
    output_dir = Path(output_dir)
    
    def embed_image(path: Path) -> str:
        """Convert image to base64 for embedding in HTML"""
        if path.exists():
            with open(path, 'rb') as f:
                data = base64.b64encode(f.read()).decode()
            return f'data:image/png;base64,{data}'
        return ''
    
    # Build HTML document
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Benchmark Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #333; border-bottom: 3px solid #3A7D44; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 40px; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #3A7D44;
        }}
        .summary-card .label {{
            color: #666;
            margin-top: 5px;
        }}
        .plot-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #3A7D44;
            color: white;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .engine-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .literag {{ background: #e8f5e9; color: #2e7d32; }}
        .graphrag_local {{ background: #fce4ec; color: #c2185b; }}
        .graphrag_global {{ background: #fff3e0; color: #ef6c00; }}
        .graphrag_drift {{ background: #ffebee; color: #c62828; }}
        .graphrag_basic {{ background: #e3f2fd; color: #1565c0; }}
    </style>
</head>
<body>
    <h1>🔬 Search Engine Benchmark Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>📊 Summary</h2>
    <div class="summary-grid">
        <div class="summary-card">
            <div class="value">{summary.get('total_queries', 0)}</div>
            <div class="label">Total Queries</div>
        </div>
        <div class="summary-card">
            <div class="value">{summary.get('successful_queries', 0)}</div>
            <div class="label">Successful</div>
        </div>
        <div class="summary-card">
            <div class="value">{summary.get('total_time_seconds', 0):.1f}s</div>
            <div class="label">Total Time</div>
        </div>
        <div class="summary-card">
            <div class="value">${summary.get('total_cost_eur', 0):.4f}</div>
            <div class="label">Total Cost</div>
        </div>
        <div class="summary-card">
            <div class="value">{summary.get('total_tokens', 0):,}</div>
            <div class="label">Total Tokens</div>
        </div>
        <div class="summary-card">
            <div class="value">{summary.get('avg_answer_accuracy', 0):.2f}</div>
            <div class="label">Avg Accuracy</div>
        </div>
    </div>
"""
    
    # Engine comparison table
    html += """
    <h2>🔧 Engine Comparison</h2>
    <table>
        <tr>
            <th>Engine</th>
            <th>Queries</th>
            <th>Avg Latency</th>
            <th>Avg Tokens</th>
            <th>Total Cost</th>
            <th>Avg Accuracy</th>
        </tr>
"""
    
    engine_stats = summary.get('engine_stats', {})
    for engine, stats in engine_stats.items():
        html += f"""
        <tr>
            <td><span class="engine-badge {engine}">{engine}</span></td>
            <td>{stats.get('queries', 0)}</td>
            <td>{stats.get('avg_latency', 0):.2f}s</td>
            <td>{stats.get('avg_tokens', 0):,.0f}</td>
            <td>${stats.get('total_cost', 0):.4f}</td>
            <td>{stats.get('avg_accuracy', 'N/A') if isinstance(stats.get('avg_accuracy'), str) else f"{stats.get('avg_accuracy', 0):.2f}"}</td>
        </tr>
"""
    
    html += "</table>"
    
    # Add plots
    plot_titles = {
        'accuracy_comprehensive': '🎯 Comprehensive Accuracy Analysis',
        'performance_heatmap_by_category': '🗺️ Performance Heatmap',
        'latency_by_method': '⏱️ Latency Distribution',
        'cost_by_method': '💰 Cost Comparison',
        'tokens_by_method': '📝 Token Usage',
        'llm_calls_by_method': '🤖 LLM Calls',
    }
    
    for plot_key, title in plot_titles.items():
        if plot_key in plots:
            html += f"""
    <h2>{title}</h2>
    <div class="plot-container">
        <img src="{embed_image(plots[plot_key])}" alt="{title}">
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    report_path = output_dir / 'benchmark_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return report_path
