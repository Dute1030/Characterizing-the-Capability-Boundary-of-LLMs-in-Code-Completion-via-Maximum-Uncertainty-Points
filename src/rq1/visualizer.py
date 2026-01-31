"""
Visualization Module for RQ1 Experiment
Creates grouped bar charts and radar charts for results comparison
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


class RQ1Visualizer:
    """Visualize RQ1 experiment results"""

    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize visualizer
        Uses consistent style with existing project visualizations

        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            # Fallback to default if style not available
            try:
                plt.style.use('seaborn-v0_8')
            except:
                plt.style.use('default')

        # Color scheme for strategies
        self.colors = {
            'random': '#1f77b4',      # Blue
            'entropy': '#ff7f0e',     # Orange
            'confidence': '#2ca02c',  # Green
            'ppl': '#d62728'          # Red
        }

    def plot_grouped_bar_chart(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],
        metric: str = 'codebleu',
        save_path: str = None,
        figsize: Tuple[float, float] = (12, 6),
        title: str = None
    ):
        """
        Create grouped bar chart comparing strategies across model sizes

        Args:
            results: Nested dict: {model_size: {strategy: {metric: value}}}
                    e.g., {'0.6b': {'random': {'codebleu': 0.5, 'em': 0.3}, ...}, ...}
            metric: Metric to visualize
            save_path: Path to save figure
            figsize: Figure size
            title: Custom title
        """
        # Extract data
        model_sizes = sorted(results.keys())
        strategies = ['random', 'entropy', 'confidence', 'ppl']

        # Prepare data for plotting
        x = np.arange(len(model_sizes))
        width = 0.2  # Width of each bar

        fig, ax = plt.subplots(figsize=figsize)

        # Plot bars for each strategy
        for i, strategy in enumerate(strategies):
            values = []
            errors = []
            for model_size in model_sizes:
                if strategy in results[model_size]:
                    val = results[model_size][strategy].get(metric, {})
                    if isinstance(val, dict):
                        values.append(val.get('mean', 0))
                        errors.append(val.get('std', 0))
                    else:
                        values.append(val)
                        errors.append(0)
                else:
                    values.append(0)
                    errors.append(0)

            offset = (i - 1.5) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=strategy.capitalize(),
                color=self.colors[strategy],
                yerr=errors,
                capsize=5,
                alpha=0.8
            )

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        f'{height:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=8
                    )

        # Customize plot
        ax.set_xlabel('Model Size', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')

        if title is None:
            title = f'{metric.upper()} Comparison Across Model Sizes and Strategies'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        ax.set_xticks(x)
        ax.set_xticklabels(model_sizes)
        ax.legend(title='Strategy', loc='best', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved grouped bar chart to {save_path}")

        plt.show()

    def plot_multiple_metrics_bar_chart(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],
        metrics: List[str] = ['codebleu', 'exact_match', 'rouge_l_f1'],
        save_path: str = None,
        figsize: Tuple[float, float] = (16, 10)
    ):
        """
        Create subplots for multiple metrics

        Args:
            results: Nested dict of results
            metrics: List of metrics to plot
            save_path: Path to save figure
            figsize: Figure size
        """
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_metrics > 1 else [axes]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            self._plot_single_metric_on_axis(ax, results, metric)

        # Hide extra subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Performance Comparison Across Metrics', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved multi-metric bar chart to {save_path}")

        plt.show()

    def _plot_single_metric_on_axis(
        self,
        ax,
        results: Dict[str, Dict[str, Dict[str, float]]],
        metric: str
    ):
        """Helper function to plot a single metric on given axis"""
        model_sizes = sorted(results.keys())
        strategies = ['random', 'entropy', 'confidence', 'ppl']

        x = np.arange(len(model_sizes))
        width = 0.2

        for i, strategy in enumerate(strategies):
            values = []
            for model_size in model_sizes:
                if strategy in results[model_size]:
                    val = results[model_size][strategy].get(metric, {})
                    if isinstance(val, dict):
                        values.append(val.get('mean', 0))
                    else:
                        values.append(val)
                else:
                    values.append(0)

            offset = (i - 1.5) * width
            ax.bar(
                x + offset,
                values,
                width,
                label=strategy.capitalize(),
                color=self.colors[strategy],
                alpha=0.8
            )

        ax.set_xlabel('Model Size', fontsize=10, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=10, fontweight='bold')
        ax.set_title(f'{metric.upper()} Comparison', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_sizes)
        ax.legend(loc='best', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    def plot_radar_chart(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = ['codebleu', 'exact_match', 'rouge_l_f1', 'bleu'],
        save_path: str = None,
        figsize: Tuple[float, float] = (10, 10),
        title: str = "Performance Comparison - Radar Chart"
    ):
        """
        Create radar chart comparing different strategies

        Args:
            results: Dict mapping strategy to metrics
                    e.g., {'random': {'codebleu': 0.5, 'em': 0.3}, ...}
            metrics: List of metrics to include
            save_path: Path to save figure
            figsize: Figure size
            title: Chart title
        """
        strategies = ['random', 'entropy', 'confidence', 'ppl']

        # Number of variables
        num_vars = len(metrics)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

        # Plot data for each strategy
        max_val = 0
        for strategy in strategies:
            if strategy not in results:
                continue

            values = []
            for metric in metrics:
                val = results[strategy].get(metric, 0)
                if isinstance(val, dict):
                    values.append(val.get('mean', 0))
                else:
                    values.append(val)

            if values:
                max_val = max(max_val, max(values))

            values += values[:1]  # Complete the circle

            ax.plot(
                angles,
                values,
                'o-',
                linewidth=2,
                label=strategy.capitalize(),
                color=self.colors[strategy]
            )
            ax.fill(angles, values, alpha=0.15, color=self.colors[strategy])

        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper().replace('_', ' ') for m in metrics], fontsize=10)

        # Set y-axis limit to max observed to let top value hit outer circle
        max_val = max(max_val, 1e-6)
        ax.set_ylim(0, max_val)
        yticks = np.linspace(0, max_val, 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:.2f}" for y in yticks], fontsize=8)

        # Add legend and title
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved radar chart to {save_path}")

        plt.show()

    def plot_radar_chart_by_model(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],
        metrics: List[str] = ['codebleu', 'exact_match', 'rouge_l_f1', 'bleu'],
        save_path: str = None,
        figsize: Tuple[float, float] = (16, 5)
    ):
        """
        Create multiple radar charts, one for each model size

        Args:
            results: Nested dict: {model_size: {strategy: {metric: value}}}
            metrics: List of metrics to include
            save_path: Path to save figure
            figsize: Figure size
        """
        model_sizes = sorted(results.keys())
        n_models = len(model_sizes)

        fig, axes = plt.subplots(1, n_models, figsize=figsize, subplot_kw=dict(projection='polar'))
        if n_models == 1:
            axes = [axes]

        for idx, model_size in enumerate(model_sizes):
            ax = axes[idx]
            self._plot_radar_on_axis(ax, results[model_size], metrics, title=f'Model: {model_size}')

        plt.suptitle('Performance Comparison Across Models', fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved multi-model radar chart to {save_path}")

        plt.show()

    def _plot_radar_on_axis(
        self,
        ax,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        title: str = ""
    ):
        """Helper function to plot radar chart on given axis"""
        strategies = ['random', 'entropy', 'confidence', 'ppl']
        num_vars = len(metrics)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        for strategy in strategies:
            if strategy not in results:
                continue

            values = []
            max_val = 0
            for metric in metrics:
                val = results[strategy].get(metric, 0)
                if isinstance(val, dict):
                    values.append(val.get('mean', 0))
                else:
                    values.append(val)
            if values:
                max_val = max(max_val, max(values))

            values += values[:1]

            ax.plot(
                angles,
                values,
                'o-',
                linewidth=2,
                label=strategy.capitalize(),
                color=self.colors[strategy],
                markersize=4
            )
            ax.fill(angles, values, alpha=0.1, color=self.colors[strategy])

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper().replace('_', '\n') for m in metrics], fontsize=8)
        max_val = max(max_val, 1e-6)
        ax.set_ylim(0, max_val)
        yticks = np.linspace(0, max_val, 4)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:.2f}" for y in yticks], fontsize=7)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=8)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)

    def plot_performance_degradation(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],
        baseline_strategy: str = 'random',
        metrics: List[str] = ['codebleu', 'exact_match'],
        save_path: str = None,
        figsize: Tuple[float, float] = (12, 6)
    ):
        """
        Plot performance degradation relative to baseline

        Args:
            results: Nested dict of results
            baseline_strategy: Strategy to use as baseline
            metrics: Metrics to compare
            save_path: Path to save figure
            figsize: Figure size
        """
        model_sizes = sorted(results.keys())
        strategies = [s for s in ['entropy', 'confidence', 'ppl'] if s != baseline_strategy]

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]

        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]

            x = np.arange(len(model_sizes))
            width = 0.25

            for i, strategy in enumerate(strategies):
                degradations = []

                for model_size in model_sizes:
                    baseline_val = results[model_size][baseline_strategy].get(metric, {})
                    strategy_val = results[model_size][strategy].get(metric, {})

                    if isinstance(baseline_val, dict):
                        baseline_val = baseline_val.get('mean', 0)
                    if isinstance(strategy_val, dict):
                        strategy_val = strategy_val.get('mean', 0)

                    # Calculate degradation percentage
                    if baseline_val > 0:
                        degradation = ((baseline_val - strategy_val) / baseline_val) * 100
                    else:
                        degradation = 0

                    degradations.append(degradation)

                offset = (i - 1) * width
                bars = ax.bar(
                    x + offset,
                    degradations,
                    width,
                    label=strategy.capitalize(),
                    color=self.colors[strategy],
                    alpha=0.8
                )

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        f'{height:.1f}%',
                        ha='center',
                        va='bottom' if height >= 0 else 'top',
                        fontsize=8
                    )

            ax.set_xlabel('Model Size', fontsize=10, fontweight='bold')
            ax.set_ylabel('Performance Degradation (%)', fontsize=10, fontweight='bold')
            ax.set_title(f'{metric.upper()} Degradation vs {baseline_strategy.capitalize()}',
                        fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(model_sizes)
            ax.legend(loc='best')
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)

        plt.suptitle(f'Performance Degradation Relative to {baseline_strategy.capitalize()}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved degradation chart to {save_path}")

        plt.show()

    def create_results_table(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],
        metrics: List[str] = ['codebleu', 'exact_match', 'rouge_l_f1'],
        save_path: str = None
    ) -> pd.DataFrame:
        """
        Create a summary table of results

        Args:
            results: Nested dict of results
            metrics: Metrics to include
            save_path: Path to save CSV
        Returns:
            DataFrame with results
        """
        rows = []

        for model_size in sorted(results.keys()):
            for strategy in ['random', 'entropy', 'confidence', 'ppl']:
                if strategy not in results[model_size]:
                    continue

                row = {'Model Size': model_size, 'Strategy': strategy}

                for metric in metrics:
                    val = results[model_size][strategy].get(metric, {})
                    if isinstance(val, dict):
                        row[f'{metric}_mean'] = val.get('mean', 0)
                        row[f'{metric}_std'] = val.get('std', 0)
                    else:
                        row[metric] = val

                rows.append(row)

        df = pd.DataFrame(rows)

        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Saved results table to {save_path}")

        return df
