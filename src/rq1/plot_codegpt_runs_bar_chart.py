"""
Plot CodeGPT-cut performance across multiple runs (e.g., entropy/confidence/ppl)
in the RQ1 bar-chart style.

Usage example:
python -m src.rq1.plot_codegpt_runs_bar_chart \
  --run entropy:experiments/rq1_codegpt_guided_entropy/codegpt_guided_20251210_000654/all_results.json \
  --run confidence:experiments/rq1_codegpt_guided_confidence/codegpt_guided_20251210_010010/all_results.json \
  --run ppl:experiments/rq1_codegpt_guided_perplexity/codegpt_guided_20251210_015443/all_results.json \
  --output experiments/rq1_codegpt_guided_full/bar_chart_all_metrics_codegpt_three_runs.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CodeGPT-cut performance across multiple runs in bar-chart style"
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="label:path_to_all_results.json (e.g., entropy:.../all_results.json)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["codebleu", "exact_match", "rouge_l_f1", "bleu"],
        help="Evaluation metrics to plot",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional explicit model order (default: inferred from first run)",
    )
    parser.add_argument("--title", default="CodeGPT-cut Performance Comparison Across Metrics")
    parser.add_argument("--output", required=True, help="Output PNG path")
    return parser.parse_args()


def load_runs(run_specs: List[str]) -> List[Tuple[str, Dict]]:
    runs = []
    for spec in run_specs:
        if ":" not in spec:
            raise ValueError(f"Invalid --run format: {spec}")
        label, path_str = spec.split(":", 1)
        data = json.loads(Path(path_str).read_text())
        runs.append((label, data))
    return runs


def plot_runs(
    runs: List[Tuple[str, Dict]], metrics: List[str], models: List[str], title: str, output: Path
) -> None:
    num_models = len(models)
    num_runs = len(runs)
    if num_models == 0 or num_runs == 0:
        raise ValueError("No models or runs to plot")

    # Use Tableau palette to mirror the reference colors
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    colors = base_colors[:num_runs]

    bar_width = 0.18
    x_base = list(range(num_models))

    num_metrics = len(metrics)
    n_cols = 2 if num_metrics > 1 else 1
    n_rows = (num_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 9), dpi=160)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        ymax = 0.0

        for j, (label, data) in enumerate(runs):
            vals = [data[m]["codegpt_cut"].get(metric, {}).get("mean", 0.0) for m in models]
            offsets = [xb + (j - (num_runs - 1) / 2) * bar_width for xb in x_base]
            ax.bar(offsets, vals, width=bar_width, color=colors[j], label=label.capitalize())
            ymax = max(ymax, max(vals) if vals else 0.0)

        ax.set_title(metric.upper())
        ax.set_xticks(x_base)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Handle unused subplots when metrics < axes
    for extra_ax in axes_flat[num_metrics:]:
        extra_ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(num_runs, 4), frameon=False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = load_runs(args.run)
    models = args.models if args.models else list(runs[0][1].keys())
    output_path = Path(args.output)
    plot_runs(runs, args.metrics, models, args.title, output_path)


if __name__ == "__main__":
    main()
