"""
Plot before/after (CodeGPT-cut vs self-cut) performance for multiple models.

Usage:
  python -m src.rq1.plot_codegpt_vs_self \
    --results experiments/rq1_codegpt_guided_full/codegpt_guided_20251209_025226/all_results.json \
    --output experiments/rq1_codegpt_guided_full/codegpt_guided_20251209_025226/codegpt_vs_self_bar.png
"""

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot CodeGPT-cut vs self-cut metrics for Qwen models")
    ap.add_argument("--results", required=True, help="Path to all_results.json from run_codegpt_guided.py")
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["codebleu", "exact_match", "rouge_l_f1", "bleu"],
        help="Metrics to plot",
    )
    ap.add_argument("--output", required=True, help="Output image path (png)")
    return ap.parse_args()


def plot_bar(results_path: Path, metrics: List[str], output_path: Path) -> None:
    data = json.loads(results_path.read_text())
    models = list(data.keys())

    # Collect values
    values_codegpt = {m: [] for m in metrics}
    values_self = {m: [] for m in metrics}

    for model in models:
        res = data[model]
        cg = res.get("codegpt_cut", {})
        sf = res.get("self_cut", {})
        for metric in metrics:
            values_codegpt[metric].append(cg.get(metric, {}).get("mean", 0.0))
            values_self[metric].append(sf.get(metric, {}).get("mean", 0.0))

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 4), dpi=160)
    if num_metrics == 1:
        axes = [axes]

    bar_width = 0.35
    x = list(range(len(models)))

    for ax, metric in zip(axes, metrics):
        cg_vals = values_codegpt[metric]
        sf_vals = values_self[metric]

        ax.bar([i - bar_width / 2 for i in x], cg_vals, width=bar_width, label="CodeGPT-cut", color="#4C72B0")
        ax.bar([i + bar_width / 2 for i in x], sf_vals, width=bar_width, label="Self-cut", color="#55A868")

        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylim(0, max(cg_vals + sf_vals) * 1.2 if cg_vals or sf_vals else 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if ax == axes[0]:
            ax.legend()

    fig.suptitle("CodeGPT-cut vs Self-cut Performance", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    output_path = Path(args.output)
    plot_bar(results_path, args.metrics, output_path)


if __name__ == "__main__":
    main()
