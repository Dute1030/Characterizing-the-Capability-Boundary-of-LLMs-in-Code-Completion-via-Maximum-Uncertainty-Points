"""
Plot CodeGPT-cut vs Random-cut performance for multiple models.

Requires:
- CodeGPT-guided all_results.json (from run_codegpt_guided.py)
- Random baseline all_results.json (from run_experiment.py, which contains 'random' strategy)

Usage:
  python -m src.rq1.plot_codegpt_vs_random \
    --codegpt experiments/rq1_codegpt_guided_full/codegpt_guided_20251209_025226/all_results.json \
    --random experiments/rq1_random_baseline/all_results.json \
    --output experiments/rq1_codegpt_guided_full/codegpt_guided_20251209_025226/codegpt_vs_random.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot CodeGPT-cut vs Random-cut metrics")
    ap.add_argument("--codegpt", required=True, help="Path to CodeGPT all_results.json")
    ap.add_argument("--random", required=True, help="Path to Random baseline all_results.json")
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["codebleu", "exact_match", "rouge_l_f1", "bleu"],
        help="Metrics to plot",
    )
    ap.add_argument("--output", required=True, help="Output PNG path")
    return ap.parse_args()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def plot(codegpt_data: Dict, random_data: Dict, metrics: List[str], output: Path) -> None:
    models = list(codegpt_data.keys())
    num_metrics = len(metrics)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    axes = axes.flatten()
    bar_width = 0.35
    x = list(range(len(models)))

    for ax, metric in zip(axes, metrics):
        cg_vals = [codegpt_data[m]["codegpt_cut"].get(metric, {}).get("mean", 0.0) for m in models]
        rd_vals = [random_data.get(m, {}).get("random", {}).get(metric, {}).get("mean", 0.0) for m in models]

        ax.bar([i - bar_width / 2 for i in x], cg_vals, width=bar_width, label="CodeGPT-cut", color="#4C72B0")
        ax.bar([i + bar_width / 2 for i in x], rd_vals, width=bar_width, label="Random", color="#DD8452")

        ax.set_title(metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ymax = max(cg_vals + rd_vals) if cg_vals or rd_vals else 1.0
        ax.set_ylim(0, ymax * 1.2)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("CodeGPT-cut vs Random-cut Performance", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cg = load_json(Path(args.codegpt))
    rd = load_json(Path(args.random))
    plot(cg, rd, args.metrics, Path(args.output))


if __name__ == "__main__":
    main()
