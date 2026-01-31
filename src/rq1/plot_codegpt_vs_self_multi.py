"""
Plot CodeGPT-cut vs self-cut performance across multiple uncertainty metrics (entropy/confidence/ppl).

Usage example:
  python -m src.rq1.plot_codegpt_vs_self_multi \
    --run entropy:experiments/rq1_codegpt_guided_full/codegpt_guided_20251209_025226/all_results.json \
    --run confidence:experiments/rq1_codegpt_guided_full_conf/codegpt_guided_conf/all_results.json \
    --run ppl:experiments/rq1_codegpt_guided_full_ppl/codegpt_guided_ppl/all_results.json \
    --output experiments/rq1_codegpt_guided_full/codegpt_guided_20251209_025226/codegpt_vs_self_multi.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot CodeGPT-cut vs self-cut across multiple uncertainty metrics")
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        help="name:path_to_all_results.json (e.g., entropy:.../all_results.json)",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["codebleu", "exact_match", "rouge_l_f1", "bleu"],
        help="Evaluation metrics to plot",
    )
    ap.add_argument("--output", required=True, help="Output PNG path")
    return ap.parse_args()


def load_runs(run_specs: List[str]) -> List[Tuple[str, Dict]]:
    runs = []
    for spec in run_specs:
        if ":" not in spec:
            raise ValueError(f"Invalid --run format: {spec}")
        name, path = spec.split(":", 1)
        data = json.loads(Path(path).read_text())
        runs.append((name, data))
    return runs


def plot(runs: List[Tuple[str, Dict]], metrics: List[str], output: Path) -> None:
    # Infer model order from first run
    models = list(runs[0][1].keys())
    num_models = len(models)
    num_labels = len(runs)

    colors = list(mcolors.TABLEAU_COLORS.values())
    if num_labels > len(colors):
        colors = colors * ((num_labels // len(colors)) + 1)

    bar_width = 0.12
    label_span = 2 * bar_width  # codegpt + self per label

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    axes = axes.flatten()

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        x_base = list(range(num_models))

        for j, (label, data) in enumerate(runs):
            cg_vals = [data[m]["codegpt_cut"].get(metric, {}).get("mean", 0.0) for m in models]
            sf_vals = [data[m]["self_cut"].get(metric, {}).get("mean", 0.0) for m in models]

            offsets = [xb + (j - (num_labels - 1) / 2) * label_span for xb in x_base]
            ax.bar(
                [o - bar_width / 2 for o in offsets],
                cg_vals,
                width=bar_width,
                color=colors[j],
                label=f"{label} (cg)" if ax_idx == 0 else None,
            )
            ax.bar(
                [o + bar_width / 2 for o in offsets],
                sf_vals,
                width=bar_width,
                color=colors[j],
                alpha=0.4,
                hatch="//",
                label=f"{label} (self)" if ax_idx == 0 else None,
            )

        ax.set_title(metric.upper())
        ax.set_xticks(x_base)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Dynamic y-limit
        ymax = 0.0
        for _, data in runs:
            for m in models:
                cg = data[m]["codegpt_cut"].get(metric, {}).get("mean", 0.0)
                sf = data[m]["self_cut"].get(metric, {}).get("mean", 0.0)
                ymax = max(ymax, cg, sf)
        ax.set_ylim(0, ymax * 1.2 if ymax > 0 else 1.0)

    # Only add legend once (from first axis labels)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(2 * num_labels, 6), frameon=False)
    fig.suptitle("CodeGPT-cut vs Self-cut across uncertainty metrics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = load_runs(args.run)
    output = Path(args.output)
    plot(runs, args.metrics, output)


if __name__ == "__main__":
    main()
