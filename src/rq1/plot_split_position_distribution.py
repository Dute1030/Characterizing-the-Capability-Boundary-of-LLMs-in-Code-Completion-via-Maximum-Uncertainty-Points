#!/usr/bin/env python
"""
Plot distribution of split token positions from *_split_positions.csv.

Usage (run from project root):
  python src/rq1/plot_split_position_distribution.py \
    --runs experiments/rq1_full_results experiments/rq1_full_results_new_20251127_132302 \
    --bins 50

Each run dir will produce split_position_distribution.png (all strategies overlaid)
and split_position_distribution_by_strategy.png (four separate subplots).
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_distribution(df: pd.DataFrame, save_path: Path, bins: int = 50):
    strategies = ['random', 'entropy', 'confidence', 'ppl']
    plt.figure(figsize=(10, 6))
    max_idx = df['split_token_index'].dropna().max()
    if pd.isna(max_idx):
        max_idx = 1
    bin_edges = np.linspace(0, max_idx, bins + 1)

    for strat in strategies:
        sub = df[df['strategy'] == strat]
        plt.hist(
            sub['split_token_index'].dropna(),
            bins=bin_edges,
            alpha=0.6,
            label=strat.capitalize(),
            edgecolor='none'
        )

    plt.xlabel('Split token index')
    plt.ylabel('Count')
    plt.title('Distribution of split token positions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved distribution to {save_path}")


def plot_distribution_by_strategy(df: pd.DataFrame, save_path: Path, bins: int = 50):
    strategies = ['random', 'entropy', 'confidence', 'ppl']
    max_idx = df['split_token_index'].dropna().max()
    if pd.isna(max_idx):
        max_idx = 1
    bin_edges = np.linspace(0, max_idx, bins + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    colors = {'random': '#1f77b4', 'entropy': '#ff7f0e', 'confidence': '#2ca02c', 'ppl': '#d62728'}

    for ax, strat in zip(axes, strategies):
        sub = df[df['strategy'] == strat]
        ax.hist(
            sub['split_token_index'].dropna(),
            bins=bin_edges,
            alpha=0.85,
            color=colors[strat],
            edgecolor='none'
        )
        ax.set_title(strat.capitalize())
        ax.set_xlabel('Split token index')
        ax.set_ylabel('Count')
        ax.grid(alpha=0.3)

    plt.suptitle('Distribution of split token positions by strategy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved per-strategy distribution to {save_path}")


def process_run(run_dir: Path, bins: int):
    csv_files = list(run_dir.glob("*_split_positions.csv"))
    if not csv_files:
        print(f"No split_positions csv found in {run_dir}")
        return
    frames = [pd.read_csv(p) for p in csv_files]
    merged = pd.concat(frames, ignore_index=True)
    out_path = run_dir / "split_position_distribution.png"
    plot_distribution(merged, out_path, bins=bins)
    out_path_by = run_dir / "split_position_distribution_by_strategy.png"
    plot_distribution_by_strategy(merged, out_path_by, bins=bins)


def main():
    parser = argparse.ArgumentParser(description="Plot split token position distributions")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories containing *_split_positions.csv")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    args = parser.parse_args()

    for run in args.runs:
        process_run(Path(run), bins=args.bins)


if __name__ == "__main__":
    main()
