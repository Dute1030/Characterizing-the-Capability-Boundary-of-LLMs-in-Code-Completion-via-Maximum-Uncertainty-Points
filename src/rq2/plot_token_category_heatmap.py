#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def derive_model_name(path: Path) -> str:
    m = re.search(r"token_node_categories_(.+)\.csv", path.name)
    return m.group(1) if m else path.stem


def plot_heatmap(df: pd.DataFrame, model_name: str, out_path: Path, drop_parser_error: bool):
    if drop_parser_error:
        df = df[df["category"] != "parser_error"]

    table = (
        df.groupby(["category", "strategy"])
        .size()
        .reset_index(name="count")
        .pivot(index="category", columns="strategy", values="count")
        .fillna(0)
        .sort_index()
    )

    plt.figure(figsize=(8, max(4, 0.35 * len(table.index))))
    ax = sns.heatmap(table, annot=True, fmt=".0f", cmap="Blues")
    ax.set_title(f"Split token AST categories heatmap ({model_name})")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Category")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved heatmap -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot heatmap of split token AST categories.")
    ap.add_argument("--inputs", nargs="+", required=True, help="token_node_categories_*.csv files")
    ap.add_argument("--output_dir", default="experiments/rq4_results", help="Where to save figures")
    ap.add_argument("--drop_parser_error", action="store_true", help="Ignore parser_error rows in the plot")
    ap.add_argument("--exclude_other", action="store_true", help="Exclude category 'other' from the plot")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in args.inputs:
        path = Path(csv_path)
        if not path.exists():
            print(f"Skip missing file: {path}")
            continue
        df = pd.read_csv(path)
        if args.exclude_other:
            df = df[df["category"] != "other"]
        model_name = derive_model_name(path)
        plot_heatmap(df, model_name, out_dir / f"category_heatmap_{model_name}.png", args.drop_parser_error)


if __name__ == "__main__":
    main()
