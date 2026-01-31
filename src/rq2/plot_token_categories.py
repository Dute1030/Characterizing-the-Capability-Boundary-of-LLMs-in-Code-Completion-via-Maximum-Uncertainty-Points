import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


def derive_model_name(path: Path) -> str:
    m = re.search(r"token_node_categories_(.+)\.csv", path.name)
    return m.group(1) if m else path.stem


def plot_single(df: pd.DataFrame, model_name: str, out_path: Path):
    counts = (
        df.groupby(["strategy", "category"])
        .size()
        .reset_index(name="count")
        .pivot(index="category", columns="strategy", values="count")
        .fillna(0)
        .sort_index()
    )

    ax = counts.plot(kind="bar", figsize=(10, 6))
    ax.set_title(f"Split token AST categories ({model_name})")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.legend(title="Strategy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot distribution of split token AST categories.")
    ap.add_argument("--inputs", nargs="+", required=True, help="token_node_categories_*.csv files")
    ap.add_argument("--output_dir", default="experiments/rq4_results", help="Where to save figures")
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
        plot_single(df, model_name, out_dir / f"category_distribution_{model_name}.png")


if __name__ == "__main__":
    main()
