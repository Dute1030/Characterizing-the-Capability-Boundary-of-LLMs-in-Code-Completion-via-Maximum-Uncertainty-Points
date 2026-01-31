import argparse
import csv
import json
import keyword
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Distribution of uncertainty token types (Model/Strategy comparison)")
    ap.add_argument(
        "--model-csv",
        action="append",
        required=True,
        help="Format name:path/to/split_positions.csv, can be passed multiple times",
    )
    ap.add_argument(
        "--strategies",
        nargs="+",
        default=["entropy", "entropy_min"],
        help="Strategy names to compare (corresponding to the 'strategy' column in CSV)",
    )
    ap.add_argument(
        "--output-dir",
        default="experiments/rq6/token_type_bars",
        help="Output directory for plots and JSON counts",
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="Whether to normalize by frequency (defaults to raw counts)",
    )
    return ap.parse_args()


PY_KEYWORDS = set(keyword.kwlist)
OPERATORS = {
    "+", "-", "*", "/", "//", "%", "**", "=", "==", "!=", "<", "<=", ">", ">=",
    "and", "or", "not", "in", "is", "+=", "-=", "*=", "/=", "%=",
}
PUNCT = {",", ".", ":", ";", "(", ")", "[", "]", "{", "}"}
NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def classify_token(tok: str) -> str:
    """
    Coarse classification: keyword, identifier, number, string, operator, punct, whitespace, newline, etc.
    """
    if tok is None:
        return "unknown"
    raw = tok
    if raw.strip() == "":
        return "whitespace"
    if "\n" in raw:
        return "newline"

    t = raw.strip()

    # String literals
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        return "string"
    if t.startswith("'''") or t.startswith('"""'):
        return "string"

    # Comments
    if t.startswith("#"):
        return "comment"

    # Numbers
    if NUMBER_RE.match(t):
        return "number"

    # Keywords
    if t in PY_KEYWORDS:
        return "keyword"

    # Operators
    if t in OPERATORS:
        return "operator"

    # Punctuation
    if t in PUNCT:
        return "punct"

    # Identifiers
    if IDENT_RE.match(t):
        return "identifier"

    return "other"


def load_csv(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_counts(rows: List[Dict], strategies: List[str]) -> Dict[str, Counter]:
    """
    Returns {strategy: Counter(category -> count)}
    """
    counts: Dict[str, Counter] = {s: Counter() for s in strategies}
    for row in rows:
        strat = row.get("strategy")
        if strat not in strategies:
            continue
        tok = row.get("split_token_text", "")
        cat = classify_token(tok)
        counts[strat][cat] += 1
    return counts


def plot_bars(model: str, counts: Dict[str, Counter], normalize: bool, out_path: Path):
    # Collect all categories
    categories = sorted({c for counter in counts.values() for c in counter.keys()})
    if not categories:
        print(f"[warn] no categories found for model {model}")
        return

    strategies = list(counts.keys())
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(strategies))

    plt.figure(figsize=(max(6, len(categories) * 0.6), 4))

    for i, strat in enumerate(strategies):
        vals = []
        total = sum(counts[strat].values()) or 1
        for cat in categories:
            if normalize:
                vals.append(counts[strat].get(cat, 0) / total)
            else:
                vals.append(counts[strat].get(cat, 0))
        plt.bar(x + i * width, vals, width=width, label=strat)

    plt.xticks(x + width * (len(strategies) - 1) / 2, categories, rotation=20)
    plt.ylabel("Relative Frequency" if normalize else "Frequency (Count)")
    plt.title(f"Uncertainty Token Type Distribution: {model}")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot: {out_path}")


def save_counts_json(all_counts: Dict[str, Dict[str, Counter]], out_path: Path):
    serializable: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(dict)
    for model, strat_counts in all_counts.items():
        for strat, counter in strat_counts.items():
            serializable[model][strat] = dict(counter)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"Saved counts JSON: {out_path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths: List[Tuple[str, Path]] = []
    for item in args.model_csv:
        if ":" not in item:
            raise ValueError(f"--model-csv requires 'name:path' format, received: {item}")
        name, path_str = item.split(":", 1)
        model_paths.append((name, Path(path_str)))

    all_counts: Dict[str, Dict[str, Counter]] = {}

    for name, path in model_paths:
        if not path.exists():
            print(f"[warn] skipping model {name}, CSV not found: {path}")
            continue
        rows = load_csv(path)
        counts = build_counts(rows, args.strategies)
        all_counts[name] = counts
        out_png = output_dir / f"{name}_token_types.png"
        plot_bars(name, counts, args.normalize, out_png)

    if all_counts:
        save_counts_json(all_counts, output_dir / "token_type_counts.json")


if __name__ == "__main__":
    main()