
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="AST macro-category distribution for uncertainty tokens (by model/strategy)")
    ap.add_argument(
        "--model-csv",
        action="append",
        required=True,
        help="Format name:path/to/split_positions.csv (can provide multiple)",
    )
    ap.add_argument(
        "--strategies",
        nargs="+",
        default=["entropy", "entropy_min"],
        help="Strategy names to compare (matches CSV 'strategy' column)",
    )
    ap.add_argument(
        "--output-dir",
        default="experiments/rq6/token_ast_groups",
        help="Output directory (images + JSON)",
    )
    return ap.parse_args()


CONTROL_FLOW = {
    "if",
    "elif",
    "else",
    "for",
    "while",
    "break",
    "continue",
    "return",
    "yield",
    "try",
    "except",
    "finally",
    "raise",
    "match",
    "case",
}

OPERATORS = {
    "+",
    "-",
    "*",
    "/",
    "//",
    "%",
    "**",
    "=",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "and",
    "or",
    "not",
    "in",
    "is",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "**=",
    "//=",
}

PROGRAM_STRUCTURE = {
    "def",
    "class",
    "import",
    "from",
    "as",
    "@",
    "async",
}

LITERAL_CONST = {"True", "False", "None"}
NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def classify_token(tok: str) -> str:
    if tok is None:
        return "Other"

    raw = tok
    if raw.strip() == "" or "\n" in raw:
        return "Other"

    t = raw.strip()
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        return "Literal"
    if t.startswith("'''") or t.startswith('"""'):
        return "Literal"
    if t in LITERAL_CONST:
        return "Literal"
    if NUMBER_RE.match(t):
        return "Literal"

    if t in PROGRAM_STRUCTURE:
        return "Program Structure"
    if t in CONTROL_FLOW:
        return "Control Flow"
    if t in OPERATORS:
        return "Operator"
    if IDENT_RE.match(t):
        return "Identifier"

    return "Expression"


def load_csv(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_counts(rows: List[Dict], strategies: List[str]) -> Dict[str, Counter]:
    counts: Dict[str, Counter] = {s: Counter() for s in strategies}
    default_cats = ["Control Flow", "Operator", "Literal", "Identifier", "Program Structure", "Expression", "Other"]
    for strat in counts:
        counts[strat].update({c: 0 for c in default_cats})
    for row in rows:
        strat = row.get("strategy")
        if strat not in strategies:
            continue
        tok = row.get("split_token_text", "")
        cat = classify_token(tok)
        counts[strat][cat] += 1
    return counts



def plot_bars(model: str, counts: Dict[str, Counter], out_path: Path):
    categories = sorted({c for counter in counts.values() for c in counter.keys()})
    # 将 Other 放到最后
    if "Other" in categories:
        categories = [c for c in categories if c != "Other"] + ["Other"]
    if not categories:
        print(f"[warn] no categories for model {model}")
        return

    strategies = list(counts.keys())
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(strategies))

    plt.figure(figsize=(max(6, len(categories) * 0.7), 4.5))
    for i, strat in enumerate(strategies):
        vals = []
        for cat in categories:
            v = counts[strat].get(cat, 0)
            vals.append(v)
        plt.bar(x + i * width, vals, width=width, label=strat)

    plt.xticks(x + width * (len(strategies) - 1) / 2, categories, rotation=15)
    plt.ylabel("Count")
    plt.title(f"{model}: uncertainty-token categories")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved: {out_path}")


def save_counts_json(all_counts: Dict[str, Dict[str, Counter]], out_path: Path):
    serializable = defaultdict(dict)
    for model, strat_counts in all_counts.items():
        for strat, counter in strat_counts.items():
            serializable[model][strat] = dict(counter)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"saved counts json: {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_paths: List[Tuple[str, Path]] = []
    for item in args.model_csv:
        if ":" not in item:
            raise ValueError(f"--model-csv requires name:path format, got: {item}")
        name, path_str = item.split(":", 1)
        model_paths.append((name, Path(path_str)))

    all_counts: Dict[str, Dict[str, Counter]] = {}

    for name, path in model_paths:
        if not path.exists():
            print(f"[warn] skip model {name}, csv not found: {path}")
            continue
        rows = load_csv(path)
        counts = build_counts(rows, args.strategies)
        all_counts[name] = counts
        plot_bars(
            model=name,
            counts=counts,
            out_path=out_dir / f"{name}_ast_groups.png",
        )

    if all_counts:
        save_counts_json(all_counts, out_dir / "ast_group_counts.json")


if __name__ == "__main__":
    main()
