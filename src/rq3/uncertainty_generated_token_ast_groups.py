

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate AST macro-category distribution for generated tokens (by model/strategy)")
    ap.add_argument(
        "--log",
        action="append",
        required=True,
        help="Path to experiment.log files (can provide multiple)",
    )
    ap.add_argument(
        "--strategies",
        nargs="+",
        default=["entropy", "confidence", "ppl"],
        help="Strategy names to include (must match 'Strategy:' lines in logs)",
    )
    ap.add_argument(
        "--output-dir",
        default="experiments/rq6/generated_token_ast_groups",
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

PROGRAM_STRUCTURE = {"def", "class", "import", "from", "as", "@", "async"}
LITERAL_CONST = {"True", "False", "None"}
NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def classify_token(tok: str) -> str:
    if tok is None:
        return "Other"

    raw = tok.replace("Ġ", " ").strip()

    if raw == "" or "\n" in raw:
        return "Other"

    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        return "Literal"
    if raw.startswith("'''") or raw.startswith('"""'):
        return "Literal"
    if raw in LITERAL_CONST:
        return "Literal"
    if NUMBER_RE.match(raw):
        return "Literal"

    if raw in PROGRAM_STRUCTURE:
        return "Program Structure"
    if raw in CONTROL_FLOW:
        return "Control Flow"
    if raw in OPERATORS:
        return "Operator"
    if IDENT_RE.match(raw):
        return "Identifier"

    return "Expression"


def init_counts(strategies: List[str]) -> Dict[str, Counter]:
    cats = ["Control Flow", "Operator", "Literal", "Identifier", "Program Structure", "Expression", "Other"]
    counts: Dict[str, Counter] = {s: Counter({c: 0 for c in cats}) for s in strategies}
    return counts


def parse_log(path: Path, strategies: List[str]) -> Dict[str, Dict[str, Counter]]:
    model_counts: Dict[str, Dict[str, Counter]] = {}
    current_model = None
    current_strategy = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "Running experiments with model:" in line:
                parts = line.split("model:")
                if len(parts) > 1:
                    current_model = parts[1].strip()
                    model_counts.setdefault(current_model, init_counts(strategies))
                continue

            if "Strategy:" in line:
                current_strategy = line.split("Strategy:")[1].strip()
                continue

            if "Predicted tokens:" in line and current_strategy in strategies and current_model:
                tokens_part = line.split("Predicted tokens:")[1].strip()
                try:
                    toks = ast.literal_eval(tokens_part)
                except Exception:
                    continue
                for tok in toks:
                    cat = classify_token(tok)
                    model_counts[current_model][current_strategy][cat] += 1

    return model_counts


def plot_bars(model: str, counts: Dict[str, Counter], out_path: Path):
    categories = sorted({c for counter in counts.values() for c in counter.keys()})
    if "Other" in categories:
        categories = [c for c in categories if c != "Other"] + ["Other"]

    strategies = list(counts.keys())
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(strategies))

    plt.figure(figsize=(max(6, len(categories) * 0.7), 4.5))
    for i, strat in enumerate(strategies):
        vals = [counts[strat].get(cat, 0) for cat in categories]
        plt.bar(x + i * width, vals, width=width, label=strat)

    plt.xticks(x + width * (len(strategies) - 1) / 2, categories, rotation=15)
    plt.ylabel("Count")
    plt.title(f"{model}: generated-token categories")
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

    all_counts: Dict[str, Dict[str, Counter]] = {}

    for log_path_str in args.log:
        log_path = Path(log_path_str)
        if not log_path.exists():
            print(f"[warn] skip missing log: {log_path}")
            continue
        model_counts = parse_log(log_path, args.strategies)
        for model_name, counts in model_counts.items():
            all_counts[model_name] = counts
            plot_bars(model_name, counts, out_dir / f"{model_name}_generated_ast_groups.png")

    if all_counts:
        save_counts_json(all_counts, out_dir / "generated_ast_group_counts.json")


if __name__ == "__main__":
    main()
