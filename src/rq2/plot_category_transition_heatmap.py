#!/usr/bin/env python3
"""
绘制切点前后 AST 大类的转移热力图：
  行：切割处所在 token 的 AST 大类
  列：切割后下一个 token 的 AST 大类
  值：计数

用法示例：
  python3 src/rq4/plot_category_transition_heatmap.py \
    --run_dir experiments/rq4_results_20251128_010316 \
    --model qwen3-0.6b \
    --output_dir experiments/rq4_results_20251128_010316 \
    --drop_parser_error

依赖：seaborn、tree_sitter_languages
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer

import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[2]
# 确保项目根、src、data 都在 sys.path 中
for p in [ROOT, ROOT / "src", ROOT / "data"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
from data.human_eval.human_eval.data import read_problems  # noqa: E402

try:
    from tree_sitter import Parser  # type: ignore
    from tree_sitter_languages import get_language, get_parser  # type: ignore
except Exception as e:
    Parser = None
    get_language = None
    get_parser = None
    print("Warning: tree_sitter import failed, AST categorization will error without it:", e)

# 与 run_rq4_pipeline 一致的分类映射
CATEGORY_MAP = {
    "control": {
        "if_statement",
        "elif_clause",
        "else_clause",
        "for_statement",
        "while_statement",
        "try_statement",
        "except_clause",
        "finally_clause",
        "match_statement",
        "case_clause",
        "return_statement",
        "break_statement",
        "continue_statement",
        "pass_statement",
    },
    "definition": {
        "function_definition",
        "class_definition",
        "parameters",
        "lambda",
        "assignment",
        "augmented_assignment",
        "typed_parameter",
        "type_annotation",
    },
    "call": {
        "call",
        "argument_list",
        "keyword_argument",
        "generator_expression",
        "list_comprehension",
        "set_comprehension",
        "dictionary_comprehension",
        "attribute",
        "subscript",
    },
    "expression": {
        "binary_operator",
        "unary_operator",
        "comparison_operator",
        "boolean_operator",
        "arithmetic_operator",
        "not_operator",
        "await",
        "yield",
    },
    "literal": {
        "identifier",
        "integer",
        "float",
        "string",
        "true",
        "false",
        "none",
        "list",
        "dictionary",
        "set",
        "tuple",
        "list_splat",
        "dictionary_splat",
    },
    "import": {
        "import_statement",
        "import_from_statement",
        "aliased_import",
        "dotted_name",
    },
    "comment": {"comment"},
}


def node_category(node_type: str) -> str:
    for cat, types in CATEGORY_MAP.items():
        if node_type in types:
            return cat
    return "other"


def find_smallest_node(node, byte_offset):
    for child in node.children:
        if child.start_byte <= byte_offset < child.end_byte:
            return find_smallest_node(child, byte_offset)
    return node


def qwen_hf_id(model_key: str) -> str:
    mapping = {
        "qwen3-0.6b": "Qwen/Qwen3-0.6B",
        "qwen3-1.7b": "Qwen/Qwen3-1.7B",
        "qwen3-4b": "Qwen/Qwen3-4B",
        "qwen3-8b": "Qwen/Qwen3-8B",
    }
    return mapping.get(model_key, model_key)


def build_parser():
    if get_parser is not None:
        try:
            return get_parser("python")
        except Exception:
            pass
    if Parser is not None and get_language is not None:
        lang = get_language("python")
        parser = Parser()
        parser.set_language(lang)
        return parser
    raise ImportError("tree_sitter or tree_sitter_languages not available.")


def category_at_index(code: str, tokens, idx: int, parser) -> str:
    prefix = tokens[: idx + 1]
    decoded = tokenizer.decode(prefix, skip_special_tokens=True)
    byte_offset = len(decoded.encode("utf-8"))
    try:
        tree = parser.parse(code.encode("utf-8"))
        node = find_smallest_node(tree.root_node, byte_offset)
        return node_category(node.type)
    except Exception:
        return "parser_error"


def plot_heatmap(df_pairs: pd.DataFrame, model_name: str, out_path: Path):
    table = (
        df_pairs.groupby(["cat_curr", "cat_next"])
        .size()
        .reset_index(name="count")
        .pivot(index="cat_curr", columns="cat_next", values="count")
        .fillna(0)
        .sort_index()
    )
    plt.figure(figsize=(8, max(4, 0.4 * len(table.index))))
    ax = sns.heatmap(table, annot=True, fmt=".0f", cmap="Blues")
    ax.set_title(f"Split token category transitions ({model_name})")
    ax.set_xlabel("Next token category")
    ax.set_ylabel("Split token category")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved heatmap -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot heatmap of category transitions at split point.")
    ap.add_argument("--run_dir", required=True, help="Directory containing *_split_positions.csv")
    ap.add_argument("--model", required=True, help="Model key, e.g., qwen3-0.6b")
    ap.add_argument("--output_dir", default="experiments/rq4_results", help="Where to save figures")
    ap.add_argument("--drop_parser_error", action="store_true", help="Drop parser_error pairs")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    split_csv = run_dir / f"{args.model}_split_positions.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"{split_csv} not found")

    splits = pd.read_csv(split_csv).dropna(subset=["split_token_index"])
    problems = read_problems()
    global tokenizer  # used in category_at_index
    tokenizer = AutoTokenizer.from_pretrained(qwen_hf_id(args.model), trust_remote_code=True)
    parser = build_parser()

    pairs = []
    for _, row in splits.iterrows():
        split_idx = int(row["split_token_index"])
        task_id = row["task_id"]
        code = problems[task_id]["prompt"] + problems[task_id]["canonical_solution"]
        token_ids = tokenizer(code, return_tensors="pt").input_ids[0]
        if split_idx >= len(token_ids) - 1:
            continue  # no next token
        cat_curr = category_at_index(code, token_ids, split_idx, parser)
        cat_next = category_at_index(code, token_ids, split_idx + 1, parser)
        pairs.append({"cat_curr": cat_curr, "cat_next": cat_next})

    df_pairs = pd.DataFrame(pairs)
    if args.drop_parser_error:
        df_pairs = df_pairs[
            (df_pairs["cat_curr"] != "parser_error") & (df_pairs["cat_next"] != "parser_error")
        ]
    model_name = args.model
    out_path = Path(args.output_dir) / f"category_transition_{model_name}.png"
    plot_heatmap(df_pairs, model_name, out_path)


if __name__ == "__main__":
    main()
