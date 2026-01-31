import argparse
import datetime
from pathlib import Path
import sys

import pandas as pd
from transformers import AutoTokenizer
try:
    from tree_sitter import Parser
    from tree_sitter_languages import get_language, get_parser 
except Exception as e:
    Parser = None
    get_language = None
    get_parser = None
    print("Warning: tree_sitter import failed, AST categorization will error without it:", e)

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
# Compatibility for human_eval reading
try:
    from human_eval.human_eval.data import read_problems  # type: ignore
except ImportError:
    import importlib.util

    data_py = ROOT / "data" / "human_eval" / "human_eval" / "data.py"
    spec = importlib.util.spec_from_file_location("human_eval_data", data_py)
    human_eval_data = importlib.util.module_from_spec(spec) if spec else None
    if not spec or not spec.loader:
        raise
    spec.loader.exec_module(human_eval_data)  # type: ignore
    read_problems = human_eval_data.read_problems  # type: ignore


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


def write_token_categories(run_dir: Path, model_key: str, out_csv: Path):
    split_csv = run_dir / f"{model_key}_split_positions.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"{split_csv} not found")
    if Parser is None and get_parser is None:
        raise ImportError("tree_sitter or tree_sitter_languages not available. Install tree_sitter_languages.")
    splits = pd.read_csv(split_csv)
    splits = splits.dropna(subset=["split_token_index"])
    if splits.empty:
        raise ValueError(f"No valid split_token_index found in {split_csv}")
    problems = read_problems()
    tokenizer = AutoTokenizer.from_pretrained(qwen_hf_id(model_key), trust_remote_code=True)
    parser = None
    parser_error = None
    # Prefer built-in parser factory if available (avoids older get_language API issues)
    if get_parser is not None:
        try:
            parser = get_parser("python")
        except Exception as e:
            parser_error = e
            parser = None
    if parser is None and Parser is not None and get_language is not None:
        try:
            lang = get_language("python")
            parser = Parser()
            parser.set_language(lang)
            parser_error = None
        except Exception as e:
            parser_error = e

    rows = []
    for _, row in splits.iterrows():
        task_id = row["task_id"]
        split_idx = int(row["split_token_index"])
        code = problems[task_id]["prompt"] + problems[task_id]["canonical_solution"]
        token_ids = tokenizer(code, return_tensors="pt").input_ids[0]
        prefix = tokenizer.decode(token_ids[: split_idx + 1], skip_special_tokens=True)
        byte_offset = len(prefix.encode("utf-8"))
        if parser is not None:
            try:
                tree = parser.parse(code.encode("utf-8"))
                node = find_smallest_node(tree.root_node, byte_offset)
                node_type = node.type
                cat = node_category(node_type)
            except Exception as e:
                node_type = f"parser_error:{e}"
                cat = "parser_error"
        else:
            node_type = f"parser_unavailable:{parser_error}"
            cat = "parser_error"

        rows.append(
            {
                "task_id": task_id,
                "strategy": row["strategy"],
                "split_token_index": split_idx,
                "split_token_text": row.get("split_token_text"),
                "node_type": node_type,
                "category": cat,
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved token categories to {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Run RQ2 AST category pipeline")
    ap.add_argument("--run_dir", required=True, help="Directory containing *_split_positions.csv")
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model keys (e.g., qwen3-0.6b qwen3-1.7b qwen3-4b)",
    )
    ap.add_argument(
        "--output_root",
        default="experiments/rq2_results",
        help="Root output dir (timestamp will be appended if exists)",
    )
    args = ap.parse_args()

    out_root = Path(args.output_root)
    if out_root.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = out_root.parent / f"{out_root.name}_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run_dir)
    for model_key in args.models:
        out_csv = out_root / f"token_node_categories_{model_key}.csv"
        print(f"Processing model {model_key} -> {out_csv}")
        write_token_categories(run_dir, model_key, out_csv)


if __name__ == "__main__":
    main()
