import argparse
from pathlib import Path
import sys
import pandas as pd
from tree_sitter_languages import get_parser
from transformers import AutoTokenizer
from data.human_eval.human_eval.data import read_problems


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
    """DFS寻找包含byte_offset的最小node"""
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


def main():
    ap = argparse.ArgumentParser(description="Token node category analysis via tree-sitter")
    ap.add_argument("--run_dir", required=True, help="Run directory containing *_split_positions.csv")
    ap.add_argument("--model", required=True, help="Model key (e.g., qwen3-0.6b) to align tokenizer with splits")
    ap.add_argument("--output_csv", required=True, help="Path to save node category CSV")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    split_csv = run_dir / f"{args.model}_split_positions.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"{split_csv} not found")

    splits = pd.read_csv(split_csv)
    problems = read_problems()

    tokenizer = AutoTokenizer.from_pretrained(qwen_hf_id(args.model), trust_remote_code=True)
    parser = get_parser("python")

    rows = []
    for _, row in splits.iterrows():
        task_id = row["task_id"]
        split_idx = int(row["split_token_index"])
        code = problems[task_id]["prompt"] + problems[task_id]["canonical_solution"]
        token_ids = tokenizer(code, return_tensors="pt").input_ids[0]
        prefix = tokenizer.decode(token_ids[: split_idx + 1], skip_special_tokens=True)
        byte_offset = len(prefix.encode("utf-8"))

        tree = parser.parse(code.encode("utf-8"))
        node = find_smallest_node(tree.root_node, byte_offset)
        cat = node_category(node.type)

        rows.append(
            {
                "task_id": task_id,
                "strategy": row["strategy"],
                "split_token_index": split_idx,
                "split_token_text": row.get("split_token_text"),
                "node_type": node.type,
                "category": cat,
            }
        )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved node categories to {out_path}")


if __name__ == "__main__":
    main()
