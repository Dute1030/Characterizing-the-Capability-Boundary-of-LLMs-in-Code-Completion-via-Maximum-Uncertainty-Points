import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Attempt to import tree-sitter
try:
    from tree_sitter import Language, Parser
    HAS_TS = True
except Exception:
    HAS_TS = False

# Attempt to import transformers
try:
    from transformers import AutoTokenizer
    HAS_HF = True
except Exception:
    HAS_HF = False

# HumanEval loader
try:
    from human_eval.human_eval.data import read_problems  # type: ignore
except ImportError:
    import importlib.util
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    data_py = ROOT / "data" / "human_eval" / "human_eval" / "data.py"
    spec = importlib.util.spec_from_file_location("human_eval_data", data_py)
    human_eval_data = importlib.util.module_from_spec(spec) if spec else None
    if spec and spec.loader:
        spec.loader.exec_module(human_eval_data)  # type: ignore
        read_problems = human_eval_data.read_problems  # type: ignore
    else:
        raise


STRUCT_MAP = {
    # Semantic Classes
    "identifier": "Identifier",
    "call": "MethodCall",
    "string": "Literal",
    "string_literal": "Literal",
    "integer": "Literal",
    "float": "Literal",
    "number": "Literal",
    "attribute": "Identifier",
    "argument_list": "Identifier",
    # Syntactic Classes
    "operator": "Operator",
    "comparison_operator": "Operator",
    "boolean_operator": "Operator",
    "assignment": "Operator",
    "if": "Control",
    "elif_clause": "Control",
    "else_clause": "Control",
    "for": "Control",
    "while": "Control",
    "return": "Control",
    "break": "Control",
    "continue": "Control",
    "try": "Control",
    "with": "Control",
    "separator": "Separator",
    "expression_list": "Separator",
}


FAILURE_TAGS = ["Hallucination", "Repetition", "Semantic", "Other"]


def build_parser() -> Parser:
    """
    Build Python parser (expandable for other languages).
    """
    py_so = Path("/tmp/tree-sitter-python.so")
    if not py_so.exists():
        logging.warning("Tree-sitter Python parser not found; AST tagging will be skipped.")
        raise RuntimeError("Parser binary missing")
    lang = Language(str(py_so), "python")
    parser = Parser()
    parser.set_language(lang)
    return parser


def locate_node(parser: Parser, code: str, start: int, end: int) -> str:
    """
    Return the type of the smallest AST node covering the byte range [start, end).
    """
    tree = parser.parse(code.encode("utf8"))
    node = tree.root_node
    target = None
    stack = [node]
    while stack:
        n = stack.pop()
        if n.start_byte <= start and n.end_byte >= end:
            target = n
            stack.extend(n.children)
    return target.type if target else "unknown"


def struct_class(node_type: str) -> str:
    """Map raw Tree-sitter node types to categorized structural labels."""
    return STRUCT_MAP.get(node_type, "Unknown")


def assign_failure(pred: str, ref: str, context: str, em: float) -> str:
    """Heuristic rule to categorize the type of prediction failure."""
    if em >= 1.0:
        return "Correct"
    if pred and pred not in context and pred.strip() != ref.strip():
        return "Hallucination"
    
    # Simple repetition detection
    tokens = pred.strip().split()
    if len(tokens) >= 2 and len(set(tokens)) <= len(tokens) // 2:
        return "Repetition"
    return "Other"


def load_samples(path: Path) -> List[Dict]:
    out = []
    with path.open() as f:
        for line in f:
            out.append(json.loads(line))
    return out


def offsets_for_token(tokenizer, code: str):
    """Retrieve token offsets from the tokenizer."""
    enc = tokenizer(code, return_offsets_mapping=True, add_special_tokens=False)
    return enc["offset_mapping"]


def main():
    ap = argparse.ArgumentParser(description="RQ6: Analysis of Code Structure and Failure Modes")
    ap.add_argument("--input", required=True, help="Input JSONL samples (must contain task_id, token_idx, prediction, reference, metrics)")
    ap.add_argument("--model-name", required=True, help="Model name identifier for output files")
    ap.add_argument("--tokenizer", required=True, help="Tokenizer name or path for offset mapping")
    ap.add_argument("--output-dir", required=True, help="Directory to save analysis results")
    ap.add_argument("--max-samples", type=int, default=None, help="Limit the number of samples processed")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")

    problems = read_problems()
    samples = load_samples(Path(args.input))
    if args.max_samples:
        samples = samples[: args.max_samples]
    logging.info(f"Loaded {len(samples)} samples.")

    if HAS_HF:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    else:
        tokenizer = None
        logging.warning("Transformers library not available; offsets will be approximated by whitespace splitting.")

    parser = None
    if HAS_TS:
        try:
            parser = build_parser()
        except Exception:
            parser = None
    if parser is None:
        logging.warning("Tree-sitter unavailable; structural category will default to 'Unknown'.")

    # Aggregate statistics
    struct_counts: Dict[str, int] = {}
    heatmap: Dict[Tuple[str, str], int] = {}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"{args.model_name}_structure_failure.jsonl"

    with detail_path.open("w") as fw:
        for s in samples:
            task_id = s.get("task_id")
            if task_id not in problems:
                continue
            
            # Combine prompt and solution to reconstruct original code context
            code = problems[task_id]["prompt"] + problems[task_id]["canonical_solution"]
            token_idx = s.get("token_idx", 0)
            prediction = s.get("prediction", "")
            reference = s.get("reference", "")
            em = float(s.get("metrics", {}).get("exact_match", 0.0))

            # Resolve Offsets
            if tokenizer:
                offsets = offsets_for_token(tokenizer, code)
                if token_idx < len(offsets):
                    start, end = offsets[token_idx]
                else:
                    start, end = 0, 0
            else:
                # Fallback: estimate position via whitespace tokens
                tokens = code.split()
                pos = min(token_idx, len(tokens) - 1)
                start = code.find(tokens[pos])
                end = start + len(tokens[pos])

            # Determine AST Structure
            node_type = "unknown"
            if parser and start is not None:
                node_type = locate_node(parser, code, start, end)
            struct = struct_class(node_type)
            struct_counts[struct] = struct_counts.get(struct, 0) + 1

            # Determine Failure Mode
            failure = assign_failure(prediction, reference, code, em)
            if failure != "Correct":
                heatmap[(struct, failure)] = heatmap.get((struct, failure), 0) + 1

            record = {
                "task_id": task_id,
                "token_idx": token_idx,
                "structure": struct,
                "node_type": node_type,
                "failure": failure,
            }
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Save aggregated results
    agg_path = out_dir / f"{args.model_name}_structure_failure_agg.json"
    agg = {
        "structure_counts": struct_counts,
        "failure_heatmap": {f"{k[0]}|{k[1]}": v for k, v in heatmap.items()},
    }
    agg_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2))
    
    logging.info(f"Analysis complete. Details saved to: {detail_path}")
    logging.info(f"Aggregation saved to: {agg_path}")


if __name__ == "__main__":
    main()