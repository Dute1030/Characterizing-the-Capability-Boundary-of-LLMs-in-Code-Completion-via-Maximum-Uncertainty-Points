"""
Rename the MUP token to a normalized identifier and measure completion performance change.

Workflow:
1) Read split_positions.csv (e.g., CodeGPT entropy peaks) to get task_id and split_token_index.
2) For each task:
   - Build prefix/suffix at the MUP token using the split tokenizer (default: CodeGPT).
   - Create a renamed version where the MUP token is replaced by a normalized identifier.
3) For each target model (Qwen3 family), run line-level completion on both original and renamed prefixes,
   evaluate against the corresponding suffix, and report metrics + deltas.

Usage example:
python -m src.rq6.rename_mup_identifier \
  --split-csv experiments/rq1_codegpt_guided_full/codegpt_guided_20251209_025226/codegpt-small-py_split_positions.csv \
  --models qwen3-0.6b qwen3-1.7b qwen3-4b \
  --replacement norm_var \
  --split-tokenizer microsoft/CodeGPT-small-py \
  --output-dir experiments/rq6/rename_mup
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.models import MODEL_FACTORY  # noqa: E402
from src.rq1.line_completion import LineCompletion  # noqa: E402
from src.rq1.metrics import MetricsCalculator  # noqa: E402
from data.human_eval.human_eval.data import read_problems  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rename MUP identifier and measure completion performance change.")
    ap.add_argument("--split-csv", required=True, help="split_positions.csv with task_id and split_token_index")
    ap.add_argument("--models", nargs="+", required=True, help="Target models (e.g., qwen3-0.6b qwen3-1.7b)")
    ap.add_argument("--replacement", default="norm_var", help="Normalized identifier to replace the MUP token")
    ap.add_argument("--split-tokenizer", default="microsoft/CodeGPT-small-py", help="Tokenizer used to interpret split_token_index")
    ap.add_argument("--strategy", default=None, help="Optional strategy filter (e.g., codegpt_entropy)")
    ap.add_argument("--device", default="cuda", help="Device for target models")
    ap.add_argument("--output-dir", default="experiments/rq6/rename_mup", help="Output directory")
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    return ap.parse_args()


def load_split_rows(path: Path, strategy: str = None) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if strategy and row.get("strategy") != strategy:
                continue
            rows.append(row)
    return rows


def build_variants(
    code: str,
    token_idx: int,
    tokenizer,
    replacement: str,
) -> Tuple[Dict, Dict]:
    """
    Build original and renamed prefix/suffix pairs using the split tokenizer.
    token_idx corresponds to entropies index => token_ids[token_idx + 1].
    """
    ids = tokenizer(code, return_tensors="pt").input_ids[0].tolist()
    pos = token_idx + 1
    if pos >= len(ids):
        return {}, {}

    target_token_id = ids[pos]
    target_text = tokenizer.decode([target_token_id], skip_special_tokens=True)

    # replacement ids
    repl_ids = tokenizer.encode(replacement, add_special_tokens=False)
    new_ids = ids[:pos] + repl_ids + ids[pos + 1 :]

    # Original prefix/suffix (include the MUP token in prefix)
    orig_prefix_ids = ids[: pos + 1]
    orig_suffix_ids = ids[pos + 1 :]
    orig = {
        "prefix": tokenizer.decode(orig_prefix_ids, skip_special_tokens=True),
        "suffix": tokenizer.decode(orig_suffix_ids, skip_special_tokens=True),
        "token_text": target_text,
    }

    # Renamed prefix/suffix (include full replacement in prefix)
    new_prefix_ids = new_ids[: pos + len(repl_ids)]
    new_suffix_ids = new_ids[pos + len(repl_ids) :]
    renamed = {
        "prefix": tokenizer.decode(new_prefix_ids, skip_special_tokens=True),
        "suffix": tokenizer.decode(new_suffix_ids, skip_special_tokens=True),
        "token_text": replacement,
    }
    return orig, renamed


def evaluate_variant(
    lc: LineCompletion,
    mc: MetricsCalculator,
    prefix: str,
    suffix_ref: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict:
    completion = lc.complete_line(
        prefix,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_at_newline=True,
    )
    # Compare only first line to align with RQ1 evaluation style
    def trim_first_line(text: str) -> str:
        pos = text.find("\n")
        return text[:pos] if pos != -1 else text

    pred = trim_first_line(completion)
    ref = trim_first_line(suffix_ref)
    metrics = mc.evaluate_completion(prediction=pred, reference=ref, lang="python")
    metrics["completion"] = pred
    return metrics


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_rows = load_split_rows(Path(args.split_csv), strategy=args.strategy)
    problems = read_problems()

    # Tokenizer for interpreting split_token_index
    split_tokenizer = AutoTokenizer.from_pretrained(args.split_tokenizer, trust_remote_code=True)
    if split_tokenizer.pad_token_id is None and split_tokenizer.eos_token_id is not None:
        split_tokenizer.pad_token_id = split_tokenizer.eos_token_id

    mc = MetricsCalculator()

    summary = {}

    for model_name in args.models:
        model, tokenizer = MODEL_FACTORY[model_name](device=args.device)
        model.to(args.device)
        model.eval()
        lc = LineCompletion(model, tokenizer, args.device)

        sample_rows: List[Dict] = []
        orig_metrics_all: List[Dict] = []
        rename_metrics_all: List[Dict] = []

        for row in split_rows:
            task_id = row["task_id"]
            if task_id not in problems:
                continue
            code = problems[task_id]["prompt"] + problems[task_id]["canonical_solution"]
            try:
                token_idx = int(row["split_token_index"])
            except Exception:
                continue

            orig, renamed = build_variants(code, token_idx, split_tokenizer, args.replacement)
            if not orig or not renamed:
                continue

            orig_metrics = evaluate_variant(
                lc,
                mc,
                orig["prefix"],
                orig["suffix"],
                args.max_new_tokens,
                args.temperature,
                args.top_p,
            )
            rename_metrics = evaluate_variant(
                lc,
                mc,
                renamed["prefix"],
                renamed["suffix"],
                args.max_new_tokens,
                args.temperature,
                args.top_p,
            )

            orig_metrics_all.append(orig_metrics)
            rename_metrics_all.append(rename_metrics)

            sample_rows.append(
                {
                    "task_id": task_id,
                    "model": model_name,
                    "orig_token": orig["token_text"],
                    "replacement": args.replacement,
                    "orig_prefix": orig["prefix"],
                    "orig_suffix": orig["suffix"],
                    "rename_prefix": renamed["prefix"],
                    "rename_suffix": renamed["suffix"],
                    "orig_completion": orig_metrics.get("completion", ""),
                    "rename_completion": rename_metrics.get("completion", ""),
                    **{f"orig_{k}": v for k, v in orig_metrics.items() if isinstance(v, float)},
                    **{f"rename_{k}": v for k, v in rename_metrics.items() if isinstance(v, float)},
                }
            )

        # Aggregate
        agg_orig = mc.aggregate_results([ {k:v for k,v in m.items() if isinstance(v,float)} for m in orig_metrics_all ])
        agg_rename = mc.aggregate_results([ {k:v for k,v in m.items() if isinstance(v,float)} for m in rename_metrics_all ])
        delta = {}
        for metric, vals in agg_orig.items():
            if metric in agg_rename and "mean" in vals and "mean" in agg_rename[metric]:
                delta[metric] = agg_rename[metric]["mean"] - vals["mean"]

        summary[model_name] = {
            "original": agg_orig,
            "renamed": agg_rename,
            "delta": delta,
        }

        # Save per-model CSV
        import pandas as pd

        df = pd.DataFrame(sample_rows)
        df_path = out_dir / f"{model_name}_rename_mup_samples.csv"
        df.to_csv(df_path, index=False, encoding="utf-8")
        print(f"saved samples: {df_path}")

        # Cleanup GPU
        del model, tokenizer, lc
        torch.cuda.empty_cache()

    # Save summary
    summary_path = out_dir / "rename_mup_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
