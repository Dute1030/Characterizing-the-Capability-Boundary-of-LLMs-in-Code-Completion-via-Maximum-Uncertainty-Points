"""
Use CodeGPT as a weak coder to pick a unified maximum-uncertainty split, then
evaluate multiple models (Qwen3 + DeepSeek) at that shared cut point.

Usage example:
    python -m src.rq1.run_codegpt_guided \\
        --metric entropy \\
        --codegpt-model codegpt-small-py \\
        --dataset py150 \\
        --py150_path data/py150 \\
        --models qwen3-0.6b qwen3-1.7b qwen3-4b deepseek-1.3b deepseek-6.7b \\
        --num-samples 200
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import csv
import json
import logging
import sys
import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
import warnings

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.models import MODEL_FACTORY  # noqa: E402
from src.rq1.uncertainty_calculator import UncertaintyCalculator  # noqa: E402
from src.rq1.code_splitter import CodeSplitter  # noqa: E402
from src.rq1.line_completion import LineCompletion  # noqa: E402
from src.rq1.metrics import MetricsCalculator  # noqa: E402
from src.rq1.datasets import load_rq1_dataset  # noqa: E402


def truncate_code_if_needed(code: str, tokenizer, max_len: int) -> Tuple[str, bool]:
    """
    GPT-style models (e.g., CodeGPT) have limited position embeddings (e.g., 1024).
    If tokenized length exceeds max_len, truncate to avoid CUDA gather OOB.
    """
    token_ids = tokenizer(code, return_tensors="pt").input_ids[0]
    if len(token_ids) <= max_len:
        return code, False
    truncated_ids = token_ids[:max_len]
    truncated_code = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    return truncated_code, True


def setup_logging(log_file: str = None) -> None:
    log_format = "[%(levelname)s] %(asctime)s - %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CodeGPT-guided unified MUP evaluation for multiple models")
    ap.add_argument("--dataset", default="humaneval", choices=["humaneval", "py150"], help="Dataset to use")
    ap.add_argument("--py150-path", default="data/py150", help="Py150 path (.py dir or json/jsonl file)")
    ap.add_argument("--py150-min-chars", type=int, default=80, help="Minimum total chars (prompt+solution) for Py150 samples")
    ap.add_argument("--py150-max-chars", type=int, default=800, help="Maximum total chars (prompt+solution) for Py150 samples")
    ap.add_argument("--py150-shuffle", dest="py150_shuffle", action="store_true", default=True, help="Shuffle Py150 candidates before sampling (default: True)")
    ap.add_argument("--no-py150-shuffle", dest="py150_shuffle", action="store_false", help="Disable shuffling for Py150 sampling")
    ap.add_argument("--py150-seed", type=int, default=42, help="Random seed for Py150 sampling/shuffling")
    ap.add_argument("--num-samples", type=int, default=None, help="Subset of samples (default: all)")
    ap.add_argument("--metric", default="entropy", choices=["entropy", "perplexity", "confidence"], help="Uncertainty metric for CodeGPT split")
    ap.add_argument("--codegpt-model", default="codegpt-small-py", help="Model key for CodeGPT in MODEL_FACTORY")
    ap.add_argument("--models", nargs="+", default=["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "deepseek-1.3b", "deepseek-6.7b"], help="Target models to evaluate at the shared cut point")
    ap.add_argument("--device", default="cuda", help="Device for all models")
    ap.add_argument("--output-dir", default="experiments/rq1_codegpt_guided", help="Base output directory")
    ap.add_argument("--run-name", default=None, help="Optional subfolder name; defaults to timestamped run id")
    ap.add_argument("--log-file", default=None, help="Optional log file path")
    ap.add_argument("--max-new-tokens", type=int, default=50, help="Max new tokens for line-level completion")
    ap.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for completion")
    ap.add_argument("--top-p", type=float, default=0.95, help="Top-p for completion sampling")
    ap.add_argument("--skip-edges", type=int, default=1, help="Skip edge tokens when detecting CodeGPT uncertainty peaks")
    return ap.parse_args()


def trim_at_first_newline(text: str) -> str:
    if text is None:
        return ""
    pos = text.find("\n")
    return text[:pos] if pos != -1 else text


def generate_line_level_completion(
    prefix: str,
    line_completion: LineCompletion,
    max_total_tokens: int = 50,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    return line_completion.complete_line(
        prefix,
        max_new_tokens=max_total_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_at_newline=True,
    )


def compute_codegpt_splits(
    dataset: List[Tuple[str, Dict]],
    model_name: str,
    metric: str,
    device: str,
    skip_edges: int,
) -> Tuple[Dict[str, Dict], List[Dict]]:
    """Use CodeGPT to pick one MUP per task."""
    logging.info("Loading CodeGPT model: %s", model_name)
    model, tokenizer = MODEL_FACTORY[model_name](device=device)
    model.to(device)
    model.eval()

    uc = UncertaintyCalculator(model, tokenizer, device)
    splitter = CodeSplitter(uc)

    split_map: Dict[str, Dict] = {}
    split_rows: List[Dict] = []

    # Use model positional limit if available (GPT2/CodeGPT typically 1024)
    max_len = getattr(getattr(model, "config", None), "n_positions", None)
    if max_len is None:
        max_len = getattr(tokenizer, "model_max_length", 1024)
    # Guard against placeholder huge values
    if max_len and max_len > 100000:
        max_len = 1024

    for task_id, problem in tqdm(dataset, desc="Computing CodeGPT splits"):
        code = problem["prompt"] + problem["canonical_solution"]
        code, truncated = truncate_code_if_needed(code, tokenizer, max_len)
        if truncated:
            warnings.warn(f"Task {task_id} truncated to {max_len} tokens for CodeGPT to avoid OOB.")
        try:
            prefix, suffix, info = splitter.split_at_max_uncertainty_token(
                code, metric=metric, skip_edges=skip_edges
            )
        except Exception as e:
            logging.error("Failed to split %s: %s", task_id, e)
            continue

        split_map[task_id] = {"prefix": prefix, "suffix": suffix, "split_info": info}
        split_rows.append(
            {
                "task_id": task_id,
                "strategy": f"codegpt_{metric}",
                "split_token_index": info.get("split_token_index"),
                "split_token_text": info.get("split_token_text"),
                "uncertainty_metric": info.get("metric"),
                "uncertainty_value": info.get("uncertainty_value"),
                "prominence": info.get("prominence"),
            }
        )

    del model, tokenizer, uc, splitter
    torch.cuda.empty_cache()
    logging.info("Computed splits for %d tasks via CodeGPT", len(split_map))
    return split_map, split_rows


def compute_model_max_uncertainty_splits(
    dataset: List[Tuple[str, Dict]],
    model_name: str,
    metric: str,
    device: str,
    skip_edges: int,
) -> Tuple[Dict[str, Dict], List[Dict]]:
    """
    Let the target model pick its own max-uncertainty split (for before/after comparison).
    """
    logging.info("Loading target model for self-split: %s", model_name)
    model, tokenizer = MODEL_FACTORY[model_name](device=device)
    model.to(device)
    model.eval()

    uc = UncertaintyCalculator(model, tokenizer, device)
    splitter = CodeSplitter(uc)

    split_map: Dict[str, Dict] = {}
    split_rows: List[Dict] = []

    max_len = getattr(getattr(model, "config", None), "n_positions", None)
    if max_len is None:
        max_len = getattr(tokenizer, "model_max_length", 4096)
    if max_len and max_len > 100000:
        max_len = 4096

    for task_id, problem in tqdm(dataset, desc=f"Self splits ({model_name})"):
        code = problem["prompt"] + problem["canonical_solution"]
        code, truncated = truncate_code_if_needed(code, tokenizer, max_len)
        if truncated:
            warnings.warn(f"Task {task_id} truncated to {max_len} tokens for {model_name} to avoid OOB.")
        try:
            prefix, suffix, info = splitter.split_at_max_uncertainty_token(
                code, metric=metric, skip_edges=skip_edges
            )
        except Exception as e:
            logging.error("Failed to split %s for %s: %s", task_id, model_name, e)
            continue

        split_map[task_id] = {"prefix": prefix, "suffix": suffix, "split_info": info}
        split_rows.append(
            {
                "task_id": task_id,
                "strategy": f"{model_name}_{metric}",
                "split_token_index": info.get("split_token_index"),
                "split_token_text": info.get("split_token_text"),
                "uncertainty_metric": info.get("metric"),
                "uncertainty_value": info.get("uncertainty_value"),
                "prominence": info.get("prominence"),
            }
        )

    del model, tokenizer, uc, splitter
    torch.cuda.empty_cache()
    logging.info("Computed self splits for %d tasks via %s", len(split_map), model_name)
    return split_map, split_rows


def evaluate_model_at_splits(
    model_name: str,
    splits: Dict[str, Dict],
    metrics_calculator: MetricsCalculator,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[Dict, List[Dict]]:
    logging.info("Evaluating model at unified splits: %s", model_name)
    model, tokenizer = MODEL_FACTORY[model_name](device=device)
    model.to(device)
    model.eval()

    lc = LineCompletion(model, tokenizer, device)

    per_sample_metrics: List[Dict] = []
    rows: List[Dict] = []

    for task_id, split in tqdm(splits.items(), desc=f"Evaluating {model_name}"):
        prefix = split["prefix"]
        suffix_full = split["suffix"]
        info = split.get("split_info", {})

        if not prefix.strip():
            logging.warning("Empty prefix for %s, skip", task_id)
            continue

        completion = generate_line_level_completion(
            prefix=prefix,
            line_completion=lc,
            max_total_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        ref = trim_at_first_newline(suffix_full)
        pred = trim_at_first_newline(completion)

        metrics = metrics_calculator.evaluate_completion(prediction=pred, reference=ref, lang="python")
        per_sample_metrics.append(metrics)

        rows.append(
            {
                "task_id": task_id,
                "prefix_len": len(prefix),
                "suffix_len": len(ref),
                "completion_len": len(pred),
                "split_token_index": info.get("split_token_index"),
                "split_token_text": info.get("split_token_text"),
                "uncertainty_metric": info.get("metric"),
                "uncertainty_value": info.get("uncertainty_value"),
                **metrics,
            }
        )

    aggregated = metrics_calculator.aggregate_results(per_sample_metrics)

    del model, tokenizer, lc
    torch.cuda.empty_cache()

    return aggregated, rows


def main():
    args = parse_args()

    dataset_kwargs = {
        "py150_path": args.py150_path,
        "py150_min_chars": args.py150_min_chars,
        "py150_max_chars": args.py150_max_chars,
        "py150_shuffle": args.py150_shuffle,
        "py150_seed": args.py150_seed,
    }

    run_suffix = args.run_name or datetime.datetime.now().strftime("codegpt_guided_%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = args.log_file or str(out_dir / "experiment.log")
    setup_logging(log_file)

    logging.info("Args: %s", vars(args))
    dataset = load_rq1_dataset(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        **dataset_kwargs,
    )
    logging.info("Loaded %d tasks from %s", len(dataset), args.dataset)
    if args.dataset == "py150":
        logging.info(
            "Py150 settings -> path=%s min_chars=%d max_chars=%s shuffle=%s seed=%d",
            args.py150_path,
            args.py150_min_chars,
            args.py150_max_chars if args.py150_max_chars is not None else "None",
            args.py150_shuffle,
            args.py150_seed,
        )

    splits, split_rows = compute_codegpt_splits(
        dataset=dataset,
        model_name=args.codegpt_model,
        metric=args.metric,
        device=args.device,
        skip_edges=args.skip_edges,
    )

    split_csv = out_dir / f"{args.codegpt_model}_split_positions.csv"
    with split_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "strategy",
                "split_token_index",
                "split_token_text",
                "uncertainty_metric",
                "uncertainty_value",
                "prominence",
            ],
        )
        writer.writeheader()
        writer.writerows(split_rows)
    logging.info("Saved CodeGPT split positions to %s", split_csv)

    metrics_calculator = MetricsCalculator()
    combined_results: Dict[str, Dict] = {}

    for model_name in args.models:
        # Evaluate at CodeGPT unified cut (after)
        aggregated_cg, rows_cg = evaluate_model_at_splits(
            model_name=model_name,
            splits=splits,
            metrics_calculator=metrics_calculator,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # Evaluate at the model's own max-uncertainty cut (before)
        self_splits, self_rows_meta = compute_model_max_uncertainty_splits(
            dataset=dataset,
            model_name=model_name,
            metric=args.metric,
            device=args.device,
            skip_edges=args.skip_edges,
        )
        aggregated_self, rows_self = evaluate_model_at_splits(
            model_name=model_name,
            splits=self_splits,
            metrics_calculator=metrics_calculator,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # Save per-model sample metrics
        if rows_cg:
            cg_detail = out_dir / f"{model_name}_details_codegpt.csv"
            with cg_detail.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows_cg[0].keys())
                writer.writeheader()
                writer.writerows(rows_cg)
            logging.info("Saved CodeGPT-cut sample metrics for %s to %s", model_name, cg_detail)

        if rows_self:
            self_detail = out_dir / f"{model_name}_details_self.csv"
            with self_detail.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows_self[0].keys())
                writer.writeheader()
                writer.writerows(rows_self)
            logging.info("Saved self-cut sample metrics for %s to %s", model_name, self_detail)

        cg_file = out_dir / f"{model_name}_results_codegpt.json"
        with cg_file.open("w", encoding="utf-8") as f:
            json.dump(aggregated_cg, f, indent=2)
        logging.info("Saved aggregated CodeGPT-cut metrics for %s to %s", model_name, cg_file)

        self_file = out_dir / f"{model_name}_results_self.json"
        with self_file.open("w", encoding="utf-8") as f:
            json.dump(aggregated_self, f, indent=2)
        logging.info("Saved aggregated self-cut metrics for %s to %s", model_name, self_file)

        # Delta summary (CodeGPT cut minus self cut)
        delta = {}
        for metric, stats in aggregated_cg.items():
            if metric in aggregated_self and "mean" in stats and "mean" in aggregated_self[metric]:
                delta[metric] = {
                    "codegpt_mean": stats["mean"],
                    "self_mean": aggregated_self[metric]["mean"],
                    "delta": stats["mean"] - aggregated_self[metric]["mean"],
                }
        comp_file = out_dir / f"{model_name}_comparison.json"
        with comp_file.open("w", encoding="utf-8") as f:
            json.dump({"codegpt": aggregated_cg, "self": aggregated_self, "delta": delta}, f, indent=2)
        logging.info("Saved before/after comparison for %s to %s", model_name, comp_file)

        combined_results[model_name] = {
            "codegpt_cut": aggregated_cg,
            "self_cut": aggregated_self,
            "delta": delta,
        }

    combined_file = out_dir / "all_results.json"
    with combined_file.open("w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)
    logging.info("Saved combined metrics to %s", combined_file)

    logging.info("Done. Results saved under %s", out_dir)


if __name__ == "__main__":
    main()
