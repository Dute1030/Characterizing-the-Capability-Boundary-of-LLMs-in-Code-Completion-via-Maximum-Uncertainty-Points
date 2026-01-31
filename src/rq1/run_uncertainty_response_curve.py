"""
RQ2: Performance–Uncertainty Response Curves

What it does
------------
1) Teacher-forcing all test problems to collect token-level uncertainties (entropy,
   confidence, perplexity).
2) Bin tokens by entropy (quantile bins) to cover the full spectrum of uncertainty.
3) Stratified sample the same number of cut points from each bin.
4) Run line-level completion from each sampled cut point and score against the
   ground-truth suffix (CodeBLEU / EM / ROUGE-L / BLEU).
5) Aggregate and save per-bin metrics so you can draw the “collapse threshold”
   curves (performance vs. uncertainty).

Notes
-----
- Reuses RQ1 components: MODEL_FACTORY, UncertaintyCalculator, LineCompletion,
  MetricsCalculator, and the HumanEval loader.
- By default uses entropy bins (quintiles). You can switch to PPL bins with
  --uncertainty-type ppl.
- Designed to be run once per model; for multiple models, pass them all via
  --models and it will iterate.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Ensure project root and data dirs are on sys.path
ROOT = Path(__file__).resolve().parents[2]  # /data/dt/AdaDec
for p in [ROOT, ROOT / "data", ROOT / "data" / "human_eval"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Project imports
from llm.models import MODEL_FACTORY
from src.rq1.uncertainty_calculator import UncertaintyCalculator
from src.rq1.line_completion import LineCompletion
from src.rq1.metrics import MetricsCalculator

# Robust import for HumanEval read_problems
try:
    from human_eval.human_eval.data import read_problems  # type: ignore
except ImportError:
    import importlib.util

    data_py = ROOT / "data" / "human_eval" / "human_eval" / "data.py"
    if not data_py.exists():
        raise
    spec = importlib.util.spec_from_file_location("human_eval_data", data_py)
    human_eval_data = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(human_eval_data)
    read_problems = human_eval_data.read_problems  # type: ignore


def setup_logging(log_file: Path = None):
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        handlers=handlers,
    )


def load_dataset(num_samples: int = None) -> List[Tuple[str, Dict]]:
    problems = [(task_id, problem) for task_id, problem in read_problems().items()]
    if num_samples:
        problems = problems[:num_samples]
    return problems


def collect_token_uncertainties(
    problems: List[Tuple[str, Dict]],
    calculator: UncertaintyCalculator,
) -> List[Dict]:
    """
    Teacher-forcing pass to collect per-token uncertainties for all problems.
    Returns a flat list of dicts with token-level stats and token indices.
    """
    all_rows = []
    for task_id, problem in tqdm(problems, desc="Collecting token uncertainties"):
        code = problem["prompt"] + problem["canonical_solution"]
        unc = calculator.compute_uncertainties_for_code(code, return_tokens=True)
        token_ids = calculator.tokenizer(code, return_tensors="pt").input_ids[0].tolist()

        ent = unc["entropies"]
        conf = unc["confidences"]
        ppl = unc["perplexities"]
        toks = unc["tokens"]

        for idx in range(len(ent)):
            row = {
                "task_id": task_id,
                "token_idx": idx,  # aligns with entropies list; token_ids[idx+1]
                "entropy": ent[idx],
                "confidence": conf[idx],
                "ppl": ppl[idx],
                "token_text": toks[idx],
                "token_ids": token_ids,  # store full sequence to rebuild prefix/suffix
            }
            all_rows.append(row)
    return all_rows


def compute_bins(values: np.ndarray, num_bins: int) -> np.ndarray:
    """Return bin edges based on quantiles (e.g., 5 bins -> quintiles)."""
    quantiles = np.linspace(0, 100, num_bins + 1)
    edges = np.percentile(values, quantiles)
    # ensure strictly increasing edges by jittering tiny amounts if needed
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges


def assign_bins(rows: List[Dict], key: str, edges: np.ndarray) -> None:
    """Add a 'bin' field to each row based on the chosen uncertainty key."""
    for r in rows:
        val = r[key]
        # np.digitize returns bin index in 1..len(edges)-1
        r["bin"] = int(np.digitize(val, edges) - 1)


def stratified_sample(rows: List[Dict], samples_per_bin: int, num_bins: int, rng: np.random.Generator) -> List[Dict]:
    """Sample the same number of rows per bin (with replacement if not enough)."""
    sampled = []
    rows_by_bin = {b: [] for b in range(num_bins)}
    for r in rows:
        if 0 <= r["bin"] < num_bins:
            rows_by_bin[r["bin"]].append(r)

    for b in range(num_bins):
        pool = rows_by_bin[b]
        if not pool:
            continue
        if len(pool) >= samples_per_bin:
            chosen = rng.choice(pool, size=samples_per_bin, replace=False)
        else:
            chosen = rng.choice(pool, size=samples_per_bin, replace=True)
        sampled.extend(list(chosen))
    return sampled


def decode_prefix_suffix(tokenizer, token_ids: List[int], ent_idx: int) -> Tuple[str, str]:
    """
    ent_idx is the index into entropies (0-based), which corresponds to token_ids[ent_idx+1].
    Prefix includes that token; suffix is the remainder.
    """
    split_pos = ent_idx + 1
    prefix_ids = token_ids[: split_pos + 1]
    suffix_ids = token_ids[split_pos + 1 :]
    prefix = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    suffix = tokenizer.decode(suffix_ids, skip_special_tokens=True)
    return prefix, suffix


def evaluate_samples(
    samples: List[Dict],
    tokenizer,
    line_completion: LineCompletion,
    metrics_calc: MetricsCalculator,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[Dict]:
    results = []
    for s in tqdm(samples, desc="Evaluating sampled cut points"):
        prefix, reference_suffix = decode_prefix_suffix(tokenizer, s["token_ids"], s["token_idx"])
        # generate one-line completion
        pred = line_completion.complete_line(
            prefix,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_at_newline=True,
        )
        metrics = metrics_calc.calculate_metrics(pred, reference_suffix)
        out = {
            **s,
            "prediction": pred,
            "reference": reference_suffix,
            "metrics": metrics,
        }
        results.append(out)
    return results


def aggregate_by_bin(samples_with_metrics: List[Dict], num_bins: int) -> Dict[int, Dict[str, float]]:
    agg = {}
    for b in range(num_bins):
        filtered = [s for s in samples_with_metrics if s.get("bin") == b]
        if not filtered:
            continue
        # pull metrics of interest
        codebleu = [s["metrics"].get("codebleu", 0) for s in filtered]
        em = [s["metrics"].get("exact_match", 0) for s in filtered]
        rouge = [s["metrics"].get("rouge_l_f1", 0) for s in filtered]
        bleu = [s["metrics"].get("bleu", 0) for s in filtered]
        agg[b] = {
            "count": len(filtered),
            "codebleu_mean": float(np.mean(codebleu)),
            "exact_match_mean": float(np.mean(em)),
            "rouge_l_f1_mean": float(np.mean(rouge)),
            "bleu_mean": float(np.mean(bleu)),
        }
    return agg


def run_for_model(
    model_name: str,
    problems: List[Tuple[str, Dict]],
    num_bins: int,
    samples_per_bin: int,
    uncertainty_type: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    out_dir: Path,
    rng: np.random.Generator,
):
    logging.info(f"Loading model {model_name}...")
    tokenizer, model = MODEL_FACTORY.get_model(model_name, device=device)
    model.eval()

    calculator = UncertaintyCalculator(model, tokenizer, device=device)
    line_completion = LineCompletion(model, tokenizer, device=device)
    metrics_calc = MetricsCalculator()

    # Step A: collect uncertainties
    rows = collect_token_uncertainties(problems, calculator)
    values = np.array([r[uncertainty_type] for r in rows], dtype=float)
    edges = compute_bins(values, num_bins)
    assign_bins(rows, uncertainty_type, edges)

    # Step B: stratified sampling
    sampled = stratified_sample(rows, samples_per_bin, num_bins, rng)

    # Step C: evaluate completions
    evaluated = evaluate_samples(
        sampled,
        tokenizer=tokenizer,
        line_completion=line_completion,
        metrics_calc=metrics_calc,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # Aggregate
    agg = aggregate_by_bin(evaluated, num_bins)

    # Save detailed and aggregated results
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"{model_name}_rq2_samples.jsonl"
    agg_path = out_dir / f"{model_name}_rq2_bins.json"
    with detail_path.open("w") as f:
        for s in evaluated:
            f.write(json.dumps(s) + "\n")
    with agg_path.open("w") as f:
        json.dump({"bin_edges": edges.tolist(), "bins": agg}, f, indent=2)

    logging.info(f"Saved samples to {detail_path}")
    logging.info(f"Saved bin aggregates to {agg_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="RQ2: Performance vs. Uncertainty curves")
    ap.add_argument("--models", nargs="+", required=True, help="Model names (e.g., qwen3-0.6b qwen3-1.7b qwen3-4b)")
    ap.add_argument("--dataset", default="humaneval", help="Dataset name (only humaneval supported)")
    ap.add_argument("--num-samples", type=int, default=None, help="Limit number of problems (None = all)")
    ap.add_argument("--num-bins", type=int, default=5, help="Number of quantile bins (default: 5)")
    ap.add_argument("--samples-per-bin", type=int, default=200, help="Samples per bin for evaluation")
    ap.add_argument("--uncertainty-type", choices=["entropy", "ppl"], default="entropy", help="Which uncertainty to bin on")
    ap.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens to generate for completion")
    ap.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    ap.add_argument("--top-p", type=float, default=0.95, help="Top-p for nucleus sampling")
    ap.add_argument("--device", default="cuda", help="Device for inference")
    ap.add_argument("--output-dir", default="experiments/rq2_uncertainty_curve", help="Output directory")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--log-file", default=None, help="Optional log file path")
    return ap.parse_args()


def main():
    args = parse_args()
    setup_logging(Path(args.log_file) if args.log_file else None)
    rng = np.random.default_rng(args.seed)

    if args.dataset != "humaneval":
        raise ValueError("Only humaneval is supported currently.")
    problems = load_dataset(args.num_samples)

    out_dir = Path(args.output_dir)
    for model_name in args.models:
        run_for_model(
            model_name=model_name,
            problems=problems,
            num_bins=args.num_bins,
            samples_per_bin=args.samples_per_bin,
            uncertainty_type=args.uncertainty_type,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
            out_dir=out_dir,
            rng=rng,
        )


if __name__ == "__main__":
    main()
