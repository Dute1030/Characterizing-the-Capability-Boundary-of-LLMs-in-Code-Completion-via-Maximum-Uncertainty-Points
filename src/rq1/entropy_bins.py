import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Ensure project root and data/human_eval are on the path
ROOT = Path(__file__).resolve().parents[2]
for p in [ROOT, ROOT / "data", ROOT / "data" / "human_eval"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from llm.models import MODEL_FACTORY
from src.rq1.line_completion import LineCompletion
from src.rq1.metrics import MetricsCalculator
from src.rq1.uncertainty_calculator import UncertaintyCalculator

# Compatibility: handle humaneval installed from source or pip
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


def setup_logging(log_file: Optional[Path] = None) -> None:
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        handlers=handlers,
    )


def load_dataset(num_samples: Optional[int]) -> List[Tuple[str, Dict]]:
    problems = [(task_id, problem) for task_id, problem in read_problems().items()]
    if num_samples:
        problems = problems[:num_samples]
    return problems


def collect_token_uncertainties(
    problems: List[Tuple[str, Dict]],
    calculator: UncertaintyCalculator,
    tokenizer,
) -> List[Dict]:
    """
    Compute uncertainty for each token under teacher-forcing.
    Each returned record includes task_id, token_idx, entropy, confidence, ppl, and the full token_ids.
    """
    rows: List[Dict] = []
    for task_id, problem in tqdm(problems, desc="Collecting token uncertainties"):
        code = problem["prompt"] + problem["canonical_solution"]
        unc = calculator.compute_uncertainties_for_code(code, return_tokens=True)
        token_ids = tokenizer(code, return_tensors="pt").input_ids[0].tolist()

        ent = unc["entropies"]
        conf = unc["confidences"]
        ppl = unc["perplexities"]

        for idx in range(len(ent)):
            rows.append(
                {
                    "task_id": task_id,
                    "token_idx": idx,  # ent[idx] corresponds to token_ids[idx+1]
                    "entropy": ent[idx],
                    "confidence": conf[idx],
                    "ppl": ppl[idx],
                    "token_ids": token_ids,
                }
            )
    return rows


def compute_bins(values: np.ndarray, num_bins: int) -> np.ndarray:
    """Compute bin edges using quantiles and ensure monotonic increasing edges."""
    edges = np.quantile(values, np.linspace(0, 1, num_bins + 1))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges


def assign_bins(rows: List[Dict], values: np.ndarray, edges: np.ndarray) -> None:
    """Assign bin indices based on the given values; values align with rows order."""
    for i, r in enumerate(rows):
        r["bin"] = int(np.digitize(values[i], edges) - 1)  # 0 ~ num_bins-1


def stratified_sample(
    rows: List[Dict], samples_per_bin: int, num_bins: int, rng: np.random.Generator
) -> List[Dict]:
    sampled: List[Dict] = []
    rows_by_bin = {b: [] for b in range(num_bins)}
    for r in rows:
        b = r.get("bin", -1)
        if 0 <= b < num_bins:
            rows_by_bin[b].append(r)

    for b in range(num_bins):
        pool = rows_by_bin[b]
        if not pool:
            continue
        replace = len(pool) < samples_per_bin
        chosen = rng.choice(pool, size=samples_per_bin, replace=replace)
        sampled.extend(list(chosen))
    return sampled


def decode_prefix_suffix(tokenizer, token_ids: List[int], ent_idx: int) -> Tuple[str, str]:
    """
    ent_idx is the index into entropies; the target token is token_ids[ent_idx+1].
    The prefix includes that token; the suffix contains the following content.
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
    lang: str,
) -> List[Dict]:
    results = []
    for s in tqdm(samples, desc="Evaluating sampled cut points"):
        prefix, reference_suffix = decode_prefix_suffix(tokenizer, s["token_ids"], s["token_idx"])
        prediction = line_completion.complete_line(
            prefix,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_at_newline=True,
        )
        metrics = metrics_calc.evaluate_completion(prediction, reference_suffix, lang=lang)
        out = {
            **s,
            "prediction": prediction,
            "reference": reference_suffix,
            "metrics": metrics,
        }
        # Reduce output size; token_ids are not needed after evaluation
        out.pop("token_ids", None)
        results.append(out)
    return results


def aggregate_by_bin(samples_with_metrics: List[Dict], num_bins: int) -> Dict[int, Dict[str, float]]:
    agg: Dict[int, Dict[str, float]] = {}
    for b in range(num_bins):
        bin_samples = [s for s in samples_with_metrics if s.get("bin") == b]
        if not bin_samples:
            agg[b] = {}
            continue
        metrics_keys = list(bin_samples[0]["metrics"].keys())
        agg[b] = {
            k: float(np.mean([s["metrics"][k] for s in bin_samples]))
            for k in metrics_keys
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
    lang: str,
    out_dir: Path,
    rng: np.random.Generator,
) -> None:
    logging.info(f"[{model_name}] Loading model/tokenizer...")
    model, tokenizer = MODEL_FACTORY[model_name](device=device)
    calculator = UncertaintyCalculator(model, tokenizer, device=device)
    line_completion = LineCompletion(model, tokenizer, device=device)
    metrics_calc = MetricsCalculator()

    # Step A: Collect uncertainties for all tokens
    rows = collect_token_uncertainties(problems, calculator, tokenizer)
    if uncertainty_type == "confidence":
        # Low confidence = high uncertainty; use (1 - confidence) for binning
        values = np.array([1.0 - r["confidence"] for r in rows])
    else:
        values = np.array([r[uncertainty_type] for r in rows])
    edges = compute_bins(values, num_bins)
    assign_bins(rows, values, edges)

    # Step B: Stratified sampling
    sampled = stratified_sample(rows, samples_per_bin, num_bins, rng)

    # Step C: Evaluate completions
    evaluated = evaluate_samples(
        sampled,
        tokenizer=tokenizer,
        line_completion=line_completion,
        metrics_calc=metrics_calc,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        lang=lang,
    )

    # Aggregation
    agg = aggregate_by_bin(evaluated, num_bins)

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"{model_name}_rq2_samples.jsonl"
    agg_path = out_dir / f"{model_name}_rq2_bins.json"
    with detail_path.open("w", encoding="utf-8") as f:
        for s in evaluated:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with agg_path.open("w", encoding="utf-8") as f:
        json.dump({"bin_edges": edges.tolist(), "bins": agg}, f, ensure_ascii=False, indent=2)

    logging.info(f"[{model_name}] Saved sample details to {detail_path}")
    logging.info(f"[{model_name}] Saved bin aggregation to {agg_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="RQ2: Performance-Uncertainty Response Curve (entropy bins)")
    ap.add_argument("--models", nargs="+", required=True, help="List of model names (keys in llm.models.MODEL_FACTORY)")
    ap.add_argument("--dataset", default="humaneval", help="Dataset name; currently only 'humaneval' is supported")
    ap.add_argument("--num-samples", type=int, default=None, help="Limit number of problems to use (default: all)")
    ap.add_argument("--num-bins", type=int, default=5, help="Number of bins (default: 5)")
    ap.add_argument("--samples-per-bin", type=int, default=200, help="Samples per bin")
    ap.add_argument(
        "--uncertainty-type",
        choices=["entropy", "ppl", "confidence"],
        default="entropy",
        help="Choose binning by entropy, PPL, or confidence (for confidence, use 1 - confidence as uncertainty)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=64, help="Maximum tokens to generate for completion")
    ap.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    ap.add_argument("--top-p", type=float, default=0.95, help="Top-p (nucleus sampling)")
    ap.add_argument("--device", default="cuda", help="Inference device")
    ap.add_argument("--lang", default="python", help="Language tag used for metrics like CodeBLEU")
    ap.add_argument("--output-dir", default="experiments/rq2_entropy_bins", help="Output directory")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--log-file", default=None, help="Optional log file path")
    return ap.parse_args()


def main():
    args = parse_args()
    setup_logging(Path(args.log_file) if args.log_file else None)
    rng = np.random.default_rng(args.seed)

    if args.dataset != "humaneval":
        raise ValueError("Currently only the 'humaneval' dataset is supported.")
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
            lang=args.lang,
            out_dir=out_dir,
            rng=rng,
        )


if __name__ == "__main__":
    main()
