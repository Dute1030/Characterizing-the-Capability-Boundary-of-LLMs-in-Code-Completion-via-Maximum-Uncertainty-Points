"""
RQ1 Main Experiment Script
Maximum Uncertainty Point (MUP) Impact on Model Performance

Usage:
    python -m src.rq1.run_experiment \\
      --models qwen3-0.6b qwen3-1.7b qwen3-4b \\
      --dataset py150 \\
      --py150_path data/py150 \\
      --num_samples 200
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
import datetime
import numpy as np
from tqdm import tqdm
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from llm.models import MODEL_FACTORY
from src.rq1.uncertainty_calculator import UncertaintyCalculator
from src.rq1.code_splitter import CodeSplitter
from src.rq1.line_completion import LineCompletion
from src.rq1.metrics import MetricsCalculator
from src.rq1.visualizer import RQ1Visualizer
from src.rq1.datasets import load_rq1_dataset


def setup_logging(log_file: str = None):
    """Setup logging configuration"""
    log_format = '[%(levelname)s] %(asctime)s - %(message)s'

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        # Ensure directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='w'))

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )


def trim_at_first_newline(text: str) -> str:
    """Return text up to (but not including) the first newline."""
    if text is None:
        return ""
    newline_pos = text.find('\n')
    return text[:newline_pos] if newline_pos != -1 else text


def format_tokens(tokens: List[str], max_tokens: int = 80) -> List[str]:
    """Pretty format tokens with optional truncation."""
    if len(tokens) <= max_tokens:
        return tokens
    return tokens[:max_tokens] + ['...', f'(truncated, total={len(tokens)})']


def log_token_uncertainties(token_uncertainties: Dict[str, List], max_tokens: int = None):
    """Log per-token uncertainties: idx, token text, entropy, confidence, perplexity."""
    tokens = token_uncertainties.get('tokens', [])
    entropies = token_uncertainties.get('entropies', [])
    confidences = token_uncertainties.get('confidences', [])
    perplexities = token_uncertainties.get('perplexities', [])

    lines = []
    total = len(tokens)
    limit = max_tokens if max_tokens is not None else total

    for i in range(min(total, limit)):
        tok = tokens[i].replace('\n', '\\n')
        ent = entropies[i] if i < len(entropies) else None
        conf = confidences[i] if i < len(confidences) else None
        ppl = perplexities[i] if i < len(perplexities) else None
        lines.append(f"{i:4d} | {tok!r} | entropy={ent} | confidence={conf} | perplexity={ppl}")

    if total > limit:
        lines.append(f"... (truncated, total tokens={total})")

    logging.info("    Token uncertainties:\n%s", "\n".join(lines))


def generate_line_level_completion(
    prefix: str,
    line_completion: LineCompletion,
    max_total_tokens: int = 50,
    temperature: float = 0.2,
    top_p: float = 0.95
) -> str:
    """
    Generate completion until first newline or reaching max_total_tokens.
    """
    return line_completion.complete_line(
        prefix,
        max_new_tokens=max_total_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_at_newline=True
    )


def run_single_experiment(
    model_name: str,
    problem: Dict,
    task_id: str,
    uncertainty_calculator: UncertaintyCalculator,
    code_splitter: CodeSplitter,
    line_completion: LineCompletion,
    metrics_calculator: MetricsCalculator
) -> Dict[str, Dict]:
    """
    Run experiment on a single problem

    Args:
        model_name: Name of the model
        problem: Problem dictionary
        task_id: Task identifier
        uncertainty_calculator: Uncertainty calculator instance
        code_splitter: Code splitter instance
        line_completion: Line completion instance
        metrics_calculator: Metrics calculator instance
    Returns:
        Dictionary with results for all strategies
    """
    # Get the complete code (prompt + canonical solution)
    prompt = problem['prompt']
    canonical_solution = problem['canonical_solution']
    complete_code = prompt + canonical_solution

    # Pre-compute token-level uncertainties for logging
    token_uncertainties = uncertainty_calculator.compute_uncertainties_for_code(
        complete_code, return_tokens=True
    )
    entropies = token_uncertainties.get('entropies', [])
    confidences = token_uncertainties.get('confidences', [])
    perplexities = token_uncertainties.get('perplexities', [])
    tokens_all = token_uncertainties.get('tokens', [])

    # Log all token-level uncertainties (may be long)
    log_token_uncertainties(token_uncertainties)

    logging.info(f"Processing {task_id}...")
    logging.debug(f"Complete code:\n{complete_code}")

    # Split code using all strategies
    try:
        splits = code_splitter.split_at_all_strategies(complete_code, seed=42)
    except Exception as e:
        logging.error(f"Error splitting code for {task_id}: {e}")
        return {}

    # Evaluate each split strategy
    results = {}
    tokenizer = line_completion.tokenizer

    for strategy, (prefix, ground_truth_suffix, info) in splits.items():
        logging.info(f"  Strategy: {strategy}")
        logging.debug(f"  Prefix ({len(prefix)} chars):\n{prefix}")
        logging.debug(f"  Ground truth suffix ({len(ground_truth_suffix)} chars):\n{ground_truth_suffix}")

        if not prefix.strip():
            logging.warning(f"  Empty prefix for strategy {strategy}, skipping")
            continue

        # Log split uncertainty info if available
        split_idx = info.get('split_token_index')
        split_metric = info.get('metric')
        split_value = info.get('uncertainty_value')
        split_token_text = info.get('split_token_text', "")
        # Log target length (chars and tokens)
        target_char_len = len(ground_truth_suffix)
        target_tok_len = len(tokenizer(ground_truth_suffix, return_tensors="pt").input_ids[0])
        logging.info(f"    Target length: {target_char_len} chars, {target_tok_len} tokens")
        if split_idx is not None:
            ent = entropies[split_idx] if split_idx < len(entropies) else None
            conf = confidences[split_idx] if split_idx < len(confidences) else None
            ppl = perplexities[split_idx] if split_idx < len(perplexities) else None
            info['entropy_at_split'] = ent
            info['confidence_at_split'] = conf
            info['ppl_at_split'] = ppl
            logging.info(f"    Split at token idx {split_idx} ({repr(split_token_text)}) "
                         f"metric={split_metric} value={split_value}")
            logging.info(f"    Uncertainty at idx -> entropy={ent}, confidence={conf}, perplexity={ppl}")
        else:
            logging.info("    Split token index not available (random split).")

        try:
            # Generate completion
            completion = generate_line_level_completion(
                prefix=prefix,
                line_completion=line_completion,
                max_total_tokens=50,
                temperature=0.2,
                top_p=0.95
            )

            logging.debug(f"  Generated completion:\n{completion}")

            ref_used = trim_at_first_newline(ground_truth_suffix)
            pred_used = trim_at_first_newline(completion)

            # Generated length (chars and tokens)
            gen_char_len = len(pred_used)
            gen_tok_len = len(tokenizer(pred_used, return_tensors="pt").input_ids[0])
            logging.info(f"    Generated length: {gen_char_len} chars, {gen_tok_len} tokens")

            # Token-level views for context/pred/reference
            ctx_tokens = format_tokens(tokenizer.tokenize(prefix))
            ref_tokens = format_tokens(tokenizer.tokenize(ref_used))
            pred_tokens = format_tokens(tokenizer.tokenize(pred_used))

            logging.info(f"    Context tokens: {ctx_tokens}")
            logging.info(f"    Reference tokens: {ref_tokens}")
            logging.info(f"    Predicted tokens: {pred_tokens}")

            # Evaluate completion
            eval_results = metrics_calculator.evaluate_completion(
                prediction=pred_used,
                reference=ref_used,
                lang="python"
            )

            results[strategy] = {
                'metrics': eval_results,
                'prefix_length': len(prefix),
                'suffix_length': len(ref_used),
                'completion_length': len(pred_used),
                'completion_token_length': gen_tok_len,
                'split_info': info
            }

            logging.info(f"    CodeBLEU: {eval_results['codebleu']:.4f}, "
                        f"EM: {eval_results['exact_match']:.4f}, "
                        f"ROUGE-L F1: {eval_results['rouge_l_f1']:.4f}")

        except Exception as e:
            logging.error(f"  Error processing strategy {strategy}: {e}")
            results[strategy] = {
                'metrics': {},
                'error': str(e)
            }

    return results


def run_experiments(
    models: List[str],
    dataset_name: str,
    num_samples: int = None,
    output_dir: str = "experiments/rq1_results",
    device: str = "cuda",
    dataset_kwargs: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """
    Run experiments across multiple models

    Args:
        models: List of model names
        dataset_name: Name of dataset
        num_samples: Number of samples to process
        output_dir: Output directory for results
        device: Device to run on
        dataset_kwargs: Extra dataset-specific arguments (e.g., py150_path/min/max/shuffle)
    Returns:
        Dictionary with all results
    """
    # Create output directory (main already handles redirect to avoid overwrite)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_kwargs = dataset_kwargs or {}

    # Load dataset
    logging.info(f"Loading dataset: {dataset_name}")
    dataset = load_rq1_dataset(
        dataset_name=dataset_name,
        num_samples=num_samples,
        **dataset_kwargs,
    )
    logging.info(f"Loaded {len(dataset)} problems")

    # Initialize metrics calculator
    metrics_calculator = MetricsCalculator()

    # Store all results
    all_results = {}

    for model_name in models:
        logging.info(f"\n{'='*80}")
        logging.info(f"Running experiments with model: {model_name}")
        logging.info(f"{'='*80}\n")

        # Load model
        try:
            model, tokenizer = MODEL_FACTORY[model_name]()
            model.to(device)
            model.eval()
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {e}")
            continue

        # Initialize components
        uncertainty_calculator = UncertaintyCalculator(model, tokenizer, device)
        code_splitter = CodeSplitter(uncertainty_calculator)
        line_completion = LineCompletion(model, tokenizer, device)

        # Run experiments on all problems
        model_results = []
        split_rows = []

        for task_id, problem in tqdm(dataset, desc=f"Processing {model_name}"):
            result = run_single_experiment(
                model_name=model_name,
                problem=problem,
                task_id=task_id,
                uncertainty_calculator=uncertainty_calculator,
                code_splitter=code_splitter,
                line_completion=line_completion,
                metrics_calculator=metrics_calculator
            )

            if result:
                result['task_id'] = task_id
                model_results.append(result)
                # collect split positions per strategy
                for strat in ['random', 'entropy', 'confidence', 'ppl', 'entropy_min', 'confidence_max', 'ppl_min']:
                    if strat in result and 'split_info' in result[strat]:
                        info = result[strat]['split_info']
                        split_rows.append({
                            'task_id': task_id,
                            'strategy': strat,
                            'split_token_index': info.get('split_token_index'),
                            'split_token_text': info.get('split_token_text'),
                            'uncertainty_metric': info.get('metric'),
                            'uncertainty_value': info.get('uncertainty_value'),
                            'prominence': info.get('prominence'),
                            'entropy_at_split': info.get('entropy_at_split'),
                            'confidence_at_split': info.get('confidence_at_split'),
                            'ppl_at_split': info.get('ppl_at_split')
                        })

        # Aggregate results by strategy
        aggregated = aggregate_results_by_strategy(model_results, metrics_calculator)
        all_results[model_name] = aggregated

        # Save intermediate results
        model_output_file = output_path / f"{model_name}_results.json"
        model_output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(model_output_file, 'w') as f:
            json.dump(aggregated, f, indent=2)
        logging.info(f"Saved results for {model_name} to {model_output_file}")

        # Save split positions table for this model
        if split_rows:
            split_file = output_path / f"{model_name}_split_positions.csv"
            split_file.parent.mkdir(parents=True, exist_ok=True)
            with open(split_file, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        'task_id', 'strategy', 'split_token_index',
                        'split_token_text', 'uncertainty_metric',
                        'uncertainty_value', 'prominence',
                        'entropy_at_split', 'confidence_at_split', 'ppl_at_split'
                    ]
                )
                writer.writeheader()
                writer.writerows(split_rows)
            logging.info(f"Saved split positions to {split_file}")

        # Clear GPU memory
        del model, tokenizer, uncertainty_calculator, code_splitter, line_completion
        torch.cuda.empty_cache()

    # Save combined results
    combined_output_file = output_path / "all_results.json"
    combined_output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Saved combined results to {combined_output_file}")

    return all_results


def aggregate_results_by_strategy(
    results: List[Dict],
    metrics_calculator: MetricsCalculator
) -> Dict[str, Dict]:
    """
    Aggregate results by splitting strategy

    Args:
        results: List of result dictionaries
        metrics_calculator: Metrics calculator instance
    Returns:
        Dictionary with aggregated results per strategy
    """
    strategies = ['random', 'entropy', 'confidence', 'ppl', 'entropy_min', 'confidence_max', 'ppl_min']
    aggregated = {}

    for strategy in strategies:
        strategy_metrics = []

        for result in results:
            if strategy in result and 'metrics' in result[strategy]:
                strategy_metrics.append(result[strategy]['metrics'])

        if strategy_metrics:
            aggregated[strategy] = metrics_calculator.aggregate_results(strategy_metrics)
        else:
            aggregated[strategy] = {}

    return aggregated


def visualize_results(
    results: Dict[str, Dict],
    output_dir: str = "experiments/rq1_results"
):
    """
    Create visualizations for results

    Args:
        results: Combined results dictionary
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if we have any results
    if not results or all(not v for v in results.values()):
        logging.warning("No results to visualize")
        return

    visualizer = RQ1Visualizer()

    logging.info("\nGenerating visualizations...")

    # Grouped bar charts for different metrics
    metrics_to_plot = ['codebleu', 'exact_match', 'rouge_l_f1', 'bleu']

    for metric in metrics_to_plot:
        try:
            visualizer.plot_grouped_bar_chart(
                results=results,
                metric=metric,
                save_path=str(output_path / f"bar_chart_{metric}.png")
            )
        except Exception as e:
            logging.warning(f"Failed to generate bar chart for {metric}: {e}")

    # Multi-metric bar chart
    try:
        visualizer.plot_multiple_metrics_bar_chart(
            results=results,
            metrics=metrics_to_plot,
            save_path=str(output_path / "bar_chart_all_metrics.png")
        )
    except Exception as e:
        logging.warning(f"Failed to generate multi-metric bar chart: {e}")

    # Radar charts - only if we have at least one model with results
    if results:
        # For each model, create a radar chart
        for model_name, model_results in results.items():
            if not model_results:
                continue

            try:
                # Convert to format expected by radar chart
                radar_data = {}
                for strategy, metrics in model_results.items():
                    radar_data[strategy] = {
                        metric: values['mean']
                        for metric, values in metrics.items()
                        if isinstance(values, dict) and 'mean' in values
                    }

                if radar_data:
                    visualizer.plot_radar_chart(
                        results=radar_data,
                        metrics=metrics_to_plot,
                        save_path=str(output_path / f"radar_chart_{model_name}.png"),
                        title=f"Performance Comparison - {model_name}"
                    )
            except Exception as e:
                logging.warning(f"Failed to generate radar chart for {model_name}: {e}")

    # Combined radar chart for all models - only if we have multiple models
    if len(results) > 1:
        try:
            visualizer.plot_radar_chart_by_model(
                results=results,
                metrics=metrics_to_plot,
                save_path=str(output_path / "radar_chart_all_models.png")
            )
        except Exception as e:
            logging.warning(f"Failed to generate combined radar chart: {e}")

    # Performance degradation chart
    try:
        visualizer.plot_performance_degradation(
            results=results,
            baseline_strategy='random',
            metrics=['codebleu', 'exact_match'],
            save_path=str(output_path / "performance_degradation.png")
        )
    except Exception as e:
        logging.warning(f"Failed to generate performance degradation chart: {e}")

    # Results table
    try:
        df = visualizer.create_results_table(
            results=results,
            metrics=metrics_to_plot,
            save_path=str(output_path / "results_table.csv")
        )

        logging.info(f"\nResults Summary:")
        if not df.empty:
            print(df.to_string(index=False))
    except Exception as e:
        logging.warning(f"Failed to generate results table: {e}")

    logging.info(f"\nVisualization complete! Results saved to {output_path}")


def run_max_entropy_demo(
    model_name: str,
    dataset_name: str,
    device: str = "cuda",
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_lines: int = None,
    num_samples: int = None,
    dataset_kwargs: Optional[Dict] = None,
):
    """
    Teacher-forcing 逐token计算熵，按最大熵切点做行级补全并打印每步上下文/生成/参考
    """
    logging.info("Running max-entropy demo with model: %s", model_name)

    dataset_kwargs = dataset_kwargs or {}
    dataset = load_rq1_dataset(dataset_name, num_samples or 1, **dataset_kwargs)
    if not dataset:
        logging.error("Dataset is empty")
        return

    task_id, problem = dataset[0]
    complete_code = problem['prompt'] + problem['canonical_solution']

    model, tokenizer = MODEL_FACTORY[model_name]()
    model.to(device)
    model.eval()

    uc = UncertaintyCalculator(model, tokenizer, device)
    lc = LineCompletion(model, tokenizer, device)
    mc = MetricsCalculator()

    # Teacher-forcing 逐token不确定性
    uncertainties = uc.compute_uncertainties_for_code(complete_code, return_tokens=True)
    entropies = uncertainties.get('entropies', [])
    tokens = uncertainties.get('tokens', [])

    if not entropies:
        logging.error("No tokens found for entropy calculation.")
        return

    max_idx = int(np.argmax(entropies))
    max_entropy = entropies[max_idx]
    max_token = tokens[max_idx] if max_idx < len(tokens) else "<unk>"

    all_token_ids = tokenizer(complete_code, return_tensors="pt").input_ids[0]
    prefix_ids = all_token_ids[: max_idx + 1]
    suffix_ids = all_token_ids[max_idx + 1 :]
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    suffix_ref = tokenizer.decode(suffix_ids, skip_special_tokens=True)

    logging.info(f"Task: {task_id}")
    logging.info(f"Max-entropy token idx {max_idx} text {repr(max_token)} value {max_entropy:.4f}")
    logging.info("---- Prefix (context) ----\n%s", prefix_text)
    logging.info("---- Ground truth suffix ----\n%s", suffix_ref)

    gt_lines = suffix_ref.splitlines(keepends=True)
    if max_lines:
        gt_lines = gt_lines[:max_lines]

    current_prefix = prefix_text
    generated_lines = []

    for step_idx, gt_line in enumerate(gt_lines, start=1):
        gen_line = lc.complete_line(
            current_prefix,
            max_new_tokens=128,
            temperature=temperature,
            top_p=top_p,
            stop_at_newline=True
        )
        generated_lines.append(gen_line)

        context_tail = "\n".join(current_prefix.split('\n')[-6:])
        logging.info(f"\n===== Step {step_idx} =====")
        logging.info("Context tail:\n%s", context_tail)
        logging.info("Generated line:\n%s", gen_line if gen_line.strip() else "(empty)")
        logging.info("Ground truth line:\n%s", gt_line.rstrip('\n'))

        current_prefix += gen_line
        if not gen_line.strip():
            logging.info("Generation returned empty line, stop.")
            break

    completion_text = "".join(generated_lines)
    metrics = mc.evaluate_completion(completion_text, suffix_ref, lang="python")

    logging.info("\n===== Metrics (generated suffix vs reference) =====")
    for k, v in metrics.items():
        if isinstance(v, float):
            logging.info("%s: %.4f", k, v)
        else:
            logging.info("%s: %s", k, v)

    # 清理显存
    del model, tokenizer, uc, lc, mc
    torch.cuda.empty_cache()


def run_min_uncertainty_demo(
    model_name: str,
    dataset_name: str,
    metric: str = "entropy",
    device: str = "cuda",
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_lines: int = None,
    num_samples: int = None,
    dataset_kwargs: Optional[Dict] = None,
):
    """
    Teacher-forcing 逐token计算不确定性，按最低不确定性切点做行级补全并打印上下文/生成/参考。
    metric:
        - entropy/perplexity: 选最小值
        - confidence: 选最大值(最高置信度代表最低不确定性)
    """
    logging.info("Running min-uncertainty demo with model: %s", model_name)
    logging.info("Using metric: %s", metric)

    dataset_kwargs = dataset_kwargs or {}
    dataset = load_rq1_dataset(dataset_name, num_samples or 1, **dataset_kwargs)
    if not dataset:
        logging.error("Dataset is empty")
        return

    task_id, problem = dataset[0]
    complete_code = problem['prompt'] + problem['canonical_solution']

    model, tokenizer = MODEL_FACTORY[model_name]()
    model.to(device)
    model.eval()

    uc = UncertaintyCalculator(model, tokenizer, device)
    lc = LineCompletion(model, tokenizer, device)
    mc = MetricsCalculator()

    uncertainties = uc.compute_uncertainties_for_code(complete_code, return_tokens=True)
    tokens = uncertainties.get('tokens', [])

    if metric == "entropy":
        values = uncertainties.get('entropies', [])
        selector = np.argmin
    elif metric == "perplexity":
        values = uncertainties.get('perplexities', [])
        selector = np.argmin
    elif metric == "confidence":
        values = uncertainties.get('confidences', [])
        selector = np.argmax
    else:
        logging.error("Unsupported metric: %s", metric)
        return

    if not values:
        logging.error("No tokens found for %s calculation.", metric)
        return

    min_idx = int(selector(values))
    min_value = values[min_idx]
    min_token = tokens[min_idx] if min_idx < len(tokens) else "<unk>"

    all_token_ids = tokenizer(complete_code, return_tensors="pt").input_ids[0]
    prefix_ids = all_token_ids[: min_idx + 1]
    suffix_ids = all_token_ids[min_idx + 1 :]
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    suffix_ref = tokenizer.decode(suffix_ids, skip_special_tokens=True)

    logging.info(f"Task: {task_id}")
    logging.info(
        "Min-uncertainty token idx %d text %s metric=%s value=%.4f",
        min_idx,
        repr(min_token),
        metric,
        min_value,
    )
    logging.info("---- Prefix (context) ----\n%s", prefix_text)
    logging.info("---- Ground truth suffix ----\n%s", suffix_ref)

    gt_lines = suffix_ref.splitlines(keepends=True)
    if max_lines:
        gt_lines = gt_lines[:max_lines]

    current_prefix = prefix_text
    generated_lines = []

    for step_idx, gt_line in enumerate(gt_lines, start=1):
        gen_line = lc.complete_line(
            current_prefix,
            max_new_tokens=128,
            temperature=temperature,
            top_p=top_p,
            stop_at_newline=True
        )
        generated_lines.append(gen_line)

        context_tail = "\n".join(current_prefix.split('\n')[-6:])
        logging.info(f"\n===== Step {step_idx} =====")
        logging.info("Context tail:\n%s", context_tail)
        logging.info("Generated line:\n%s", gen_line if gen_line.strip() else "(empty)")
        logging.info("Ground truth line:\n%s", gt_line.rstrip('\n'))

        current_prefix += gen_line
        if not gen_line.strip():
            logging.info("Generation returned empty line, stop.")
            break

    completion_text = "".join(generated_lines)
    metrics = mc.evaluate_completion(completion_text, suffix_ref, lang="python")

    logging.info("\n===== Metrics (generated suffix vs reference) =====")
    for k, v in metrics.items():
        if isinstance(v, float):
            logging.info("%s: %.4f", k, v)
        else:
            logging.info("%s: %s", k, v)

    del model, tokenizer, uc, lc, mc
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="RQ1: MUP Impact Experiment")

    parser.add_argument(
        '--models',
        nargs='+',
        default=['qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b', 'deepseek-1.3b', 'deepseek-6.7b'],
        help='List of models to evaluate'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='humaneval',
        choices=['humaneval', 'py150', 'mbpp'],
        help='Dataset to use'
    )
    parser.add_argument(
        '--py150_path',
        type=str,
        default='data/py150',
        help='Path to Py150 dataset (directory of .py files or a .json/.jsonl file)'
    )
    parser.add_argument(
        '--py150_min_chars',
        type=int,
        default=80,
        help='Minimum total characters (prompt + solution) to keep a Py150 sample'
    )
    parser.add_argument(
        '--py150_max_chars',
        type=int,
        default=800,
        help='Maximum total characters (prompt + solution) to keep a Py150 sample'
    )
    parser.add_argument(
        '--py150_shuffle',
        action='store_true',
        default=True,
        help='Shuffle Py150 candidates before sampling (default: True)'
    )
    parser.add_argument(
        '--no_py150_shuffle',
        dest='py150_shuffle',
        action='store_false',
        help='Disable shuffling for Py150 sampling'
    )
    parser.add_argument(
        '--py150_seed',
        type=int,
        default=42,
        help='Random seed for Py150 shuffling/sampling'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to process (default: all available)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/rq1_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on'
    )
    parser.add_argument(
        '--visualize_only',
        action='store_true',
        help='Only generate visualizations from existing results'
    )
    parser.add_argument(
        '--max_entropy_demo',
        action='store_true',
        help='Run max-entropy line completion demo on first sample and exit'
    )
    parser.add_argument(
        '--min_uncertainty_demo',
        action='store_true',
        help='Run min-uncertainty line completion demo on first sample and exit'
    )
    parser.add_argument(
        '--demo_metric',
        type=str,
        default='entropy',
        choices=['entropy', 'confidence', 'perplexity'],
        help='Metric used for min_uncertainty_demo (entropy/perplexity=min value, confidence=max value)'
    )
    parser.add_argument(
        '--demo_temperature',
        type=float,
        default=0.2,
        help='Sampling temperature for demo runs'
    )
    parser.add_argument(
        '--demo_top_p',
        type=float,
        default=0.95,
        help='Top-p for demo runs'
    )
    parser.add_argument(
        '--demo_max_lines',
        type=int,
        default=None,
        help='Limit generated lines in demo runs (optional)'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Log file path'
    )

    args = parser.parse_args()
    dataset_kwargs = {
        "py150_path": args.py150_path,
        "py150_min_chars": args.py150_min_chars,
        "py150_max_chars": args.py150_max_chars,
        "py150_shuffle": args.py150_shuffle,
        "py150_seed": args.py150_seed,
    }

    # If not visualize-only/demo, avoid overwriting existing output_dir by appending timestamp when non-empty
    if not args.visualize_only and not args.max_entropy_demo and not args.min_uncertainty_demo:
        out_path = Path(args.output_dir)
        if out_path.exists() and any(out_path.iterdir()):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_out = out_path.parent / f"{out_path.name}_{ts}"
            print(f"[INFO] Output dir exists and not empty, redirecting to: {new_out}")
            args.output_dir = str(new_out)

    # Setup logging
    log_file = args.log_file or os.path.join(args.output_dir, 'experiment.log')
    setup_logging(log_file)

    if args.max_entropy_demo and args.min_uncertainty_demo:
        logging.error("Choose only one demo mode: max_entropy_demo or min_uncertainty_demo.")
        return

    if args.max_entropy_demo:
        model_name = args.models[0]
        if len(args.models) > 1:
            logging.info("max_entropy_demo uses only the first model: %s", model_name)

        logging.info("Max-entropy demo mode. Skipping full experiment.")
        run_max_entropy_demo(
            model_name=model_name,
            dataset_name=args.dataset,
            device=args.device,
            temperature=args.demo_temperature,
            top_p=args.demo_top_p,
            max_lines=args.demo_max_lines,
            num_samples=args.num_samples,
            dataset_kwargs=dataset_kwargs,
        )
        return

    if args.min_uncertainty_demo:
        model_name = args.models[0]
        if len(args.models) > 1:
            logging.info("min_uncertainty_demo uses only the first model: %s", model_name)

        logging.info("Min-uncertainty demo mode. Skipping full experiment.")
        run_min_uncertainty_demo(
            model_name=model_name,
            dataset_name=args.dataset,
            metric=args.demo_metric,
            device=args.device,
            temperature=args.demo_temperature,
            top_p=args.demo_top_p,
            max_lines=args.demo_max_lines,
            num_samples=args.num_samples,
            dataset_kwargs=dataset_kwargs,
        )
        return

    if args.visualize_only:
        # Load existing results and visualize
        results_file = Path(args.output_dir) / "all_results.json"
        if not results_file.exists():
            logging.error(f"Results file not found: {results_file}")
            return

        with open(results_file, 'r') as f:
            results = json.load(f)

        visualize_results(results, args.output_dir)
    else:
        # Run full experiment
        logging.info("Starting RQ1 Experiment")
        logging.info(f"Models: {args.models}")
        logging.info(f"Dataset: {args.dataset}")
        sample_desc = args.num_samples if args.num_samples else "ALL"
        logging.info(f"Number of samples: {sample_desc}")
        if args.dataset == "py150":
            logging.info(
                "Py150 settings -> path=%s min_chars=%d max_chars=%s shuffle=%s seed=%d",
                args.py150_path,
                args.py150_min_chars,
                args.py150_max_chars if args.py150_max_chars is not None else "None",
                args.py150_shuffle,
                args.py150_seed,
            )
        logging.info(f"Output directory: {args.output_dir}")

        results = run_experiments(
            models=args.models,
            dataset_name=args.dataset,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            device=args.device,
            dataset_kwargs=dataset_kwargs,
        )

        # Generate visualizations
        visualize_results(results, args.output_dir)

        logging.info("\nExperiment complete!")


if __name__ == "__main__":
    main()
