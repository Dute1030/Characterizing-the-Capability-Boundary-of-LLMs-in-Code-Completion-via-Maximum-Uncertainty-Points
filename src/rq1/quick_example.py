"""
Quick Example Script for RQ1 Experiment
Demonstrates how to use the RQ1 modules on a single example
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
from llm.models import MODEL_FACTORY
from src.rq1 import (
    UncertaintyCalculator,
    CodeSplitter,
    LineCompletion,
    MetricsCalculator,
    RQ1Visualizer
)


def main():
    # Example code to analyze
    example_code = """def calculate_sum(numbers):
    \"\"\"Calculate the sum of a list of numbers\"\"\"
    total = 0
    for num in numbers:
        if num > 0:
            total += num
    return total

def find_max(numbers):
    \"\"\"Find the maximum number in a list\"\"\"
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    return max_val
"""

    print("=" * 80)
    print("RQ1 Experiment - Quick Example")
    print("=" * 80)
    print()

    # Load a small model for quick demo
    model_name = "qwen3-0.6b"
    print(f"Loading model: {model_name}...")
    model, tokenizer = MODEL_FACTORY[model_name]()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    print()

    # Initialize components
    print("Initializing components...")
    uncertainty_calculator = UncertaintyCalculator(model, tokenizer, device)
    code_splitter = CodeSplitter(uncertainty_calculator)
    line_completion = LineCompletion(model, tokenizer, device)
    metrics_calculator = MetricsCalculator()
    print("Components initialized")
    print()

    # Step 1: Calculate uncertainties
    print("Step 1: Calculating uncertainties for the code...")
    uncertainties = uncertainty_calculator.compute_uncertainties_for_code(
        example_code,
        return_tokens=True
    )
    print(f"  - Computed uncertainties for {len(uncertainties['entropies'])} tokens")
    print(f"  - Average entropy: {sum(uncertainties['entropies']) / len(uncertainties['entropies']):.4f}")
    print(f"  - Average confidence: {sum(uncertainties['confidences']) / len(uncertainties['confidences']):.4f}")
    print()

    # Step 2: Split code using different strategies
    print("Step 2: Splitting code using different strategies...")
    splits = code_splitter.split_at_all_strategies(example_code, seed=42)

    for strategy, (prefix, suffix, info) in splits.items():
        print(f"\n  Strategy: {strategy.upper()}")
        print(f"    Prefix lines: {len(prefix.split(chr(10)))}")
        print(f"    Suffix lines: {len(suffix.split(chr(10)))}")
        if 'split_line_number' in info:
            print(f"    Split at line: {info['split_line_number']}")
        if 'uncertainty_value' in info:
            print(f"    Uncertainty value: {info['uncertainty_value']:.4f}")
    print()

    # Step 3: Perform completions for each split
    print("Step 3: Performing code completions...")
    results = {}

    for strategy, (prefix, suffix, info) in splits.items():
        print(f"\n  Completing for strategy: {strategy.upper()}")

        if not prefix.strip():
            print("    Skipping (empty prefix)")
            continue

        # Generate completion
        completion = line_completion.complete_until_valid_code(
            prefix,
            max_new_tokens=128,
            temperature=0.2,
            top_p=0.95
        )

        print(f"    Generated {len(completion)} characters")
        print(f"    Preview: {completion[:100]}...")

        # Evaluate
        eval_results = metrics_calculator.evaluate_completion(
            prediction=completion,
            reference=suffix,
            lang="python"
        )

        results[strategy] = eval_results

        print(f"    Metrics:")
        print(f"      - Exact Match: {eval_results['exact_match']:.4f}")
        print(f"      - CodeBLEU: {eval_results['codebleu']:.4f}")
        print(f"      - ROUGE-L F1: {eval_results['rouge_l_f1']:.4f}")
        print(f"      - BLEU: {eval_results['bleu']:.4f}")

    print()
    print("=" * 80)

    # Step 4: Visualize results (single model)
    print("\nStep 4: Creating visualizations...")

    visualizer = RQ1Visualizer()

    # Create a simple radar chart
    visualizer.plot_radar_chart(
        results=results,
        metrics=['codebleu', 'exact_match', 'rouge_l_f1', 'bleu'],
        save_path="rq1_quick_example_radar.png",
        title=f"Quick Example - {model_name}"
    )

    print("Visualization saved to: rq1_quick_example_radar.png")
    print()

    # Print summary
    print("=" * 80)
    print("Summary:")
    print("=" * 80)
    print()
    print("Performance by Strategy:")
    print()
    for strategy in ['random', 'entropy', 'confidence', 'ppl']:
        if strategy in results:
            r = results[strategy]
            print(f"{strategy.upper():12s}: CodeBLEU={r['codebleu']:.4f}, "
                  f"EM={r['exact_match']:.4f}, "
                  f"ROUGE-L={r['rouge_l_f1']:.4f}")

    print()
    print("Experiment complete!")


if __name__ == "__main__":
    main()
