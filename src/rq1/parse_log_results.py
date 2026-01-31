#!/usr/bin/env python
"""
Parse experiment.log to extract per-task, per-strategy metrics into CSV.

Usage:
  python src/rq1/parse_log_results.py --log_file experiments/rq1_full_results_new/experiment.log \
      --output_csv experiments/rq1_full_results_new/parsed_results.csv
"""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Dict


def parse_log(log_text: str) -> List[Dict]:
    """Parse log content into a list of metric rows."""
    rows = []
    current_task = None
    current_strategy = None

    task_re = re.compile(r"Processing\s+([A-Za-z0-9_/.-]+)")
    strat_re = re.compile(r"Strategy:\s+(\w+)")
    metrics_re = re.compile(
        r"CodeBLEU:\s*([0-9.]+),\s*EM:\s*([0-9.]+),\s*ROUGE-L F1:\s*([0-9.]+)",
        re.IGNORECASE
    )

    for line in log_text.splitlines():
        task_match = task_re.search(line)
        if task_match:
            current_task = task_match.group(1)
            continue

        strat_match = strat_re.search(line)
        if strat_match:
            current_strategy = strat_match.group(1)
            continue

        metrics_match = metrics_re.search(line)
        if metrics_match and current_task and current_strategy:
            codebleu, em, rouge_f1 = metrics_match.groups()
            rows.append({
                "task_id": current_task,
                "strategy": current_strategy,
                "codebleu": float(codebleu),
                "exact_match": float(em),
                "rouge_l_f1": float(rouge_f1),
            })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Parse experiment.log into CSV metrics")
    parser.add_argument("--log_file", required=True, help="Path to experiment.log")
    parser.add_argument(
        "--output_csv",
        help="Path to output CSV (default: same dir as log_file, parsed_results.csv)"
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    log_text = log_path.read_text(encoding="utf-8")

    rows = parse_log(log_text)
    if not rows:
        print("No metrics found in log.")
        return

    out_path = Path(args.output_csv) if args.output_csv else log_path.parent / "parsed_results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "strategy", "codebleu", "exact_match", "rouge_l_f1"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
