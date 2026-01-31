# -*- coding: utf-8 -*-
"""
从 RQ1 split_positions.csv 构造 RQ3 所需的样本 JSONL。

输入：
  - split_positions.csv（包含 task_id / strategy / split_token_index 等）
  - 指定策略（如 entropy_min / entropy / confidence 等）
  - 模型名称，用于 tokenizer 生成 token_ids

输出：JSONL，每行字段
  - task_id
  - token_idx      （来自 split_token_index）
  - code           （prompt + canonical_solution）
  - token_ids      （完整 token 序列，含 BOS/EOS）
  - split_token_text / uncertainty_metric / uncertainty_value / prominence（若存在）
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.models import MODEL_FACTORY  # noqa: E402

# 兼容 human_eval 读取
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


def parse_args():
    ap = argparse.ArgumentParser(description="Build RQ3 samples from RQ1 split_positions.csv")
    ap.add_argument("--csv", help="split_positions.csv path (legacy)")
    ap.add_argument("--split-csv", help="split_positions.csv path (alias for --csv)")
    ap.add_argument("--strategy", help="Keep only rows with this strategy (e.g., entropy_min / entropy / confidence)")
    ap.add_argument("--model", required=True, help="Model name (used to load tokenizer)")
    ap.add_argument("--output", help="Output JSONL path (legacy)")

    ap.add_argument("--out-mup", help="Output JSONL path for MUP (high-uncertainty) samples")
    ap.add_argument("--out-baseline", help="Output JSONL path for baseline (low-uncertainty) samples")
    ap.add_argument("--mup-strategy", default="entropy", help="Strategy name for MUP samples (default: entropy)")
    ap.add_argument("--baseline-strategy", default="entropy_min", help="Strategy name for baseline samples (default: entropy_min)")

    ap.add_argument("--device", default="cuda", help="Device (only for tokenizer, default: cuda)")
    return ap.parse_args()


def load_humaneval() -> Dict[str, Dict]:
    return {tid: prob for tid, prob in read_problems().items()}


def _write_samples_from_rows(rows: List[Dict], problems: Dict[str, Dict], tokenizer, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as fw:
        for row in rows:
            task_id = row["task_id"]
            if task_id not in problems:
                continue
            prob = problems[task_id]
            code = prob["prompt"] + prob["canonical_solution"]
            token_ids = tokenizer(code, return_tensors="pt").input_ids[0].tolist()
            idx = int(row["split_token_index"]) if row.get("split_token_index") else None
            if idx is None or idx + 1 >= len(token_ids):
                continue

            sample = {
                "task_id": task_id,
                "token_idx": idx,
                "code": code,
                "token_ids": token_ids,
            }
            for k in ["split_token_text", "uncertainty_metric", "uncertainty_value", "prominence"]:
                if k in row and row[k] != "":
                    sample[k] = row[k] if k != "uncertainty_value" else float(row[k])

            fw.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
    return written


def main():
    args = parse_args()
    problems = load_humaneval()

    model, tokenizer = MODEL_FACTORY[args.model](device=args.device)

    csv_path = args.split_csv or args.csv
    if not csv_path:
        raise ValueError("Either --csv/--split-csv must be provided")

    # Load rows once and dispatch depending on requested outputs
    rows: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if args.output:
        if not args.strategy:
            raise ValueError("--strategy must be provided when using --output (legacy mode)")
        matched = [r for r in rows if r.get("strategy") == args.strategy]
        written = _write_samples_from_rows(matched, problems, tokenizer, Path(args.output))
        print(f"Total rows matched strategy {args.strategy}: {len(matched)}")
        print(f"Valid samples written: {written} -> {args.output}")
        return

    # Convenience mode: write mup and/or baseline outputs
    if not args.out_mup and not args.out_baseline:
        raise ValueError("Either --output (legacy) or --out-mup/--out-baseline must be provided")

    if args.out_mup:
        mup_rows = [r for r in rows if r.get("strategy") == args.mup_strategy]
        written_mup = _write_samples_from_rows(mup_rows, problems, tokenizer, Path(args.out_mup))
        print(f"MUP: rows matched strategy {args.mup_strategy}: {len(mup_rows)}")
        print(f"MUP valid samples written: {written_mup} -> {args.out_mup}")

    if args.out_baseline:
        baseline_rows = [r for r in rows if r.get("strategy") == args.baseline_strategy]
        written_baseline = _write_samples_from_rows(baseline_rows, problems, tokenizer, Path(args.out_baseline))
        print(f"Baseline: rows matched strategy {args.baseline_strategy}: {len(baseline_rows)}")
        print(f"Baseline valid samples written: {written_baseline} -> {args.out_baseline}")


if __name__ == "__main__":
    main()
