import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from human_eval.human_eval.data import read_problems
except ImportError:
    import importlib.util

    data_py = ROOT / "data" / "human_eval" / "human_eval" / "data.py"
    spec = importlib.util.spec_from_file_location("human_eval_data", data_py)
    human_eval_data = importlib.util.module_from_spec(spec) if spec else None
    if not spec or not spec.loader:
        raise
    spec.loader.exec_module(human_eval_data) 
    read_problems = human_eval_data.read_problems  

def parse_args():
    ap = argparse.ArgumentParser(description="Convert split_positions CSV to logitlens JSONL")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-rows", type=int, default=None)
    return ap.parse_args()


def load_humaneval() -> Dict[str, Dict]:
    return read_problems()


def main():
    args = parse_args()
    problems = load_humaneval()

    rows: List[Dict] = []
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if args.max_rows and i >= args.max_rows:
                break
            if args.strategy and r.get("strategy") != args.strategy:
                continue
            rows.append(r)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fw:
        for r in rows:
            task_id = r["task_id"]
            if task_id not in problems:
                continue
            code = problems[task_id]["prompt"] + problems[task_id]["canonical_solution"]
            token_idx = int(r["split_token_index"])
            sample = {"task_id": task_id, "token_idx": token_idx, "code": code}
            sample.update(r)
            fw.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Converted {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
