# -*- coding: utf-8 -*-
"""
Generate low-uncertainty samples for logit lens analysis.
Method: Under teacher forcing on the dataset (HumanEval), select the token(s) with the lowest entropy per problem (or top k).
Outputs a JSONL with fields:
- task_id
- token_idx       (index of the uncertain token, based on token_ids[1:], zero-based)
- code            (prompt + canonical_solution)
- entropy         (entropy of that token)
- token_ids       (full input sequence, including BOS/EOS)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import logging

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.models import MODEL_FACTORY  # noqa: E402
from src.rq1.uncertainty_calculator import UncertaintyCalculator  # noqa: E402

# Compatibility for human_eval reading
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
    ap = argparse.ArgumentParser(description="Prepare low-uncertainty samples for logit lens")
    ap.add_argument("--model", required=True, help="Model name (see llm.models.MODEL_FACTORY)")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--k", type=int, default=1, help="Take the k tokens with lowest entropy per problem")
    ap.add_argument("--max-samples", type=int, default=None, help="Maximum number of problems to process (default: all)")
    ap.add_argument("--device", default="cuda", help="Device")
    ap.add_argument("--verbose", action="store_true", help="Log the lowest-entropy positions per sample")
    return ap.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    problems = list(read_problems().items())
    if args.max_samples:
        problems = problems[: args.max_samples]
    logger.info(f"Total problems: {len(problems)}")

    logger.info(f"Loading model {args.model} on {args.device} ...")
    model, tokenizer = MODEL_FACTORY[args.model](device=args.device)
    calc = UncertaintyCalculator(model, tokenizer, device=args.device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    logger.info(f"Writing samples to {out_path}")
    with out_path.open("w", encoding="utf-8") as fw:
        for task_id, prob in tqdm(problems, desc="Collecting low-uncertainty tokens"):
            code = prob["prompt"] + prob["canonical_solution"]
            unc = calc.compute_uncertainties_for_code(code, return_tokens=True)
            ent = np.array(unc["entropies"])
            token_ids = tokenizer(code, return_tensors="pt").input_ids[0].tolist()

            idxs = np.argsort(ent)[: args.k]
            if args.verbose:
                msg = f"{task_id}: min_entropy_idx={idxs[0]}, val={ent[idxs[0]]:.4f}"
                logger.info(msg)
            for idx in idxs:
                sample = {
                    "task_id": task_id,
                    "token_idx": int(idx),
                    "code": code,
                    "entropy": float(ent[idx]),
                    "token_ids": token_ids,
                }
                fw.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total += 1

    logger.info(f"Saved {total} samples to {out_path}")


if __name__ == "__main__":
    main()
