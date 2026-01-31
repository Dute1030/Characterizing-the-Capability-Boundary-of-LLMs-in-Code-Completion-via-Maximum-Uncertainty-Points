import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.models import MODEL_FACTORY  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="RQ3: Logit Lens trajectories for MUP vs Baseline")
    ap.add_argument("--model", required=True, help="Model name (see llm.models.MODEL_FACTORY)")
    ap.add_argument("--mup-jsonl", required=True, help="High-uncertainty samples JSONL (contains token_idx and code/token_ids)")
    ap.add_argument("--baseline-jsonl", required=True, help="Low-uncertainty samples JSONL (control group)")
    ap.add_argument("--output-dir", default="experiments/rq3/logitlens", help="Output directory")
    ap.add_argument("--max-samples", type=int, default=100, help="Max samples to process per group")
    ap.add_argument("--top-k", type=int, default=5, help="Record top-k predictions")
    ap.add_argument("--device", default="cuda", help="Inference device")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-first-n", type=int, default=3, help="Print per-layer top-1 token trajectory for first n samples")
    return ap.parse_args()


def load_samples(path: Path, max_samples: Optional[int] = None) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            rows.append(json.loads(line))
    return rows


def ensure_token_ids(sample: Dict, tokenizer) -> List[int]:
    if "token_ids" in sample:
        return sample["token_ids"]
    if "code" not in sample:
        raise ValueError("Sample must have token_ids or code")
    return tokenizer(sample["code"], return_tensors="pt").input_ids[0].tolist()


def project_logits(model, hidden: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        hidden = model.model.norm(hidden)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        hidden = model.transformer.ln_f(hidden)
    return model.lm_head(hidden)


def token_rank(logits: torch.Tensor, target_id: int) -> Tuple[int, float]:
    target_logit = logits[target_id].item()
    rank = int((logits > target_logit).sum().item() + 1)
    probs = torch.softmax(logits, dim=-1)
    return rank, probs[target_id].item()


def compute_layer_metrics(
    model,
    tokenizer,
    context_ids: List[int],
    target_id: int,
    top_k: int,
    device: torch.device,
) -> List[Dict]:
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # len = n_layers + 1

    per_layer = []
    for layer_idx, h in enumerate(hidden_states):
        last_hidden = h[0, -1, :]
        logits = project_logits(model, last_hidden)
        probs = torch.softmax(logits, dim=-1)
        entropy = float(-(probs * torch.log(probs + 1e-12)).sum().item())
        rank, target_prob = token_rank(logits, target_id)
        topk_prob, topk_id = torch.topk(probs, k=min(top_k, probs.size(-1)))
        per_layer.append(
            {
                "layer": layer_idx,
                "entropy": entropy,
                "target_rank": rank,
                "target_prob": float(target_prob),
                "top1_token_id": int(topk_id[0]),
                "top1_token": tokenizer.decode([int(topk_id[0])], skip_special_tokens=True),
                "top1_prob": float(topk_prob[0]),
                "topk": [
                    {
                        "token_id": int(tok_id),
                        "token": tokenizer.decode([int(tok_id)], skip_special_tokens=True),
                        "prob": float(p),
                    }
                    for tok_id, p in zip(topk_id.tolist(), topk_prob.tolist())
                ],
            }
        )
    return per_layer


def aggregate_layers(layer_stats: List[List[Dict]]) -> Dict[int, Dict[str, float]]:
    if not layer_stats:
        return {}
    n_layers = len(layer_stats[0])
    agg: Dict[int, Dict[str, float]] = {}
    for l in range(n_layers):
        ent = [s[l]["entropy"] for s in layer_stats]
        rank = [s[l]["target_rank"] for s in layer_stats]
        prob = [s[l]["target_prob"] for s in layer_stats]
        agg[l] = {
            "entropy_mean": float(np.mean(ent)),
            "entropy_std": float(np.std(ent)),
            "target_rank_mean": float(np.mean(rank)),
            "target_rank_std": float(np.std(rank)),
            "target_prob_mean": float(np.mean(prob)),
            "target_prob_std": float(np.std(prob)),
        }
    return agg


def clean_tok(tok: str) -> str:
    tok = tok.replace("\n", "\\n")
    if tok == "":
        tok = "<blank>"
    return tok


def process_group(
    group_name: str,
    samples: List[Dict],
    model,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[Path, Path]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_path = out_dir / f"{group_name}_{args.model}_logitlens_samples.jsonl"
    layer_stats_all: List[List[Dict]] = []

    with detail_path.open("w", encoding="utf-8") as fw:
        for i, sample in enumerate(tqdm(samples, desc=f"{group_name} samples")):
            token_ids = ensure_token_ids(sample, tokenizer)
            idx = sample["token_idx"]
            if idx + 1 >= len(token_ids):
                continue
            context_ids = token_ids[: idx + 1]
            target_id = token_ids[idx + 1]

            per_layer = compute_layer_metrics(
                model=model,
                tokenizer=tokenizer,
                context_ids=context_ids,
                target_id=target_id,
                top_k=args.top_k,
                device=device,
            )
            layer_stats_all.append(per_layer)

            record = {
                "token_idx": idx,
                "target_id": target_id,
                "target_token": tokenizer.decode([target_id], skip_special_tokens=True),
                "layers": per_layer,
            }
            for k in ["entropy", "confidence", "ppl", "task_id"]:
                if k in sample:
                    record[k] = sample[k]
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")

            if i < args.print_first_n:
                top1_tokens = [clean_tok(l["top1_token"]) for l in per_layer]
                top1_probs = [f"{l['top1_prob']:.3f}" for l in per_layer]
                print(f"\n[{group_name}] sample #{i} task={sample.get('task_id', 'NA')} idx={idx}")
                print("Top1 tokens per layer:")
                print("  " + " | ".join(top1_tokens))
                print("Top1 probs per layer:")
                print("  " + " | ".join(top1_probs))
                print(f"GT token: {clean_tok(record['target_token'])}")

    agg = aggregate_layers(layer_stats_all)
    agg_path = out_dir / f"{group_name}_{args.model}_logitlens_agg.json"
    agg_path.write_text(json.dumps({"model": args.model, "group": group_name, "layers": agg}, ensure_ascii=False, indent=2))

    print(f"[{group_name}] Saved per-sample to {detail_path}")
    print(f"[{group_name}] Saved agg to {agg_path}")
    return detail_path, agg_path


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, tokenizer = MODEL_FACTORY[args.model](device=args.device)
    model.eval()
    device = next(model.parameters()).device

    mup_samples = load_samples(Path(args.mup_jsonl), args.max_samples)
    base_samples = load_samples(Path(args.baseline_jsonl), args.max_samples)

    process_group("mup", mup_samples, model, tokenizer, device, args)
    process_group("baseline", base_samples, model, tokenizer, device, args)


if __name__ == "__main__":
    main()
