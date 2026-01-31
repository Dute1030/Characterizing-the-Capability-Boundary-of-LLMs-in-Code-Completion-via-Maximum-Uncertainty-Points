import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.models import MODEL_FACTORY  


def parse_args():
    ap = argparse.ArgumentParser(description="RQ6: Logit Lens on uncertain tokens")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="experiments/rq6_logitlens")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_samples(path: Path, max_samples: Optional[int] = None) -> List[Dict]:
    samples: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            obj = json.loads(line)
            samples.append(obj)
    return samples


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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, tokenizer = MODEL_FACTORY[args.model](device=args.device)
    model.eval()
    device = next(model.parameters()).device

    samples = load_samples(Path(args.input), args.max_samples)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_stats_all: List[List[Dict]] = []
    detail_path = out_dir / f"{args.model}_logitlens_samples.jsonl"

    with detail_path.open("w", encoding="utf-8") as fw:
        for sample in tqdm(samples, desc="Logit lens on samples"):
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

    agg = aggregate_layers(layer_stats_all)
    agg_path = out_dir / f"{args.model}_logitlens_agg.json"
    agg_path.write_text(json.dumps({"model": args.model, "layers": agg}, ensure_ascii=False, indent=2))

    print(f"Saved per-sample logit lens to {detail_path}")
    print(f"Saved aggregated layer stats to {agg_path}")


if __name__ == "__main__":
    main()
