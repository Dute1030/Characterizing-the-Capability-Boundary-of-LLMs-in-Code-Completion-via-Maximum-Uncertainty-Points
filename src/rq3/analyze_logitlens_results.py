# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description="Analyze RQ3 logit lens outputs (MUP vs Baseline)")
    ap.add_argument("--mup-jsonl", required=True, help="mup_*_logitlens_samples.jsonl")
    ap.add_argument("--baseline-jsonl", required=True, help="baseline_*_logitlens_samples.jsonl")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-label", default="")
    ap.add_argument("--target-topk", type=int, default=10)
    return ap.parse_args()


def load_samples(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def agg_layers(samples: List[Dict]) -> Dict[int, Dict[str, float]]:
    if not samples:
        return {}
    n_layers = len(samples[0]["layers"])
    agg: Dict[int, Dict[str, float]] = {}
    for l in range(n_layers):
        ent = [s["layers"][l]["entropy"] for s in samples]
        rank = [s["layers"][l]["target_rank"] for s in samples]
        prob = [s["layers"][l]["target_prob"] for s in samples]
        agg[l] = {
            "entropy_mean": float(np.mean(ent)),
            "target_rank_mean": float(np.mean(rank)),
            "target_prob_mean": float(np.mean(prob)),
        }
    return agg


def plot_metric(layers: List[int], mup_vals: List[float], base_vals: List[float], ylabel: str, title: str, out_path: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(layers, mup_vals, marker="o", label="MUP")
    plt.plot(layers, base_vals, marker="o", label="Baseline")
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved {out_path}")


def top1_change_stats(samples: List[Dict], target_topk: int) -> Dict[str, float]:
    changes = []
    enter_layers: List[int] = []
    top1_hit = 0
    total = 0
    for s in samples:
        tops = [layer["top1_token"] for layer in s["layers"]]
        changes.append(sum(1 for i in range(1, len(tops)) if tops[i] != tops[i - 1]))
        layer_entered: Optional[int] = None
        for l, layer in enumerate(s["layers"]):
            if layer["target_rank"] <= target_topk:
                layer_entered = l
                break
        if layer_entered is not None:
            enter_layers.append(layer_entered)
        for layer in s["layers"]:
            total += 1
            if layer["target_rank"] == 1:
                top1_hit += 1
    stats = {
        "top1_changes_mean": float(np.mean(changes)) if changes else 0.0,
        "top1_changes_median": float(np.median(changes)) if changes else 0.0,
        "target_enter_topk_mean_layer": float(np.mean(enter_layers)) if enter_layers else -1,
        "target_enter_topk_hit_rate": float(len(enter_layers) / len(samples)) if samples else 0.0,
        "target_top1_fraction": float(top1_hit / total) if total > 0 else 0.0,
    }
    return stats


def series(agg: Dict[int, Dict[str, float]], key: str, layers: List[int]) -> List[float]:
    return [agg[l][key] for l in layers]


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mup_samples = load_samples(Path(args.mup_jsonl))
    base_samples = load_samples(Path(args.baseline_jsonl))

    agg_mup = agg_layers(mup_samples)
    agg_base = agg_layers(base_samples)
    layers = sorted(agg_mup.keys())
    metrics = [
        ("entropy_mean", "Entropy"),
        ("target_rank_mean", "GT Rank"),
        ("target_prob_mean", "GT Prob"),
    ]
    label = f" {args.model_label}" if args.model_label else ""
    for key, title in metrics:
        plot_metric(
            layers,
            series(agg_mup, key, layers),
            series(agg_base, key, layers),
            ylabel=title,
            title=f"MUP vs Baseline{label}: {title}",
            out_path=out_dir / f"{key}.png",

    stats = {
        "mup": top1_change_stats(mup_samples, args.target_topk),
        "baseline": top1_change_stats(base_samples, args.target_topk),
        "target_topk_threshold": args.target_topk,
        "model_label": args.model_label,
    }
    stats_path = out_dir / "summary.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"saved {stats_path}")


if __name__ == "__main__":
    main()
