# -*- coding: utf-8 -*-
"""
RQ6 可视化：高不确定 vs 低不确定的层级熵轨迹 + Rank 热力图。

输入：
- 高不确定样本的 logit lens 明细 JSONL（来自 logit_lens_analysis.py 的 *_logitlens_samples.jsonl）
- 低不确定样本的 logit lens 明细 JSONL（同格式，使用 prepare_low_uncertainty_samples 生成后再跑 logit_lens_analysis）

输出：
- entropy 曲线（高 vs 低）：experiments/rq6/figs/entropy_evolution.png
- rank 热力图（高组样本）：experiments/rq6/figs/rank_heatmap_high.png

运行示例：
python3 rq6/rq6_analysis.py \
  --high-jsonl /data/dt/AdaDec/experiments/rq6/logitlens/qwen3-0.6b_logitlens_samples.jsonl \
  --low-jsonl  /data/dt/AdaDec/experiments/rq6/logitlens_low/qwen3-0.6b_low_uncertain_logitlens_samples.jsonl \
  --output-dir /data/dt/AdaDec/experiments/rq6/figs
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description="RQ6: logit lens 可视化（熵轨迹 + rank 热力图）")
    ap.add_argument("--high-jsonl", required=True, help="高不确定样本 logit lens JSONL")
    ap.add_argument("--low-jsonl", required=True, help="低不确定样本 logit lens JSONL")
    ap.add_argument("--output-dir", required=True, help="输出目录")
    ap.add_argument("--max-samples-heatmap", type=int, default=100, help="热力图最多展示的高不确定样本数")
    return ap.parse_args()


def load_samples(path: Path, max_rows: int = None) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            rows.append(json.loads(line))
    return rows


def layer_stats(samples: List[Dict]) -> Dict[int, Dict[str, float]]:
    """
    计算每层的平均 entropy/rank/prob
    """
    if not samples:
        return {}
    n_layers = len(samples[0]["layers"])
    agg = {}
    for l in range(n_layers):
        ent = []
        rank = []
        prob = []
        for s in samples:
            layer = s["layers"][l]
            ent.append(layer.get("entropy", np.nan))
            rank.append(layer.get("target_rank", np.nan))
            prob.append(layer.get("target_prob", np.nan))
        ent = np.nan_to_num(ent, nan=np.nan)
        rank = np.nan_to_num(rank, nan=np.nan)
        prob = np.nan_to_num(prob, nan=np.nan)
        agg[l] = {
            "entropy_mean": float(np.nanmean(ent)),
            "rank_mean": float(np.nanmean(rank)),
            "prob_mean": float(np.nanmean(prob)),
        }
    return agg


def plot_entropy(high_stats: Dict[int, Dict[str, float]], low_stats: Dict[int, Dict[str, float]], out_path: Path):
    layers = sorted(high_stats.keys())
    high_ent = [high_stats[l]["entropy_mean"] for l in layers]
    low_ent = [low_stats[l]["entropy_mean"] for l in layers]

    plt.figure(figsize=(6, 4))
    plt.plot(layers, high_ent, marker="o", label="High-uncertainty")
    plt.plot(layers, low_ent, marker="o", label="Low-uncertainty")
    plt.xlabel("Layer")
    plt.ylabel("Entropy (mean)")
    plt.title("Layer-wise Entropy Evolution")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved: {out_path}")


def plot_rank_heatmap(high_samples: List[Dict], out_path: Path, max_rows: int):
    # 取前 max_rows 个样本
    samples = high_samples[:max_rows]
    if not samples:
        print("No high-uncertainty samples to plot heatmap.")
        return
    n_layers = len(samples[0]["layers"])
    mat = np.zeros((len(samples), n_layers))
    for i, s in enumerate(samples):
        ranks = [layer.get("target_rank", np.nan) for layer in s["layers"]]
        mat[i, :] = ranks

    plt.figure(figsize=(10, max(4, len(samples) * 0.15)))
    im = plt.imshow(np.log1p(mat), aspect="auto", cmap="magma")
    plt.colorbar(im, label="log(1 + rank)")
    plt.xlabel("Layer")
    plt.ylabel("Sample (high uncertainty)")
    plt.title("Ground Truth Rank Heatmap (High-uncertainty)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved: {out_path}")


def main():
    args = parse_args()
    high_path = Path(args.high_jsonl)
    low_path = Path(args.low_jsonl)
    out_dir = Path(args.output_dir)

    high_samples = load_samples(high_path)
    low_samples = load_samples(low_path)

    high_stats = layer_stats(high_samples)
    low_stats = layer_stats(low_samples)

    plot_entropy(
        high_stats,
        low_stats,
        out_dir / "entropy_evolution.png",
    )
    plot_rank_heatmap(
        high_samples,
        out_dir / "rank_heatmap_high.png",
        args.max_samples_heatmap,
    )


if __name__ == "__main__":
    main()
