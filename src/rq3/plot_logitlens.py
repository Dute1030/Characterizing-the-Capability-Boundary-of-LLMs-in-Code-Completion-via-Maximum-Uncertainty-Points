import json
from pathlib import Path
import matplotlib.pyplot as plt

DEFAULT_FILES = {
    "qwen3-0.6b": "/data/dt/AdaDec/experiments/rq6/logitlens/qwen3-0.6b_logitlens_agg.json",
    "qwen3-1.7b": "/data/dt/AdaDec/experiments/rq6/logitlens/qwen3-1.7b_logitlens_agg.json",
    "qwen3-4b": "/data/dt/AdaDec/experiments/rq6/logitlens/qwen3-4b_logitlens_agg.json",
}


def load_agg(path: str):
    path = Path(path)
    if not path.exists():
        return None
    data = json.load(path.open())
    return data.get("layers", {})


def plot_curves(agg_dict, out_path: Path):
    plt.figure(figsize=(12, 3.5))
    titles = ["Entropy vs Layer", "Target Rank vs Layer", "Target Prob. vs Layer"]
    ylabels = ["Entropy", "Target Rank", "Target Prob."]
    keys = ["entropy_mean", "target_rank_mean", "target_prob_mean"]

    for i, (title, ylabel, key) in enumerate(zip(titles, ylabels, keys), start=1):
        plt.subplot(1, 3, i)
        plt.title(title)
        plt.xlabel("Layer")
        plt.ylabel(ylabel)
        for name, agg in agg_dict.items():
            layers = sorted(int(k) for k in agg.keys())
            vals = [agg[str(l)][key] for l in layers]
            plt.plot(layers, vals, marker="o", label=name)
        plt.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"saved: {out_path}")
    plt.close()


def main():
    agg_dict = {}
    for name, path in DEFAULT_FILES.items():
        agg = load_agg(path)
        if agg:
            agg_dict[name] = agg
        else:
            print(f"skip (not found): {path}")

    if not agg_dict:
        print("No aggregate files found. Update DEFAULT_FILES or provide valid paths.")
        return

    out_path = Path("/data/dt/AdaDec/experiments/rq6/logitlens/logitlens_comparison.png")
    plot_curves(agg_dict, out_path)


if __name__ == "__main__":
    main()
