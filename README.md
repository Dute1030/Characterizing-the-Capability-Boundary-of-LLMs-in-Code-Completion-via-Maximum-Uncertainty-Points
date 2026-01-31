# MUP Repository

This repository contains the code and analysis tools for the MUP project ("Characterizing the Capability Boundary of LLMs in Code Completion via Maximum Uncertainty Points"). It includes data preparation, experiment runners, and analysis utilities for RQ1, RQ2, and RQ3. This README provides a quick-start guide, dependency notes, and example commands for running the RQ1–RQ3 experiments and analyses.

---

## Project layout 🔧
- `src/rq1/` — RQ1: uncertainty split detection, model comparison, and visualization (includes `run_experiment.py`, `quickstart.sh`, etc.)
- `src/rq2/` — RQ2: mapping split points to AST node types and aggregated statistics (`token_category_analysis.py`, `run_rq2_pipeline.py`)
- `src/rq3/` — RQ3: logit lens layer-wise trajectories and uncertainty vs. baseline analysis (`logit_lens_rq3.py`, `prepare_*` scripts)
- `data/`, `results/` — datasets and experiment outputs

---

## Quickstart & Dependencies 💡
1. We recommend using a Conda environment (example):

```bash
conda create -n mup python=3.9 -y
conda activate mup
pip install -r src/rq1/requirements.txt
pip install tree_sitter_languages
```

2. GPU recommendation: NVIDIA GPU with 16GB+ VRAM is recommended for large models. Use the appropriate CUDA-enabled PyTorch build.

---

## RQ1 — Training / Evaluation (Uncertainty split detection) ✅
Goal: Identify split points using multiple strategies (random/entropy/confidence/ppl) and compare model generation performance.

Main scripts:
- `src/rq1/run_experiment.py` — main experiment driver
- `src/rq1/quickstart.sh` / `src/rq1/run_full_experiment.sh` — convenience run scripts

Example usage:

```bash
python3 -m src.rq1.run_experiment \
  --models qwen3-0.6b qwen3-1.7b qwen3-4b \
  --dataset humaneval \
  --output_dir experiments/rq1_full_results \
  --device cuda
```

Important outputs:
- `*_split_positions.csv`: split point info per task (used as RQ2/RQ3 input)
- `all_results.json` / `results_table.csv`: aggregated metrics (CodeBLEU, Exact Match, ROUGE, etc.)

---

## RQ2 — AST node mapping & category statistics 🧭
Goal: Map RQ1 split points to AST node types and aggregate into high-level categories (control/definition/call/…).

Main scripts:
- `src/rq2/token_category_analysis.py` — analyze a single model/run
- `src/rq2/run_rq2_pipeline.py` — batch pipeline for multiple models

Dependencies:
- `tree_sitter_languages` (for AST parsing)
- Model tokenizers via `transformers.AutoTokenizer` (e.g., Qwen series)

Example usage:

```bash
# Single-model analysis
python3 src/rq2/token_category_analysis.py \
  --run_dir experiments/rq1_full_results \
  --model qwen3-0.6b \
  --output_csv experiments/rq2_qwen3-0.6b_token_node_categories.csv

# Batch processing for multiple models
python3 src/rq2/run_rq2_pipeline.py \
  --run_dir experiments/rq1_full_results \
  --models qwen3-0.6b qwen3-1.7b qwen3-4b \
  --output_root experiments/rq2_results
```

Notes:
- If tree-sitter imports fail, ensure `tree_sitter` and `tree_sitter_languages` are installed and a parser is available.
- Output CSVs include `task_id`, `strategy`, `split_token_index`, `node_type`, `category`, etc., for downstream plotting and analysis.

---

## RQ3 — Logit Lens layer-wise analysis 🔬
Goal: For uncertain tokens (MUP) and low-uncertainty baseline samples, use logit lens to record model predictions layer-by-layer and compare aggregated behavior.

Main scripts:
- `src/rq3/prepare_samples_from_split_csv.py`, `prepare_logitlens_samples.py`, `prepare_low_uncertainty_samples.py` — generate RQ3 JSONL inputs from RQ1 split files
- `src/rq3/logit_lens_rq3.py` — main analysis script that records top-k/top1, entropy, target_rank, target_prob per layer
- `src/rq3/plot_logitlens.py`, `src/rq3/analyze_logitlens_results.py` — visualization and aggregation tools

End-to-end example:

1) Build RQ3 inputs from a split CSV (generate MUP and baseline JSONL):

```bash
python3 src/rq3/prepare_samples_from_split_csv.py \
  --split-csv experiments/rq1_full_results/qwen3-0.6b_split_positions.csv \
  --model qwen3-0.6b \
  --out-mup experiments/rq3/mup_qwen3-0.6b_mup.jsonl \
  --out-baseline experiments/rq3/mup_qwen3-0.6b_baseline.jsonl
```

2) Run Logit Lens analysis:

```bash
python3 -m src.rq3.logit_lens_rq3 \
  --model qwen3-0.6b \
  --mup-jsonl experiments/rq3/mup_qwen3-0.6b_mup.jsonl \
  --baseline-jsonl experiments/rq3/mup_qwen3-0.6b_baseline.jsonl \
  --output-dir experiments/rq3/logitlens \
  --max-samples 200 \
  --top-k 5 \
  --device cuda
```

3) Visualize / aggregate:

```bash
python3 src/rq3/plot_logitlens.py --input experiments/rq3/logitlens --output experiments/rq3/plots
```

---

**Quick reference: key scripts**
- `src/rq1/run_experiment.py`
- `src/rq2/token_category_analysis.py`, `src/rq2/run_rq2_pipeline.py`
- `src/rq3/logit_lens_rq3.py`, `src/rq3/prepare_samples_from_split_csv.py`

---
