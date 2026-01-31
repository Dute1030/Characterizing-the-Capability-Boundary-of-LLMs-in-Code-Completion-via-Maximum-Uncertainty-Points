#!/usr/bin/env bash
# Generate low-uncertainty samples and run logit lens (0.6B / 1.7B / 4B)
# Ensure transformers can load Qwen3; update or specify local checkpoints if needed.

set -euo pipefail
set -x  # print commands

MODELS=("qwen3-0.6b" "qwen3-1.7b" "qwen3-4b")
MAX_SAMPLES=164
K=1
DEVICE="cuda"

BASE=/data/dt/AdaDec/experiments/rq6

for m in "${MODELS[@]}"; do
  echo "=== ${m}: prepare low-uncertainty samples ==="
  python3 rq6/prepare_low_uncertainty_samples.py \
    --model "$m" \
    --output "${BASE}/${m}_low_uncertain.jsonl" \
    --k ${K} \
    --max-samples ${MAX_SAMPLES} \
    --device ${DEVICE}

  echo "=== ${m}: logit lens analysis ==="
  python3 rq6/logit_lens_analysis.py \
    --model "$m" \
    --input "${BASE}/${m}_low_uncertain.jsonl" \
    --output "${BASE}/logitlens_low" \
    --device ${DEVICE}
done

echo "Done. Outputs in ${BASE} and ${BASE}/logitlens_low"
