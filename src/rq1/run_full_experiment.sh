#!/bin/bash
# Run full RQ1 experiment - all 164 HumanEval samples, 3 models


if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Cancelled"
    exit 0
fi

echo ""
echo "▶ Starting experiment..."
echo ""

# Change to project root
cd /data/dt/AdaDec

# Run experiment
python -m src.rq1.run_experiment \
    --models qwen3-0.6b qwen3-1.7b qwen3-4b \
    --dataset humaneval \
    --output_dir experiments/rq1_full_results \
    --log_file experiments/rq1_full_results/experiment.log

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                      Experiment completed!                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Result files:"
echo "  - Log: experiments/rq1_full_results/experiment.log"
echo "  - Data: experiments/rq1_full_results/all_results.json"
echo "  - Table: experiments/rq1_full_results/results_table.csv"
echo "  - Plots: experiments/rq1_full_results/*.png"
echo ""
echo "📖 View summary:"
echo "  cat experiments/rq1_full_results/results_table.csv"
echo ""
