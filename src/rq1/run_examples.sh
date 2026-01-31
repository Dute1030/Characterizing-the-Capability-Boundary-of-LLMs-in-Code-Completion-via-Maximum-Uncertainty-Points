#!/bin/bash
# RQ1 experiment run examples

echo "======================================"
echo "RQ1 Experiment - Example Runs"
echo "======================================"
echo ""

# 1. Quick demo (single sample)
echo "Running quick demo..."
echo "python src/rq1/quick_example.py"
echo ""

# 2. Small-scale experiment (test)
echo "Small-scale experiment (10 samples, single model):"
echo "python -m src.rq1.run_experiment \\"
echo "    --models qwen3-0.6b \\"
echo "    --dataset humaneval \\"
echo "    --num_samples 10 \\"
echo "    --output_dir experiments/rq1_test"
echo ""

# 3. Medium-scale experiment
echo "Medium-scale experiment (50 samples, single model):"
echo "python -m src.rq1.run_experiment \\"
echo "    --models qwen3-0.6b \\"
echo "    --dataset humaneval \\"
echo "    --num_samples 50 \\"
echo "    --output_dir experiments/rq1_medium"
echo ""

# 4. Full experiment (all 164 samples, single model)
echo "Full experiment - Single model (all 164 samples):"
echo "python -m src.rq1.run_experiment \\"
echo "    --models qwen3-0.6b \\"
echo "    --dataset humaneval \\"
echo "    --output_dir experiments/rq1_full_single"
echo ""

# 5. Full experiment (all 164 samples, 3 models)
echo "Full experiment - All models (all 164 samples, 3 models):"
echo "python -m src.rq1.run_experiment \\"
echo "    --models qwen3-0.6b qwen3-1.7b qwen3-4b \\"
echo "    --dataset humaneval \\"
echo "    --output_dir experiments/rq1_full_results"
echo ""

# 6. Visualization only
echo "Generate visualization only (based on existing results):"
echo "python -m src.rq1.run_experiment \\"
echo "    --visualize_only \\"
echo "    --output_dir experiments/rq1_results"
echo ""

echo "======================================"
echo "Choose a command to run, or customize parameters"
echo ""
echo "Notes:"
echo "  - The full experiment (164 samples) requires a long time"
echo "  - It's recommended to use tmux or screen to run in background"
echo "  - You can run different models separately and merge results later"
echo "======================================"
