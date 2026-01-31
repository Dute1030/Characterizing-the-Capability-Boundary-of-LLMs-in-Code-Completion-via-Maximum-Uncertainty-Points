#!/bin/bash
# RQ1实验快速启动脚本
# Quick Start Script for RQ1 Experiment

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║       RQ1: Maximum Uncertainty Point Impact Experiment        ║"
echo "║              最大不确定性点对模型性能影响实验                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 检测Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python版本: $(python --version)"
echo ""

# 检查当前目录
if [ ! -f "src/rq1/run_experiment.py" ]; then
    echo "⚠️  Warning: 请在项目根目录 /data/dt/AdaDec 运行此脚本"
    echo "   cd /data/dt/AdaDec && bash src/rq1/quickstart.sh"
    exit 1
fi

echo "📂 当前目录: $(pwd)"
echo ""

# 菜单选择
echo "请选择运行模式："
echo ""
echo "  [1] 快速示例 (Quick Example)"
echo "      - 单个代码样本演示"
echo "      - 运行时间: ~1-2分钟"
echo "      - 适合: 第一次使用，了解流程"
echo ""
echo "  [2] 小规模测试 (Small Test)"
echo "      - 10个HumanEval样本"
echo "      - 单个模型: qwen3-0.6b"
echo "      - 运行时间: ~5-10分钟"
echo "      - 适合: 验证环境配置"
echo ""
echo "  [3] 中等规模实验 (Medium Experiment)"
echo "      - 50个HumanEval样本"
echo "      - 单个模型: qwen3-0.6b"
echo "      - 运行时间: ~30-60分钟"
echo "      - 适合: 初步实验结果"
echo ""
echo "  [4] 完整实验 - 全部模型 (Full Experiment - All Models)"
echo "      - 全部164个HumanEval样本"
echo "      - 三个模型: 0.6b, 1.7b, 4b"
echo "      - 运行时间: ~4-8小时"
echo "      - 适合: 完整的研究实验（RQ1最终结果）"
echo ""
echo "  [5] 完整实验 - 单模型 (Full Experiment - Single Model)"
echo "      - 全部164个HumanEval样本"
echo "      - 单个模型: qwen3-0.6b"
echo "      - 运行时间: ~2-3小时"
echo "      - 适合: 单模型深度分析"
echo ""
echo "  [6] 仅生成可视化 (Visualize Only)"
echo "      - 从已有结果生成图表"
echo "      - 运行时间: <1分钟"
echo ""
echo "  [0] 退出"
echo ""

read -p "请输入选项 [0-6]: " choice

case $choice in
    1)
        echo ""
        echo "▶ 运行快速示例..."
        python src/rq1/quick_example.py
        ;;
    2)
        echo ""
        echo "▶ 运行小规模测试..."
        python -m src.rq1.run_experiment \
            --models qwen3-0.6b \
            --dataset humaneval \
            --num_samples 10 \
            --output_dir experiments/rq1_test
        ;;
    3)
        echo ""
        echo "▶ 运行中等规模实验（单模型）..."
        python -m src.rq1.run_experiment \
            --models qwen3-0.6b \
            --dataset humaneval \
            --num_samples 50 \
            --output_dir experiments/rq1_medium
        ;;
    4)
        echo ""
        echo "▶ 运行完整实验（全部164个HumanEval样本，3个模型）..."
        echo "⚠️  这将需要较长时间（约4-8小时），强烈建议使用 tmux/screen"
        read -p "确认继续? (y/n): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python -m src.rq1.run_experiment \
                --models qwen3-0.6b qwen3-1.7b qwen3-4b \
                --dataset humaneval \
                --output_dir experiments/rq1_full_results
        else
            echo "已取消"
        fi
        ;;
    5)
        echo ""
        echo "▶ 运行完整实验（全部164个样本，单模型）..."
        echo "⚠️  这将需要约2-3小时"
        read -p "确认继续? (y/n): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python -m src.rq1.run_experiment \
                --models qwen3-0.6b \
                --dataset humaneval \
                --output_dir experiments/rq1_full_single
        else
            echo "已取消"
        fi
        ;;
    6)
        echo ""
        echo "▶ 生成可视化..."
        if [ -f "experiments/rq1_results/all_results.json" ]; then
            python -m src.rq1.run_experiment \
                --visualize_only \
                --output_dir experiments/rq1_results
        else
            echo "❌ 未找到结果文件: experiments/rq1_results/all_results.json"
            echo "   请先运行实验"
        fi
        ;;
    0)
        echo "退出"
        exit 0
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                      实验完成！                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 查看结果:"
echo "   - 日志: experiments/rq1_*/experiment.log"
echo "   - 数据: experiments/rq1_*/all_results.json"
echo "   - 表格: experiments/rq1_*/results_table.csv"
echo "   - 图表: experiments/rq1_*/*.png"
echo ""
echo "📖 文档:"
echo "   - README: src/rq1/README.md"
echo "   - 使用指南: src/rq1/USAGE_GUIDE.txt"
echo "   - 项目总结: src/rq1/PROJECT_SUMMARY.md"
echo ""
