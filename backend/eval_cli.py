# backend/eval_cli.py
"""
评估命令行工具
用于从命令行运行 RAGAS 评估

Usage:
    python -m backend.eval_cli                        # 评估所有策略，所有问题
    python -m backend.eval_cli --strategy hybrid_rerank  # 只评估指定策略
    python -m backend.eval_cli --limit 10               # 限制评估问题数量
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import sys
import time
from backend.services.eval_service import eval_service


def main():
    """
    评估 CLI 的主函数
    解析命令行参数，加载数据集，执行评估，输出结果
    """
    parser = argparse.ArgumentParser(description="KnowRAG Evaluation CLI")
    parser.add_argument(
        "--strategy", default="all",
        choices=["all", "vector", "hybrid", "hybrid_rerank"],
        help="检索策略 (default: all)"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="限制问题数量 (0 = 全部)"
    )
    args = parser.parse_args()

    # 确定要评估的策略列表
    strategies = (
        ["vector", "hybrid", "hybrid_rerank"]
        if args.strategy == "all"
        else [args.strategy]
    )
    total = len(eval_service.load_dataset())
    count = args.limit if args.limit > 0 else total

    print(f"Dataset: {total} QA pairs, evaluating {count}")
    print(f"Strategies: {', '.join(strategies)}")
    print("-" * 60)

    # 逐个策略执行评估
    for strat in strategies:
        print(f"\n[{strat}] Running evaluation...")
        t0 = time.time()
        run_id = eval_service.run_evaluation(strategy=strat, limit=args.limit)
        elapsed = time.time() - t0

        # 获取评估结果并打印
        detail = eval_service.get_run_detail(run_id)
        print(f"  Completed in {elapsed:.1f}s")
        print(f"  Average metrics:")
        for metric in ["avg_faithfulness", "avg_context_recall", "avg_context_precision", "avg_answer_correctness", "avg_answer_accuracy"]:
            label = metric.replace("avg_", "")
            val = detail.get(metric)
            if val is not None:
                print(f"    {label}: {val:.4f}")
            else:
                print(f"    {label}: N/A")

    print("\n" + "=" * 60)
    print("Evaluation complete. View details via API: /api/eval/results")


if __name__ == "__main__":
    main()
