"""
评估命令行工具

用于在终端运行 RAG 系统评估。

使用方法：
    python -m backend.eval_cli --strategy all --limit 10

参数：
    --strategy: 评估策略（all/vector/hybrid/hybrid_rerank/fast/precise/deep/auto）
                - all: 评估所有策略（fast, precise, deep）
                - 其他: 评估指定策略
    --limit: 限制评估的问题数量（0 = 全部）

输出：
    - 每个策略的平均指标（faithfulness, context_recall, context_precision, answer_relevancy）
    - 评估完成后可通过 API 查看详细结果：/api/eval/results
"""

import argparse
import os
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from backend.utils.logging import setup_logging
setup_logging()

from backend.services.eval_service import eval_service


def main():
    parser = argparse.ArgumentParser(description="KnowRAG Evaluation CLI")
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["all", "vector", "hybrid", "hybrid_rerank", "fast", "precise", "deep", "auto"],
        help="retrieval strategy (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="limit question count (0 = all)",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=1,
        help="user ID for document isolation (default: 1)",
    )
    args = parser.parse_args()

    strategies = ["fast", "precise", "deep"] if args.strategy == "all" else [args.strategy]
    total = len(eval_service.load_dataset())
    count = args.limit if args.limit > 0 else total

    print(f"Dataset: {total} QA pairs, evaluating {count}")
    print(f"Strategies: {', '.join(strategies)}")
    print("-" * 60)

    for strategy in strategies:
        print(f"\n[{strategy}] Running evaluation...")
        started = time.time()
        run_id = eval_service.run_evaluation(strategy=strategy, limit=args.limit, user_id=args.user_id)
        elapsed = time.time() - started
        detail = eval_service.get_run_detail(run_id, user_id=args.user_id)

        print(f"  Completed in {elapsed:.1f}s")
        print(f"  Status: {detail.get('status')}")
        if detail.get("error_message"):
            print(f"  Error: {detail['error_message']}")
        print("  Average metrics:")
        for metric in [
            "avg_faithfulness",
            "avg_context_recall",
            "avg_context_precision",
            "avg_answer_relevancy",
        ]:
            label = metric.replace("avg_", "")
            value = detail.get(metric)
            print(f"    {label}: {value:.4f}" if value is not None else f"    {label}: N/A")

    print("\n" + "=" * 60)
    print("Evaluation complete. View details via API: /api/eval/results")


if __name__ == "__main__":
    main()
