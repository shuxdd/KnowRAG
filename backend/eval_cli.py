# backend/eval_cli.py
"""Run RAGAS evaluation from command line.
Usage:
    python -m backend.eval_cli                        # all strategies, all questions
    python -m backend.eval_cli --strategy hybrid_rerank
    python -m backend.eval_cli --limit 10
"""
import argparse
import sys
import time
from backend.services.eval_service import eval_service


def main():
    parser = argparse.ArgumentParser(description="KnowRAG Evaluation CLI")
    parser.add_argument(
        "--strategy", default="all",
        choices=["all", "vector", "hybrid", "hybrid_rerank"],
        help="Retrieval strategy to evaluate (default: all)"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of questions (0 = all)"
    )
    args = parser.parse_args()

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

    for strat in strategies:
        print(f"\n[{strat}] Running evaluation...")
        t0 = time.time()
        run_id = eval_service.run_evaluation(strategy=strat, limit=args.limit)
        elapsed = time.time() - t0

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
