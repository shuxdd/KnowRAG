from fastapi import APIRouter, HTTPException
from backend.models.schemas import (
    EvalListResponse, EvalRunDetail, EvalRunInfo, EvalRunRequest,
)
from backend.services.eval_service import eval_service

# 创建 /api/eval 前缀的路由组
router = APIRouter(prefix="/api/eval", tags=["eval"])


@router.get("/results", response_model=EvalListResponse)
async def list_eval_runs():
    """
    获取评估历史列表接口（V3）

    Returns:
        所有评估运行的列表，按开始时间降序排列
    """
    runs = eval_service.get_runs()
    return EvalListResponse(runs=[EvalRunInfo(**r) for r in runs])


@router.get("/results/{run_id}", response_model=EvalRunDetail)
async def get_eval_run(run_id: str):
    """
    获取指定评估运行的详细信息（V3）

    Args:
        run_id: 评估运行的唯一 ID

    Returns:
        评估运行的详细信息，包括：
        - 运行的基本信息（策略、数据集、起止时间）
        - 平均指标（faithfulness、context_recall 等）
        - 每道题的具体评估结果

    Raises:
        HTTPException: 如果评估运行不存在，返回 404 错误
    """
    detail = eval_service.get_run_detail(run_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return EvalRunDetail(
        id=detail["id"],
        strategy=detail["strategy"],
        dataset_name=detail["dataset_name"],
        started_at=detail["started_at"],
        completed_at=detail.get("completed_at"),
        avg_faithfulness=detail.get("avg_faithfulness"),
        avg_context_recall=detail.get("avg_context_recall"),
        avg_context_precision=detail.get("avg_context_precision"),
        avg_answer_correctness=detail.get("avg_answer_correctness"),
        avg_answer_accuracy=detail.get("avg_answer_accuracy"),
        results=[
            {
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "faithfulness": r.get("faithfulness"),
                "context_recall": r.get("context_recall"),
                "context_precision": r.get("context_precision"),
                "answer_correctness": r.get("answer_correctness"),
                "answer_accuracy": r.get("answer_accuracy"),
            }
            for r in detail["results"]
        ],
    )


@router.post("/run")
async def trigger_eval(req: EvalRunRequest):
    """
    触发新的评估运行接口（V3）

    Args:
        req: 包含评估策略的请求体
             - strategy: 评估策略（all/vector/hybrid/hybrid_rerank）

    Returns:
        包含新创建的评估运行 ID
    """
    run_id = eval_service.run_evaluation(strategy=req.strategy)
    return {"run_id": run_id}
