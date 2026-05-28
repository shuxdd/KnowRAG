"""
评估路由模块

提供 RAG 系统评估的接口，支持评估检索策略的效果。

接口列表：
- GET /api/eval/results: 获取评估历史列表
- GET /api/eval/results/{run_id}: 获取评估运行的详细信息
- DELETE /api/eval/results/{run_id}: 删除评估运行记录
- POST /api/eval/run: 触发新的评估运行

评估指标：
- faithfulness: 答案忠诚度（答案与上下文的吻合程度）
- context_recall: 上下文召回率（上下文覆盖真实答案的程度）
- context_precision: 上下文精确度（上下文相关内容的比例）
- answer_relevancy: 答案相关性（答案与问题的相关程度）

评估流程：
1. 加载测试数据集（QA 对）
2. 对每个问题执行检索和问答
3. 使用 RAGAS 框架计算各项指标
4. 存储评估结果
"""

from fastapi import APIRouter, HTTPException, Depends
from backend.models.schemas import (
    EvalListResponse, EvalRunDetail, EvalRunInfo, EvalRunRequest,
)
from backend.services.eval_service import eval_service
from backend.utils.auth import get_current_user, CurrentUser

router = APIRouter(prefix="/api/eval", tags=["eval"])


@router.get("/results", response_model=EvalListResponse)
async def list_eval_runs(
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    获取评估历史列表接口（V3）

    Returns:
        所有评估运行的列表，按开始时间降序排列
    """
    runs = eval_service.get_runs(user_id=current_user.id)
    return EvalListResponse(runs=[EvalRunInfo(**r) for r in runs])


@router.get("/results/{run_id}", response_model=EvalRunDetail)
async def get_eval_run(
    run_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
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
    detail = eval_service.get_run_detail(run_id, user_id=current_user.id)
    if not detail:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return EvalRunDetail(
        id=detail["id"],
        strategy=detail["strategy"],
        dataset_name=detail["dataset_name"],
        status=detail["status"],
        started_at=detail["started_at"],
        completed_at=detail.get("completed_at"),
        error_message=detail.get("error_message"),
        avg_faithfulness=detail.get("avg_faithfulness"),
        avg_context_recall=detail.get("avg_context_recall"),
        avg_context_precision=detail.get("avg_context_precision"),
        avg_answer_relevancy=detail.get("avg_answer_relevancy"),
        results=[
            {
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "answer": r["answer"],
                "strategy": r.get("strategy"),
                "contexts": r["contexts"],
                "faithfulness": r.get("faithfulness"),
                "context_recall": r.get("context_recall"),
                "context_precision": r.get("context_precision"),
                "answer_relevancy": r.get("answer_relevancy"),
            }
            for r in detail["results"]
        ],
    )


@router.delete("/results/{run_id}")
async def delete_eval_run(
    run_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    deleted = eval_service.delete_run(run_id, user_id=current_user.id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return {"detail": f"Evaluation run {run_id} deleted"}


@router.post("/run")
async def trigger_eval(
    req: EvalRunRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    触发新的评估运行接口（V3）

    Args:
        req: 包含评估策略的请求体
             - strategy: 评估策略（all/vector/hybrid/hybrid_rerank）

    Returns:
        包含新创建的评估运行 ID
    """
    run_id = eval_service.run_evaluation(strategy=req.strategy, user_id=current_user.id)
    return {"run_id": run_id}
