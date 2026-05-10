from fastapi import APIRouter, HTTPException
from backend.models.schemas import (
    EvalListResponse, EvalRunDetail, EvalRunInfo, EvalRunRequest,
)
from backend.services.eval_service import eval_service

router = APIRouter(prefix="/api/eval", tags=["eval"])


@router.get("/results", response_model=EvalListResponse)
async def list_eval_runs():
    runs = eval_service.get_runs()
    return EvalListResponse(runs=[EvalRunInfo(**r) for r in runs])


@router.get("/results/{run_id}", response_model=EvalRunDetail)
async def get_eval_run(run_id: str):
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
    run_id = eval_service.run_evaluation(strategy=req.strategy)
    return {"run_id": run_id}
