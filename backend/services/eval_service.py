"""
评估服务模块

使用 RAGAS 框架评估 RAG 系统的性能。

评估指标：
- faithfulness: 答案忠诚度（答案内容是否忠于检索到的上下文）
- context_recall: 上下文召回率（检索上下文覆盖真实答案的程度）
- context_precision: 上下文精确度（检索上下文中相关内容的比例）
- answer_relevancy: 答案相关性（答案与问题的相关程度）

支持的检索策略评估：
- fast: 向量检索
- precise: 混合检索（向量 + BM25）
- deep: 深度检索（向量 + BM25 + HyDE + Rerank）

评估流程：
1. 加载测试数据集（JSON 格式的 QA 对）
2. 对每个问题执行选定策略的检索和问答
3. 使用 RAGAS 计算各项指标
4. 存储评估结果到 PostgreSQL 数据库
"""

import json
import uuid
import logging
from datetime import datetime, timezone

from sqlalchemy import func
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness
from langchain_core.embeddings import Embeddings

from backend.config import get_settings
from backend.db import SessionFactory
from backend.models.db_models import EvalRunORM, EvalResultORM
from backend.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)
settings = get_settings()


class _EvalEmbeddings(Embeddings):
    """将应用现有的 embedding_service 包装为 LangChain Embeddings 接口，供 RAGAS AnswerRelevancy 使用。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embedding_service.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return embedding_service.embed_query(text)


eval_embeddings = _EvalEmbeddings()


class EvalService:
    """评估服务，使用 RAGAS 框架评估 RAG 系统性能，结果存储到 PostgreSQL。"""

    SUPPORTED_STRATEGIES: tuple[str, ...] = ("fast", "precise", "deep")
    STRATEGY_ALIASES: dict[str, str] = {
        "vector": "fast",
        "hybrid": "precise",
        "hybrid_rerank": "deep",
    }
    DEFAULT_STRATEGIES: list[str] = ["fast", "precise", "deep"]

    def _normalize_strategy(self, strategy: str) -> str:
        normalized = self.STRATEGY_ALIASES.get(strategy, strategy)
        if normalized == "all":
            return normalized
        if normalized not in self.SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported evaluation strategy: {strategy}")
        return normalized

    def _expand_strategies(self, strategy: str) -> list[str]:
        normalized = self._normalize_strategy(strategy)
        return self.DEFAULT_STRATEGIES if normalized == "all" else [normalized]

    def _mark_run_failed(self, run_id: str, exc: Exception):
        with SessionFactory() as db:
            db.query(EvalRunORM).filter(EvalRunORM.id == run_id).update({
                "status": "failed",
                "completed_at": datetime.now(timezone.utc),
                "error_message": str(exc),
            })
            db.commit()

    def _mark_run_completed(self, run_id: str):
        with SessionFactory() as db:
            agg = db.query(
                func.avg(EvalResultORM.faithfulness).label("avg_f"),
                func.avg(EvalResultORM.context_recall).label("avg_cr"),
                func.avg(EvalResultORM.context_precision).label("avg_cp"),
                func.avg(EvalResultORM.answer_relevancy).label("avg_ar"),
            ).filter(EvalResultORM.run_id == run_id).first()

            db.query(EvalRunORM).filter(EvalRunORM.id == run_id).update({
                "status": "completed",
                "completed_at": datetime.now(timezone.utc),
                "error_message": None,
                "avg_faithfulness": agg.avg_f,
                "avg_context_recall": agg.avg_cr,
                "avg_context_precision": agg.avg_cp,
                "avg_answer_relevancy": agg.avg_ar,
            })
            db.commit()

    def load_dataset(self, path: str = "data/test_qa_pairs.json") -> list[dict]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            raise FileNotFoundError(f"QA dataset not found at {path}. Create the file or specify a different path.")
        except json.JSONDecodeError as exc:
            raise json.JSONDecodeError(f"Invalid JSON in {path}: {exc.msg}", exc.doc, exc.pos)

    def run_evaluation(self, strategy: str = "all", limit: int = 0, user_id: int = 0) -> str:
        from backend.services.qa_service import qa_service

        all_pairs = self.load_dataset()
        if limit > 0:
            all_pairs = all_pairs[:limit]

        strategies = self._expand_strategies(strategy)

        from openai import OpenAI

        openai_client = OpenAI(
            api_key=settings.mimo_api_key,
            base_url=settings.mimo_base_url,
        )
        ragas_llm = llm_factory(
            settings.mimo_model,
            client=openai_client,
            max_tokens=settings.mimo_max_tokens,
            temperature=0.0,
        )

        last_run_id = None
        for strat in strategies:
            run_id = str(uuid.uuid4())
            last_run_id = run_id
            now = datetime.now(timezone.utc)

            with SessionFactory() as db:
                db.add(EvalRunORM(
                    id=run_id,
                    user_id=user_id,
                    strategy=strat,
                    dataset_name="test_qa_pairs.json",
                    question_count=len(all_pairs),
                    status="running",
                    started_at=now,
                ))
                db.commit()

            try:
                samples: list[SingleTurnSample] = []
                for pair in all_pairs:
                    question = pair["question"]
                    ground_truth = pair["answer"]
                    actual_strategy = qa_service.resolve_strategy(question, strat)

                    docs = qa_service.search(
                        question,
                        strategy=actual_strategy,
                        top_k=5,
                        use_cache=False,
                        user_id=user_id,
                    )
                    contexts = [doc.page_content for doc in docs] if docs else []
                    answer_result = qa_service.answer_from_docs(question, docs)
                    answer = answer_result["answer"]

                    samples.append(
                        SingleTurnSample(
                            user_input=question,
                            response=answer,
                            retrieved_contexts=contexts if contexts else ["(no context)"],
                            reference=ground_truth,
                        )
                    )

                    with SessionFactory() as db:
                        db.add(EvalResultORM(
                            run_id=run_id,
                            question=question,
                            ground_truth=ground_truth,
                            answer=answer,
                            strategy=actual_strategy,
                            contexts=contexts,
                        ))
                        db.commit()

                metrics_df = None
                if samples:
                    dataset = EvaluationDataset(samples=samples)
                    ragas_result = evaluate(
                        dataset,
                        metrics=[
                            Faithfulness(llm=ragas_llm),
                            ContextRecall(llm=ragas_llm),
                            ContextPrecision(llm=ragas_llm),
                            AnswerRelevancy(llm=ragas_llm, embeddings=eval_embeddings),
                        ],
                        llm=ragas_llm,
                        show_progress=True,
                    )
                    metrics_df = ragas_result.to_pandas()

                if metrics_df is not None:
                    with SessionFactory() as db:
                        rows = (
                            db.query(EvalResultORM.id)
                            .filter(EvalResultORM.run_id == run_id)
                            .order_by(EvalResultORM.id)
                            .all()
                        )
                        for index, (result_id,) in enumerate(rows):
                            if index >= len(metrics_df):
                                break
                            metric_row = metrics_df.iloc[index]
                            update_vals: dict[str, float | None] = {}
                            if "faithfulness" in metric_row:
                                update_vals["faithfulness"] = float(metric_row["faithfulness"])
                            if "context_recall" in metric_row:
                                update_vals["context_recall"] = float(metric_row["context_recall"])
                            if "context_precision" in metric_row:
                                update_vals["context_precision"] = float(metric_row["context_precision"])
                            if "answer_relevancy" in metric_row:
                                update_vals["answer_relevancy"] = float(metric_row["answer_relevancy"])
                            if update_vals:
                                db.query(EvalResultORM).filter(EvalResultORM.id == result_id).update(update_vals)
                        db.commit()

                self._mark_run_completed(run_id)
            except Exception as exc:
                logger.error(f"Evaluation run {run_id} failed: {exc}", exc_info=True)
                self._mark_run_failed(run_id, exc)
                raise

        return last_run_id

    def delete_run(self, run_id: str, user_id: int | None = None) -> bool:
        with SessionFactory() as db:
            query = db.query(EvalRunORM).filter(EvalRunORM.id == run_id)
            if user_id is not None:
                query = query.filter(EvalRunORM.user_id == user_id)
            run = query.first()
            if not run:
                return False
            db.query(EvalResultORM).filter(EvalResultORM.run_id == run_id).delete()
            db.delete(run)
            db.commit()
            return True

    def get_runs(self, user_id: int | None = None) -> list[dict]:
        with SessionFactory() as db:
            query = db.query(EvalRunORM)
            if user_id is not None:
                query = query.filter(EvalRunORM.user_id == user_id)
            rows = query.order_by(EvalRunORM.started_at.desc()).all()
            return [
                {
                    "id": r.id,
                    "strategy": r.strategy,
                    "dataset_name": r.dataset_name,
                    "question_count": r.question_count,
                    "status": r.status,
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                    "error_message": r.error_message,
                    "avg_faithfulness": r.avg_faithfulness,
                    "avg_context_recall": r.avg_context_recall,
                    "avg_context_precision": r.avg_context_precision,
                    "avg_answer_relevancy": r.avg_answer_relevancy,
                }
                for r in rows
            ]

    def get_run_detail(self, run_id: str, user_id: int | None = None) -> dict | None:
        with SessionFactory() as db:
            query = db.query(EvalRunORM).filter(EvalRunORM.id == run_id)
            if user_id is not None:
                query = query.filter(EvalRunORM.user_id == user_id)
            run = query.first()
            if not run:
                return None

            results = (
                db.query(EvalResultORM)
                .filter(EvalResultORM.run_id == run_id)
                .order_by(EvalResultORM.id)
                .all()
            )
            return {
                "id": run.id,
                "strategy": run.strategy,
                "dataset_name": run.dataset_name,
                "question_count": run.question_count,
                "status": run.status,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "error_message": run.error_message,
                "avg_faithfulness": run.avg_faithfulness,
                "avg_context_recall": run.avg_context_recall,
                "avg_context_precision": run.avg_context_precision,
                "avg_answer_relevancy": run.avg_answer_relevancy,
                "results": [
                    {
                        "question": r.question,
                        "ground_truth": r.ground_truth,
                        "answer": r.answer,
                        "strategy": r.strategy,
                        "contexts": r.contexts,
                        "faithfulness": r.faithfulness,
                        "context_recall": r.context_recall,
                        "context_precision": r.context_precision,
                        "answer_relevancy": r.answer_relevancy,
                    }
                    for r in results
                ],
            }


eval_service = EvalService()
