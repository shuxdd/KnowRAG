"""
评估服务模块

使用 RAGAS 框架评估 RAG 系统的性能。

评估指标：
- faithfulness: 答案忠诚度（答案内容是否忠于检索到的上下文）
- context_recall: 上下文召回率（检索上下文覆盖真实答案的程度）
- context_precision: 上下文精确度（检索上下文中相关内容的比例）
- answer_correctness: 答案正确性（与标准答案的对比）
- answer_accuracy: 答案准确率（LLM 评判生成答案的准确性）

支持的检索策略评估：
- fast: 向量检索
- precise: 混合检索（向量 + BM25）
- deep: 深度检索（向量 + BM25 + HyDE + Rerank）

评估流程：
1. 加载测试数据集（JSON 格式的 QA 对）
2. 对每个问题执行选定策略的检索和问答
3. 使用 RAGAS 计算各项指标
4. 存储评估结果到 SQLite 数据库
"""

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness

from backend.config import get_settings

settings = get_settings()


class EvalService:
    """
    评估服务

    使用 RAGAS 框架评估 RAG 系统的性能。
    支持多种检索策略的对比评估。
    """

    SUPPORTED_STRATEGIES: tuple[str, ...] = ("fast", "precise", "deep")
    STRATEGY_ALIASES: dict[str, str] = {
        "vector": "fast",
        "hybrid": "precise",
        "hybrid_rerank": "deep",
    }
    DEFAULT_STRATEGIES: list[str] = ["fast", "precise", "deep"]

    def __init__(self, db_path: str = "data/eval_results.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id TEXT PRIMARY KEY,
                    strategy TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    question_count INTEGER,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    avg_faithfulness REAL,
                    avg_context_recall REAL,
                    avg_context_precision REAL,
                    avg_answer_correctness REAL,
                    avg_answer_accuracy REAL
                );
                CREATE TABLE IF NOT EXISTS eval_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    ground_truth TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    contexts TEXT NOT NULL,
                    faithfulness REAL,
                    context_recall REAL,
                    context_precision REAL,
                    answer_correctness REAL,
                    answer_accuracy REAL,
                    FOREIGN KEY (run_id) REFERENCES eval_runs(id)
                );
                """
            )
            conn.execute("PRAGMA journal_mode=WAL")
            self._ensure_column(conn, "eval_runs", "status", "TEXT NOT NULL DEFAULT 'running'")
            self._ensure_column(conn, "eval_runs", "error_message", "TEXT")
            self._ensure_column(conn, "eval_results", "strategy", "TEXT")
            self._ensure_column(conn, "eval_results", "answer_accuracy_raw", "TEXT")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str):
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

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
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE eval_runs SET status = ?, completed_at = ?, error_message = ? WHERE id = ?",
                ("failed", datetime.now(timezone.utc).isoformat(), str(exc), run_id),
            )

    def _mark_run_completed(self, run_id: str):
        with sqlite3.connect(self.db_path) as conn:
            agg = conn.execute(
                """
                SELECT AVG(faithfulness), AVG(context_recall), AVG(context_precision),
                       AVG(answer_correctness), AVG(answer_accuracy)
                FROM eval_results
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            conn.execute(
                """
                UPDATE eval_runs
                SET status = ?, completed_at = ?, error_message = ?, avg_faithfulness = ?,
                    avg_context_recall = ?, avg_context_precision = ?, avg_answer_correctness = ?,
                    avg_answer_accuracy = ?
                WHERE id = ?
                """,
                (
                    "completed",
                    datetime.now(timezone.utc).isoformat(),
                    None,
                    agg[0],
                    agg[1],
                    agg[2],
                    agg[3],
                    agg[4],
                    run_id,
                ),
            )

    def load_dataset(self, path: str = "data/test_qa_pairs.json") -> list[dict]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            raise FileNotFoundError(f"QA dataset not found at {path}. Create the file or specify a different path.")
        except json.JSONDecodeError as exc:
            raise json.JSONDecodeError(f"Invalid JSON in {path}: {exc.msg}", exc.doc, exc.pos)

    def run_evaluation(self, strategy: str = "all", limit: int = 0) -> str:
        from backend.services.qa_service import qa_service

        all_pairs = self.load_dataset()
        if limit > 0:
            all_pairs = all_pairs[:limit]

        strategies = self._expand_strategies(strategy)

        eval_llm = ChatOpenAI(
            model=settings.qwen_model,
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            temperature=0.0,
        )

        from openai import OpenAI

        openai_client = OpenAI(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
        )
        ragas_llm = llm_factory(
            settings.qwen_model,
            client=openai_client,
            max_tokens=settings.qwen_max_tokens,
        )

        last_run_id = None
        for strat in strategies:
            run_id = str(uuid.uuid4())
            last_run_id = run_id
            started_at = datetime.now(timezone.utc).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO eval_runs (id, strategy, dataset_name, question_count, started_at, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (run_id, strat, "test_qa_pairs.json", len(all_pairs), started_at, "running"),
                )

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
                    )
                    contexts = [doc.page_content for doc in docs] if docs else []
                    answer_result = qa_service.answer_from_docs(question, docs)
                    answer = answer_result["answer"]
                    accuracy, accuracy_raw = self._compute_answer_accuracy(answer, ground_truth, eval_llm)

                    samples.append(
                        SingleTurnSample(
                            user_input=question,
                            response=answer,
                            retrieved_contexts=contexts if contexts else ["(no context)"],
                            reference=ground_truth,
                        )
                    )

                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute(
                            """
                            INSERT INTO eval_results (
                                run_id, question, ground_truth, answer, strategy, contexts,
                                answer_accuracy, answer_accuracy_raw
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                run_id,
                                question,
                                ground_truth,
                                answer,
                                actual_strategy,
                                json.dumps(contexts, ensure_ascii=False),
                                accuracy,
                                accuracy_raw,
                            ),
                        )

                metrics_df = None
                if samples:
                    dataset = EvaluationDataset(samples=samples)
                    ragas_result = evaluate(
                        dataset,
                        metrics=[
                            Faithfulness(llm=ragas_llm),
                            ContextRecall(llm=ragas_llm),
                            ContextPrecision(llm=ragas_llm),
                            AnswerCorrectness(llm=ragas_llm, weights=[1.0, 0.0]),
                        ],
                        llm=ragas_llm,
                        show_progress=True,
                    )
                    metrics_df = ragas_result.to_pandas()

                if metrics_df is not None:
                    with sqlite3.connect(self.db_path) as conn:
                        rows = conn.execute(
                            "SELECT id FROM eval_results WHERE run_id = ? ORDER BY id",
                            (run_id,),
                        ).fetchall()
                        for index, row in enumerate(rows):
                            if index >= len(metrics_df):
                                break
                            metric_row = metrics_df.iloc[index]
                            conn.execute(
                                """
                                UPDATE eval_results
                                SET faithfulness = ?, context_recall = ?, context_precision = ?, answer_correctness = ?
                                WHERE id = ?
                                """,
                                (
                                    float(metric_row.get("faithfulness", 0)) if "faithfulness" in metric_row else None,
                                    float(metric_row.get("context_recall", 0)) if "context_recall" in metric_row else None,
                                    float(metric_row.get("context_precision", 0)) if "context_precision" in metric_row else None,
                                    float(metric_row.get("answer_correctness", 0)) if "answer_correctness" in metric_row else None,
                                    row[0],
                                ),
                            )

                self._mark_run_completed(run_id)
            except Exception as exc:
                self._mark_run_failed(run_id, exc)
                raise

        return last_run_id

    def _compute_answer_accuracy(self, answer: str, ground_truth: str, llm) -> tuple[float | None, str]:
        prompt = f"""You are evaluating answer accuracy. Given the reference answer and the generated answer, rate how many key facts from the reference are covered in the generated answer.

Reference: {ground_truth}

Generated: {answer}

Return ONLY a number between 0.0 and 1.0, where:
- 1.0 = all key facts covered
- 0.5 = about half covered
- 0.0 = none covered

Score:"""
        response = llm.invoke(prompt)
        text = response.content.strip()
        match = re.search(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", text)
        if not match:
            return None, text
        return max(0.0, min(1.0, float(match.group(1)))), text

    def delete_run(self, run_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM eval_results WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM eval_runs WHERE id = ?", (run_id,))
            return conn.total_changes > 0

    def get_runs(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM eval_runs ORDER BY started_at DESC").fetchall()
            return [dict(row) for row in rows]

    def get_run_detail(self, run_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            run = conn.execute(
                "SELECT * FROM eval_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if not run:
                return None

            results = conn.execute(
                "SELECT * FROM eval_results WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
            run_dict = dict(run)
            run_dict["results"] = [
                {
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "answer": row["answer"],
                    "strategy": row["strategy"],
                    "contexts": json.loads(row["contexts"]),
                    "faithfulness": row["faithfulness"],
                    "context_recall": row["context_recall"],
                    "context_precision": row["context_precision"],
                    "answer_correctness": row["answer_correctness"],
                    "answer_accuracy": row["answer_accuracy"],
                }
                for row in results
            ]
            return run_dict


eval_service = EvalService()
