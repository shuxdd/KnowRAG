import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    faithfulness, context_recall, context_precision, answer_correctness,
)
from langchain_openai import ChatOpenAI

from backend.config import get_settings

settings = get_settings()


class EvalService:
    def __init__(self, db_path: str = "data/eval_results.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
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
            """)
            conn.execute("PRAGMA journal_mode=WAL")

    def load_dataset(self, path: str = "data/test_qa_pairs.json") -> list[dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"QA dataset not found at {path}. Create the file or specify a different path.")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in {path}: {e.msg}", e.doc, e.pos)

    def run_evaluation(self, strategy: str = "all", limit: int = 0) -> str:
        """Run evaluation. strategy: all | vector | hybrid | hybrid_rerank. limit: 0=all."""
        from backend.services.qa_service import qa_service

        all_pairs = self.load_dataset()
        if limit > 0:
            all_pairs = all_pairs[:limit]

        strategies = (
            ["vector", "hybrid", "hybrid_rerank"]
            if strategy == "all"
            else [strategy]
        )

        # LangChain LLM for custom answer accuracy scoring
        eval_llm = ChatOpenAI(
            model=settings.qwen_model,
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            temperature=0.0,
        )

        # RAGAS-compatible Instructor LLM for metrics evaluation
        from openai import OpenAI
        _openai_client = OpenAI(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
        )
        ragas_llm = llm_factory(settings.qwen_model, client=_openai_client)

        last_run_id = None
        for strat in strategies:
            run_id = str(uuid.uuid4())
            last_run_id = run_id
            now = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO eval_runs (id, strategy, dataset_name, question_count, started_at) VALUES (?, ?, ?, ?, ?)",
                    (run_id, strat, "test_qa_pairs.json", len(all_pairs), now),
                )

            samples = []
            for pair in all_pairs:
                question = pair["question"]
                ground_truth = pair["ground_truth"]

                # Retrieve
                docs = qa_service.search(question, strategy=strat, top_k=5)
                contexts = [doc.page_content for doc in docs] if docs else []

                # Generate answer
                if docs:
                    result = qa_service.ask(question, strategy=strat, top_k=5)
                    answer = result["answer"]
                else:
                    answer = "未在知识库中找到相关信息。"

                # Compute accuracy (custom metric)
                accuracy = self._compute_answer_accuracy(answer, ground_truth, eval_llm)

                # Build RAGAS sample — use correct field names for ragas 0.4.3
                samples.append(
                    SingleTurnSample(
                        user_input=question,
                        response=answer,
                        retrieved_contexts=contexts if contexts else ["(no context)"],
                        reference=ground_truth,
                    )
                )

                # Persist individual result
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "INSERT INTO eval_results (run_id, question, ground_truth, answer, contexts, answer_accuracy) VALUES (?, ?, ?, ?, ?, ?)",
                        (run_id, question, ground_truth, answer, json.dumps(contexts, ensure_ascii=False), accuracy),
                    )

            # RAGAS batch evaluation
            if samples:
                ds = EvaluationDataset(samples=samples)
                ragas_result = evaluate(
                    ds,
                    metrics=[
                        faithfulness(),
                        context_recall(),
                        context_precision(),
                        answer_correctness(),
                    ],
                    llm=ragas_llm,
                    show_progress=True,
                )
                metrics_df = ragas_result.to_pandas()
            else:
                metrics_df = None

            # Update individual results with RAGAS metrics
            if metrics_df is not None:
                with sqlite3.connect(self.db_path) as conn:
                    rows = conn.execute(
                        "SELECT id FROM eval_results WHERE run_id = ? ORDER BY id",
                        (run_id,),
                    ).fetchall()
                    for i, row in enumerate(rows):
                        if i < len(metrics_df):
                            mr = metrics_df.iloc[i]
                            conn.execute(
                                "UPDATE eval_results SET faithfulness=?, context_recall=?, context_precision=?, answer_correctness=? WHERE id=?",
                                (
                                    float(mr.get("faithfulness", 0)) if "faithfulness" in mr else None,
                                    float(mr.get("context_recall", 0)) if "context_recall" in mr else None,
                                    float(mr.get("context_precision", 0)) if "context_precision" in mr else None,
                                    float(mr.get("answer_correctness", 0)) if "answer_correctness" in mr else None,
                                    row[0],
                                ),
                            )

            # Aggregate and update run summary
            with sqlite3.connect(self.db_path) as conn:
                agg = conn.execute(
                    "SELECT AVG(faithfulness), AVG(context_recall), AVG(context_precision), AVG(answer_correctness), AVG(answer_accuracy) FROM eval_results WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                conn.execute(
                    "UPDATE eval_runs SET completed_at=?, avg_faithfulness=?, avg_context_recall=?, avg_context_precision=?, avg_answer_correctness=?, avg_answer_accuracy=? WHERE id=?",
                    (datetime.now(timezone.utc).isoformat(), agg[0], agg[1], agg[2], agg[3], agg[4], run_id),
                )

        return last_run_id

    def _compute_answer_accuracy(self, answer: str, ground_truth: str, llm) -> float:
        prompt = f"""You are evaluating answer accuracy. Given the reference answer and the generated answer, rate how many key facts from the reference are covered in the generated answer.

Reference: {ground_truth}

Generated: {answer}

Return ONLY a number between 0.0 and 1.0, where:
- 1.0 = all key facts covered
- 0.5 = about half covered
- 0.0 = none covered

Score:"""
        response = llm.invoke(prompt)
        try:
            text = response.content.strip()
            return max(0.0, min(1.0, float(text)))
        except ValueError:
            return 0.0

    def get_runs(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM eval_runs ORDER BY started_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_run_detail(self, run_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            run = conn.execute(
                "SELECT * FROM eval_runs WHERE id = ?", (run_id,)
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
                    "question": r["question"],
                    "ground_truth": r["ground_truth"],
                    "answer": r["answer"],
                    "contexts": json.loads(r["contexts"]),
                    "faithfulness": r["faithfulness"],
                    "context_recall": r["context_recall"],
                    "context_precision": r["context_precision"],
                    "answer_correctness": r["answer_correctness"],
                    "answer_accuracy": r["answer_accuracy"],
                }
                for r in results
            ]
            return run_dict


eval_service = EvalService()
