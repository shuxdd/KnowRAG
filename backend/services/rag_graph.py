"""
Agentic RAG 流水线

用 LangGraph StateGraph 实现 Agent + Pipeline 混合架构：
- Agent 做决策（策略选择、查询改写、质量评估）
- Pipeline 做执行（检索、生成、持久化）

图拓扑:
  START → route → [chat/rag]
                    ├─ chat → chat_generate ─────────────┐
                    └─ rag  → agent_plan → retrieve → generate → agent_evaluate → [ok/retry]
                                ↑                                            │
                                └──── retry（LLM 自主决定新策略+查询）────────┘
                                                                          ↓
                                                                    stream_generate → persist → END
"""

import json
import asyncio
import logging
from typing import Any, List, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer

from backend.models.schemas import Source
from backend.services.hybrid_retriever import (
    hybrid_retriever, retrieval_cache, rrf_fusion,
    get_retrieval_progress,
)
from backend.services.query_router import query_router
from backend.services.session_service import session_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    "fast": "fast",
    "vector": "fast",
    "precise": "precise",
    "hybrid": "precise",
    "deep": "deep",
    "hybrid_rerank": "deep",
}

ROUTE_DESC = {
    "deep": "深度检索（向量 + BM25 + Rerank）",
    "precise": "精确检索（向量 + BM25 混合）",
    "fast": "快速检索（仅向量检索）",
    "chat": "闲聊模式",
}

INTENT_DESC = {
    "compare": "对比分析",
    "define": "概念定义",
    "list": "列举归纳",
    "how_to": "步骤指导",
    "factoid": "事实查询",
}

AGENT_PLAN_PROMPT = """你是企业知识库的检索规划器。根据用户问题和对话历史，决定最佳检索方案。

用户问题: {question}
对话历史: {history}
当前重试次数: {retry_count}/{max_retries}

可用检索策略:
- fast: 纯向量检索，速度快，适合简单事实问题（是什么/在哪/是谁）
- precise: 向量+BM25混合，适合一般问题（列举/步骤/方法）
- deep: 向量+BM25+Rerank，适合复杂/对比/推理/多跳问题

请返回 JSON（不要包含其他文字）:
{{
  "strategy": "fast/precise/deep",
  "queries": ["查询1"],
  "intent": "compare/define/list/how_to/factoid",
  "reasoning": "选择理由"
}}

规则:
- queries: 改写后的查询，1-3个。指代消解（把"它"还原为具体实体）、纠错、扩展
- 如果问题清晰且无需改写，queries 只包含原问题
- 对比/拆分类问题可拆成多个子查询
- intent 用于选择回答模板"""

AGENT_EVALUATE_PROMPT = """你是企业知识库的质量评估器。评估答案是否充分回答了用户问题。

问题: {question}
检索到的上下文（前500字）:
{context_snippet}

生成的答案:
{answer}

当前重试次数: {retry_count}/{max_retries}

评估标准:
1. 答案是否直接回答了问题
2. 答案是否有上下文支撑（非幻觉）
3. 答案是否足够详细完整

规则:
- 如果答案说"未在知识库中找到相关信息"且上下文非空，verdict="retry"
- 如果答案明显不完整、偏离主题或存在幻觉，verdict="retry"
- 如果上下文为空（检索无结果），verdict="ok"（重试也没用）
- 已达重试上限时，verdict="ok"
- 怀疑不足时倾向于 retry

请返回 JSON（不要包含其他文字）:
{{
  "verdict": "ok/retry",
  "reason": "评估理由",
  "new_strategy": "precise/deep",
  "new_queries": ["改进后的查询"]
}}

new_strategy 和 new_queries 仅在 verdict="retry" 时需要。
new_strategy 应与当前策略不同，尝试用更好的方式检索。"""


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    # 输入（不可变）
    question: str
    session_id: str
    user_id: int | None
    top_k: int
    strategy: str              # 用户指定的策略 ("auto" / "fast" / ...)
    history_text: str

    # route 节点产出
    actual_strategy: str       # fast/precise/deep/chat

    # agent_plan 节点产出
    plan: dict                 # LLM 返回的完整规划 JSON
    queries: List[str]         # 改写后的查询列表
    intent: str                # 意图分类
    plan_reasoning: str        # 规划理由

    # retrieve 节点产出
    docs: List[Document]
    sources: List[Source]
    context: str

    # generate 节点产出
    answer: str
    gen_messages: list

    # agent_evaluate 节点产出
    evaluation: str            # "ok" / "retry"
    eval_reason: str
    retry_count: int
    max_retries: int
    needs_retry: bool

    # 兼容字段（供前端和 qa_service 使用）
    rewrite_result: dict
    reflection: str
    reflection_reason: str


# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------

_qa_service = None

def _get_qa_service():
    global _qa_service
    if _qa_service is None:
        from backend.services.qa_service import qa_service
        _qa_service = qa_service
    return _qa_service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(writer, event_type: str, data: Any):
    writer({"type": event_type, "data": data})

def _sse_thinking(writer, step: str, text: str):
    _sse(writer, "thinking", {"step": step, "text": text})

def _retrieve(query: str, strategy: str, top_k: int, user_id: int | None) -> list[Document]:
    fn_map = {
        "fast": hybrid_retriever._fast_retrieve,
        "precise": hybrid_retriever._precise_retrieve,
        "deep": hybrid_retriever._deep_retrieve,
    }
    normalized = STRATEGY_MAP.get(strategy, "deep")
    fn = fn_map[normalized]
    return fn(query, top_k, user_id=user_id)

def _parse_json_response(raw: str) -> dict:
    """从 LLM 响应中提取 JSON。"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

async def route_node(state: RAGState) -> dict:
    """策略路由：正则规则零成本判断 chat/rag。"""
    writer = get_stream_writer()
    query = state["question"]
    user_strategy = state["strategy"]

    if user_strategy != "auto":
        actual = STRATEGY_MAP.get(user_strategy, user_strategy)
    else:
        actual = query_router.route(query, llm_hint=None)

    route_text = ROUTE_DESC.get(actual, actual)
    _sse_thinking(writer, "route", f"策略路由: {route_text}")

    return {"actual_strategy": actual}


async def agent_plan_node(state: RAGState) -> dict:
    """Agent 规划：一次 LLM 调用完成策略选择 + 查询改写 + 意图分类。"""
    writer = get_stream_writer()
    qs = _get_qa_service()
    question = state["question"]
    history = state["history_text"]
    user_strategy = state["strategy"]
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    # 如果用户显式指定了策略且不是重试，只做查询改写和意图分类
    if user_strategy != "auto" and retry_count == 0:
        prompt = AGENT_PLAN_PROMPT.format(
            question=question,
            history=history or "(无历史对话)",
            retry_count=retry_count,
            max_retries=max_retries,
        )
        try:
            response = await qs.llm.ainvoke(prompt)
            plan = _parse_json_response(response.content)
        except Exception:
            logger.warning("Agent plan parse failed, using defaults", exc_info=True)
            plan = {"strategy": user_strategy, "queries": [question], "intent": "factoid", "reasoning": "解析失败"}

        # 强制使用用户指定的策略
        plan["strategy"] = user_strategy
    else:
        prompt = AGENT_PLAN_PROMPT.format(
            question=question,
            history=history or "(无历史对话)",
            retry_count=retry_count,
            max_retries=max_retries,
        )
        try:
            response = await qs.llm.ainvoke(prompt)
            plan = _parse_json_response(response.content)
        except Exception:
            logger.warning("Agent plan parse failed, using defaults", exc_info=True)
            plan = {"strategy": "deep", "queries": [question], "intent": "factoid", "reasoning": "解析失败，默认深度检索"}

    # 提取字段
    actual_strategy = STRATEGY_MAP.get(plan.get("strategy", "deep"), "deep")
    queries = plan.get("queries", [question])
    if not queries:
        queries = [question]
    intent = plan.get("intent", "factoid")
    reasoning = plan.get("reasoning", "")

    # 构建兼容的 rewrite_result
    rewrite_result = {
        "original": question,
        "rewritten": queries[0] if queries else question,
        "sub_queries": queries[1:] if len(queries) > 1 else [],
        "changes": [],
        "route": actual_strategy,
    }

    plan_text = f"策略: {actual_strategy}, 意图: {INTENT_DESC.get(intent, intent)}, 查询: {len(queries)}个"
    if reasoning:
        plan_text += f" ({reasoning[:60]})"
    _sse_thinking(writer, "plan", f"Agent 规划: {plan_text}")

    if len(queries) > 1:
        sub_text = " → ".join(queries)
        _sse_thinking(writer, "sub_queries", f"查询拆分: {sub_text}")

    return {
        "plan": plan,
        "actual_strategy": actual_strategy,
        "queries": queries,
        "intent": intent,
        "plan_reasoning": reasoning,
        "rewrite_result": rewrite_result,
    }


async def retrieve_node(state: RAGState) -> dict:
    """文档检索。"""
    writer = get_stream_writer()
    query = state["question"]
    strategy = state["actual_strategy"]
    queries = state.get("queries", [query])
    top_k = state["top_k"]
    user_id = state["user_id"]
    history = state["history_text"]

    # 检索缓存
    cache_key = retrieval_cache.build_cache_key(
        namespace="qa_search",
        query=query,
        strategy=strategy,
        top_k=top_k,
        chat_history=history,
        extra={"queries": queries, "user_id": user_id, "retry": state.get("retry_count", 0)},
    )
    cached = retrieval_cache.get(cache_key, label=query)
    if cached:
        docs = cached
    else:
        if len(queries) == 1:
            docs = _retrieve(queries[0], strategy, top_k, user_id)
        else:
            all_docs = [_retrieve(q, strategy, top_k, user_id) for q in queries]
            docs = rrf_fusion(all_docs, top_n=top_k)
        retrieval_cache.set(cache_key, docs, label=query)

    # 发送检索进度
    progress = get_retrieval_progress()
    if progress:
        for entry in progress:
            _sse_thinking(writer, entry["stage"], entry["text"])

    qs = _get_qa_service()
    sources = qs._extract_sources(docs)
    context = qs._build_context(docs) if docs else ""
    rewrite_result = state.get("rewrite_result", {})

    writer({
        "type": "sources",
        "data": [s.model_dump() for s in sources],
        "rewrite": rewrite_result,
        "route": strategy,
    })

    return {"docs": docs, "sources": sources, "context": context}


async def generate_node(state: RAGState) -> dict:
    """非流式生成答案（缓冲，供 evaluate 评估）。"""
    writer = get_stream_writer()
    docs = state["docs"]
    intent = state.get("intent", "factoid")
    qs = _get_qa_service()

    if not docs:
        return {"answer": "未在知识库中找到相关信息。", "gen_messages": []}

    _sse_thinking(writer, "synthesize", f"LLM 生成答案中（意图: {intent}, 参考 {len(docs)} 篇文档）...")

    prompt = qs._get_prompt(intent)
    messages = prompt.format_messages(
        chat_history=state["history_text"] or "(无历史对话)",
        context=state["context"],
        question=state["question"],
    )

    response = await qs.llm.ainvoke(messages)
    return {"answer": response.content, "gen_messages": messages}


async def chat_generate_node(state: RAGState) -> dict:
    """闲聊路径：直接流式生成，跳过检索和评估。"""
    writer = get_stream_writer()
    qs = _get_qa_service()

    chat_prompt = ChatPromptTemplate.from_template(
        "You are a helpful enterprise knowledge base assistant. "
        "Answer greetings or casual conversation naturally and briefly.\n\n"
        "User: {question}\nAssistant:"
    )
    messages = chat_prompt.format_messages(question=state["question"])

    writer({"type": "sources", "data": [], "route": "chat", "rewrite": {}})

    full_answer = ""
    async for chunk in qs.llm.astream(messages):
        token = chunk.content
        if token:
            full_answer += token
            _sse(writer, "token", token)

    return {"answer": full_answer, "sources": [], "gen_messages": []}


async def agent_evaluate_node(state: RAGState) -> dict:
    """Agent 评估：LLM 自主判断答案质量，决定是否重试。"""
    writer = get_stream_writer()
    qs = _get_qa_service()
    answer = state["answer"]
    context = state.get("context", "")
    question = state["question"]
    strategy = state["actual_strategy"]
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    # 空答案 + 有上下文 + 未达上限 → 直接重试
    if answer == "未在知识库中找到相关信息。" and context and retry_count < max_retries:
        _sse_thinking(writer, "evaluate", "答案质量评估: 未找到相关信息但存在相关文档，自动重试")
        # 用默认升级策略
        escalation = {"fast": "precise", "vector": "precise", "precise": "deep", "hybrid": "deep"}
        new_strategy = escalation.get(strategy, "deep")
        _sse_thinking(writer, "retry", f"升级检索策略: {strategy} → {new_strategy}")
        return {
            "evaluation": "retry",
            "eval_reason": "未找到相关信息但存在相关文档",
            "needs_retry": True,
            "actual_strategy": new_strategy,
            "retry_count": retry_count + 1,
            "reflection": "insufficient",
            "reflection_reason": "未找到相关信息但存在相关文档",
        }

    # 空答案 + 无上下文 → 无法改善，直接返回
    if answer == "未在知识库中找到相关信息。" and not context:
        _sse_thinking(writer, "evaluate", "答案质量评估: 检索无结果，无法改善")
        return {
            "evaluation": "ok",
            "eval_reason": "检索无结果",
            "needs_retry": False,
            "reflection": "ok",
            "reflection_reason": "检索无结果",
        }

    # 已达重试上限
    if retry_count >= max_retries:
        _sse_thinking(writer, "evaluate", f"答案质量评估: 跳过（已达重试上限 {max_retries}）")
        return {
            "evaluation": "ok",
            "eval_reason": "已达重试上限",
            "needs_retry": False,
            "reflection": "ok",
            "reflection_reason": "已达重试上限",
        }

    # LLM 评估
    context_snippet = context[:500] if context else "(无上下文)"
    prompt = AGENT_EVALUATE_PROMPT.format(
        question=question,
        context_snippet=context_snippet,
        answer=answer[:1000],
        retry_count=retry_count,
        max_retries=max_retries,
    )

    try:
        response = await qs.llm.ainvoke(prompt)
        result = _parse_json_response(response.content)
        verdict = result.get("verdict", "ok")
        reason = result.get("reason", "")
        new_strategy = result.get("new_strategy", "")
        new_queries = result.get("new_queries", [])
    except Exception:
        logger.warning("Agent evaluate parse failed, defaulting to ok", exc_info=True)
        verdict, reason, new_strategy, new_queries = "ok", "评估解析失败", "", []

    if verdict == "retry":
        # 使用 LLM 建议的新策略，或默认升级
        escalation = {"fast": "precise", "vector": "precise", "precise": "deep", "hybrid": "deep"}
        if new_strategy and new_strategy in ("fast", "precise", "deep"):
            actual_new = new_strategy
        else:
            actual_new = escalation.get(strategy, "deep")

        _sse_thinking(writer, "evaluate", f"答案质量评估: 不足 - {reason}")
        _sse_thinking(writer, "retry", f"重试: 策略 {strategy} → {actual_new}" + (f", 查询: {new_queries}" if new_queries else ""))

        update = {
            "evaluation": "retry",
            "eval_reason": reason,
            "needs_retry": True,
            "actual_strategy": actual_new,
            "retry_count": retry_count + 1,
            "reflection": "insufficient",
            "reflection_reason": reason,
        }
        if new_queries:
            update["queries"] = new_queries
            update["rewrite_result"] = {
                "original": question,
                "rewritten": new_queries[0],
                "sub_queries": new_queries[1:] if len(new_queries) > 1 else [],
                "changes": ["evaluate_retry"],
                "route": actual_new,
            }
        return update

    _sse_thinking(writer, "evaluate", f"答案质量评估: 合格 - {reason}")
    return {
        "evaluation": "ok",
        "eval_reason": reason,
        "needs_retry": False,
        "reflection": "ok",
        "reflection_reason": reason,
    }


async def stream_generate_node(state: RAGState) -> dict:
    """用 llm.astream() 重新流式生成答案（prompt cache 命中）。"""
    writer = get_stream_writer()
    gen_messages = state.get("gen_messages", [])
    qs = _get_qa_service()

    if not gen_messages:
        _sse(writer, "token", state["answer"])
        return {}

    async for chunk in qs.llm.astream(gen_messages):
        token = chunk.content
        if token:
            _sse(writer, "token", token)

    return {}


async def persist_node(state: RAGState) -> dict:
    """持久化会话消息、自动标题、压缩。"""
    writer = get_stream_writer()
    session_id = state["session_id"]
    question = state["question"]
    answer = state["answer"]
    sources = state.get("sources", [])

    session_service.add_message(session_id, "user", question)
    session_service.add_message(
        session_id, "assistant", answer,
        [s.model_dump() for s in sources],
    )

    session = session_service.get_session(session_id)
    if session and session.get("title") == "新对话":
        title = question[:30] + ("..." if len(question) > 30 else "")
        session_service.update_title(session_id, title)

    qs = _get_qa_service()
    qs._maybe_compress(session_id)

    _sse(writer, "done", None)
    return {}


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def route_after_route(state: RAGState) -> str:
    return "chat" if state["actual_strategy"] == "chat" else "rag"

def route_after_evaluate(state: RAGState) -> str:
    return "retry" if state["needs_retry"] else "continue"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    workflow = StateGraph(RAGState)

    workflow.add_node("route", route_node)
    workflow.add_node("agent_plan", agent_plan_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("chat_generate", chat_generate_node)
    workflow.add_node("agent_evaluate", agent_evaluate_node)
    workflow.add_node("stream_generate", stream_generate_node)
    workflow.add_node("persist", persist_node)

    workflow.add_edge(START, "route")

    workflow.add_conditional_edges("route", route_after_route, {
        "chat": "chat_generate",
        "rag": "agent_plan",
    })

    workflow.add_edge("agent_plan", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "agent_evaluate")

    workflow.add_conditional_edges("agent_evaluate", route_after_evaluate, {
        "retry": "retrieve",
        "continue": "stream_generate",
    })

    workflow.add_edge("chat_generate", "persist")
    workflow.add_edge("stream_generate", "persist")
    workflow.add_edge("persist", END)

    return workflow.compile()


rag_graph = build_graph()
