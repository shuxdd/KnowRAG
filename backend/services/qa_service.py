"""
问答服务模块

提供 RAG 问答的核心功能：
1. 检索策略选择：auto 模式下自动选择最佳检索策略
2. 查询改写：利用对话历史改写查询（指代消解、扩展、分解）
3. 文档检索：调用混合检索器获取相关文档
4. LLM 问答：基于检索结果生成答案
5. 流式输出：支持 SSE 流式返回答案

检索策略映射：
- fast/vector: _fast_retrieve（仅向量检索）
- precise/hybrid: _precise_retrieve（向量 + BM25 混合）
- hybrid_rerank/deep: _deep_retrieve（向量 + BM25 + HyDE + Rerank）
"""

import json
import logging
from typing import AsyncIterator, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from backend.config import get_settings

logger = logging.getLogger(__name__)
from backend.models.schemas import Source
from backend.services.hybrid_retriever import (
    hybrid_retriever, retrieval_cache, rrf_fusion,
    get_last_hyde_answer, get_retrieval_progress,
)
from backend.services.query_rewriter import QueryRewriter
from backend.services.query_router import query_router
from backend.services.session_service import session_service

settings = get_settings()

# 意图分类规则：从上到下匹配，命中即返回，未命中 → factoid
INTENT_RULES: list[tuple[str, list[str]]] = [
    ("compare", ["对比", "比较", "区别", "不同", "异同", "哪个好", "哪个更",
                  "优缺点", "优劣", "差异", "vs", " VS "]),
    ("define",  ["是什么", "什么叫", "什么是", "指的是", "定义", "含义", "概念"]),
    ("list",    ["列出", "有哪些", "几个", "哪些", "分类", "种类", "包括哪些", "一共"]),
    ("how_to",  ["怎么", "如何", "怎样", "步骤", "方法", "怎么做", "流程",
                  "配置", "部署", "安装", "搭建", "实现"]),
]

INTENT_PROMPTS: dict[str, str] = {
    "compare": """You are an enterprise knowledge base assistant. Compare and contrast based on the provided context. Use a table for side-by-side comparison when applicable. Highlight similarities, differences, and give a recommendation if asked.

Previous conversation:
{chat_history}

Context:
{context}

Question: {question}

Answer (use table for comparison, then explain):""",

    "define": """You are an enterprise knowledge base assistant. First give a concise one-sentence definition, then expand with key details and examples from the context. If relevant information is not found, explicitly state "未在知识库中找到相关信息".

Previous conversation:
{chat_history}

Context:
{context}

Question: {question}

Answer (definition first, then details):""",

    "list": """You are an enterprise knowledge base assistant. List items clearly based on the context. State the total count first, then enumerate each item with a brief description. Group related items if applicable.

Previous conversation:
{chat_history}

Context:
{context}

Question: {question}

Answer (total count + numbered list):""",

    "how_to": """You are an enterprise knowledge base assistant. Provide step-by-step instructions based on the context. List prerequisites first, then number each step. Note caveats or common pitfalls for each step.

Previous conversation:
{chat_history}

Context:
{context}

Question: {question}

Answer (prerequisites → numbered steps → caveats):""",

    "factoid": """You are an enterprise knowledge base assistant. Answer questions strictly based on the provided document context and conversation history. If relevant information is not found, explicitly state "未在知识库中找到相关信息". Cite specific document sources when answering.

Previous conversation:
{chat_history}

Context:
{context}

Question: {question}

Answer:""",
}


class QAService:
    """
    问答服务

    提供 RAG 问答的核心功能，包括检索、问答和流式输出。
    """

    STRATEGIES = {
        "fast": hybrid_retriever._fast_retrieve,
        "precise": hybrid_retriever._precise_retrieve,
        "deep": hybrid_retriever._deep_retrieve,
        "vector": hybrid_retriever._fast_retrieve,
        "hybrid": hybrid_retriever._precise_retrieve,
        "hybrid_rerank": hybrid_retriever._deep_retrieve,
    }

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.mimo_model,
            api_key=settings.mimo_api_key,
            base_url=settings.mimo_base_url,
            temperature=0.3,
        )
        self._prompts = {
            intent: ChatPromptTemplate.from_template(tpl)
            for intent, tpl in INTENT_PROMPTS.items()
        }
        self.rewriter = QueryRewriter()
        self.router = query_router

    @staticmethod
    def _classify_intent(query: str) -> str:
        for intent, keywords in INTENT_RULES:
            if any(k in query for k in keywords):
                return intent
        return "factoid"

    def _get_prompt(self, intent: str) -> ChatPromptTemplate:
        return self._prompts.get(intent, self._prompts["factoid"])

    def _build_context(self, docs: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs):
            filename = doc.metadata.get("filename", "unknown")
            parts.append(f"[Source {i + 1}: {filename}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, docs: List[Document]) -> List[Source]:
        return [
            Source(
                content=doc.page_content[:300],
                filename=doc.metadata.get("filename", "unknown"),
                score=round(doc.metadata.get("score", 0.0), 4),
            )
            for doc in docs
        ]

    @staticmethod
    def _format_messages(messages: list) -> str:
        """Format raw messages list into text."""
        if not messages:
            return ""
        lines = []
        for msg in messages[-4:]:  # last 2 turns
            role = "用户" if (isinstance(msg, dict) and msg.get("role") == "user") or (hasattr(msg, "type") and msg.type == "human") else "助手"
            content = msg.get("content") if isinstance(msg, dict) else msg.content
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _format_history(self, session_id: str, messages: list) -> str:
        summary = session_service.get_summary(session_id)
        recent = self._format_messages(messages)
        if summary:
            return f"[对话摘要]\n{summary}\n\n[最近对话]\n{recent}"
        return recent

    COMPRESS_PROMPT = (
        "Summarize the following conversation in 2-3 sentences. "
        "Keep key facts, decisions, entity names, and topics discussed. "
        "Write in the same language as the conversation.\n\n"
        "{input_text}\n\nSummary:"
    )

    def _maybe_compress(self, session_id: str):
        msgs = session_service.get_messages(session_id)
        if len(msgs) <= 10:
            return
        old_summary = session_service.get_summary(session_id)
        old_text = self._format_messages(msgs[:-4])
        if not old_text:
            return

        input_text = old_text
        if old_summary:
            input_text = f"Previous summary: {old_summary}\n\nNew messages: {old_text}"

        prompt = self.COMPRESS_PROMPT.format(input_text=input_text)
        try:
            response = self.llm.invoke(prompt)
            new_summary = response.content.strip()
            if new_summary:
                session_service.update_summary(session_id, new_summary)
        except Exception:
            logger.warning("Conversation compression failed", exc_info=True)

    def resolve_strategy(self, query: str, strategy: str = "auto", chat_history: str = "") -> str:
        if strategy != "auto":
            return strategy

        llm_hint = None
        if chat_history and not self._is_trivial(query):
            rewrite_result = self.rewriter.rewrite(query, chat_history)
            llm_hint = rewrite_result.get("route")
        return self.router.route(query, llm_hint)

    @staticmethod
    def _is_trivial(query: str) -> bool:
        """极短的语气/指令/序号追问，无需 LLM 改写。"""
        q = query.strip().rstrip("。！？，.!?；;")
        if not q or len(q) < 2:
            return True
        trivial = {
            "继续说", "继续", "接着说", "接着",
            "然后呢", "然后", "还有呢", "还有", "还有吗",
            "详细点", "具体点", "具体说说", "详细说说", "展开说说", "展开",
            "第二个", "第三个", "下一个", "第一点", "第二点", "第三点",
            "比如", "例如", "比方说",
        }
        return q in trivial

    def _prepare_search(
        self,
        query: str,
        strategy: str = "auto",
        chat_history: str = "",
    ) -> tuple[str, list[str], dict]:
        rewrite_result = {
            "original": query,
            "rewritten": query,
            "sub_queries": [],
            "changes": [],
        }
        llm_hint = None
        needs_rewrite = not self._is_trivial(query)
        if needs_rewrite:
            # 无对话历史时，仅对比/步骤类问题需要改写（分解+扩展）
            # 有历史时，所有非 trivial 问题都改写（指代消解+扩展+分解）
            if not chat_history and self._classify_intent(query) not in ("compare", "how_to"):
                needs_rewrite = False
        if needs_rewrite:
            rewrite_result = self.rewriter.rewrite(query, chat_history)
            llm_hint = rewrite_result.get("route")

        actual_strategy = self.router.route(query, llm_hint) if strategy == "auto" else strategy
        if actual_strategy == "chat":
            return actual_strategy, [], rewrite_result

        queries = self.rewriter.get_queries(rewrite_result) if needs_rewrite else [query]
        return actual_strategy, queries, rewrite_result

    def _search_with_plan(
        self,
        query: str,
        actual_strategy: str,
        queries: List[str],
        top_k: int,
        chat_history: str = "",
        use_cache: bool = True,
    ) -> List[Document]:
        return self._search_with_plan_user(
            query, actual_strategy, queries, top_k, chat_history, use_cache, user_id=None,
        )

    def _search_with_plan_user(
        self,
        query: str,
        actual_strategy: str,
        queries: List[str],
        top_k: int,
        chat_history: str = "",
        use_cache: bool = True,
        user_id: int | None = None,
    ) -> List[Document]:
        if actual_strategy == "chat":
            return []

        cache_key = retrieval_cache.build_cache_key(
            namespace="qa_search",
            query=query,
            strategy=actual_strategy,
            top_k=top_k,
            chat_history=chat_history,
            extra={"queries": queries, "user_id": user_id},
        )
        if use_cache:
            cached = retrieval_cache.get(cache_key, label=query)
            if cached:
                return cached

        if len(queries) == 1:
            docs = self._retrieve_user(queries[0], actual_strategy, top_k, user_id)
        else:
            docs = self._multi_query_retrieve_user(queries, actual_strategy, top_k, user_id)

        if use_cache:
            retrieval_cache.set(cache_key, docs, label=query)
        return docs

    def search(
        self,
        query: str,
        strategy: str = "auto",
        top_k: int = 5,
        chat_history: str = "",
        use_cache: bool = True,
        user_id: int | None = None,
    ) -> List[Document]:
        actual_strategy, queries, _rewrite_result = self._prepare_search(query, strategy, chat_history)
        return self._search_with_plan_user(
            query=query,
            actual_strategy=actual_strategy,
            queries=queries,
            top_k=top_k,
            chat_history=chat_history,
            use_cache=use_cache,
            user_id=user_id,
        )

    def _retrieve(self, query: str, strategy: str, top_k: int) -> List[Document]:
        return self._retrieve_user(query, strategy, top_k, user_id=None)

    def _retrieve_user(self, query: str, strategy: str, top_k: int, user_id: int | None = None) -> List[Document]:
        retriever_fn = self.STRATEGIES.get(strategy, hybrid_retriever._deep_retrieve)
        return retriever_fn(query, top_k, user_id=user_id)

    def _multi_query_retrieve(self, queries: List[str], strategy: str, top_k: int) -> List[Document]:
        return self._multi_query_retrieve_user(queries, strategy, top_k, user_id=None)

    def _multi_query_retrieve_user(self, queries: List[str], strategy: str, top_k: int, user_id: int | None = None) -> List[Document]:
        all_docs = []
        for query in queries:
            docs = self._retrieve_user(query, strategy, top_k, user_id)
            all_docs.append(docs)
        return rrf_fusion(all_docs, top_n=top_k)

    def answer_from_docs(self, question: str, docs: List[Document], chat_history: str = ""):
        if not docs:
            return {"answer": "未在知识库中找到相关信息。", "sources": []}

        intent = self._classify_intent(question)
        prompt = self._get_prompt(intent)
        context = self._build_context(docs)
        messages = prompt.format_messages(
            chat_history=chat_history or "(无历史对话)",
            context=context,
            question=question,
        )
        response = self.llm.invoke(messages)
        return {
            "answer": response.content,
            "sources": self._extract_sources(docs),
        }

    def ask(self, question: str, strategy: str = "auto", top_k: int = 5, user_id: int | None = None):
        actual_strategy = self.resolve_strategy(question, strategy)
        if actual_strategy == "chat":
            chat_prompt = ChatPromptTemplate.from_template(
                "User: {question}\nAssistant (friendly, brief):"
            )
            messages = chat_prompt.format_messages(question=question)
            response = self.llm.invoke(messages)
            return {"answer": response.content, "sources": []}

        docs = self.search(question, actual_strategy, top_k, user_id=user_id)
        return self.answer_from_docs(question, docs)

    async def ask_stream(
        self,
        question: str,
        session_id: str,
        strategy: str = "auto",
        top_k: int = 5,
        user_id: int | None = None,
    ) -> AsyncIterator[str]:
        history = session_service.get_history(session_id)
        history_text = self._format_history(session_id, history.messages)

        actual_strategy, queries, rewrite_result = self._prepare_search(
            question,
            strategy,
            history_text,
        )

        # 思考过程：意图分类
        if actual_strategy != "chat":
            intent = self._classify_intent(question)
            intent_desc = {
                "compare": "对比分析",
                "define": "概念定义",
                "list": "列举归纳",
                "how_to": "步骤指导",
                "factoid": "事实查询",
            }
            yield f"data: {json.dumps({'type': 'thinking', 'data': {'step': 'intent', 'text': f'意图分类: {intent_desc.get(intent, intent)}'}}, ensure_ascii=False)}\n\n"

        # 思考过程：路由决策
        route_desc = {
            "deep": "深度检索（向量 + BM25 + HyDE + Rerank，适合复杂问题）",
            "hybrid_rerank": "深度检索（向量 + BM25 + HyDE + Rerank）",
            "precise": "精确检索（向量 + BM25 混合，适合一般问题）",
            "hybrid": "精确检索（向量 + BM25 混合）",
            "fast": "快速检索（仅向量检索，适合简单问题）",
            "vector": "快速检索（仅向量检索）",
            "chat": "闲聊模式",
        }
        route_text = route_desc.get(actual_strategy, actual_strategy)
        yield f"data: {json.dumps({'type': 'thinking', 'data': {'step': 'route', 'text': f'策略路由: {route_text}'}}, ensure_ascii=False)}\n\n"

        # 思考过程：查询改写
        if actual_strategy != "chat":
            rewritten = rewrite_result.get("rewritten", question)
            if rewritten != question:
                changes = rewrite_result.get("changes", [])
                changes_text = f"（{', '.join(changes)}）" if changes else ""
                yield f"data: {json.dumps({'type': 'thinking', 'data': {'step': 'rewrite', 'text': f'查询改写: {rewritten} {changes_text}'}}, ensure_ascii=False)}\n\n"
            if rewrite_result.get("sub_queries"):
                sub_text = " → ".join(rewrite_result["sub_queries"])
                yield f"data: {json.dumps({'type': 'thinking', 'data': {'step': 'sub_queries', 'text': f'子问题拆分: {sub_text}'}}, ensure_ascii=False)}\n\n"

        if actual_strategy == "chat":
            chat_prompt = ChatPromptTemplate.from_template(
                "You are a helpful enterprise knowledge base assistant. "
                "Answer greetings or casual conversation naturally and briefly.\n\n"
                "User: {question}\nAssistant:"
            )
            messages = chat_prompt.format_messages(question=question)
            full_answer = ""
            yield f"data: {json.dumps({'type': 'sources', 'data': [], 'route': 'chat', 'rewrite': {}}, ensure_ascii=False)}\n\n"
            async for chunk in self.llm.astream(messages):
                token = chunk.content
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)}\n\n"
            session_service.add_message(session_id, "user", question)
            session_service.add_message(session_id, "assistant", full_answer, [])
            session = session_service.get_session(session_id)
            if session and session.get("title") == "新对话":
                title = question[:30] + ("..." if len(question) > 30 else "")
                session_service.update_title(session_id, title)
            self._maybe_compress(session_id)
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
            return

        docs = self._search_with_plan_user(
            query=question,
            actual_strategy=actual_strategy,
            queries=queries,
            top_k=top_k,
            chat_history=history_text,
            use_cache=True,
            user_id=user_id,
        )
        context = self._build_context(docs) if docs else ""

        # 思考过程：检索流水线各步骤
        progress = get_retrieval_progress()
        if progress:
            for entry in progress:
                yield f"data: {json.dumps({'type': 'thinking', 'data': {'step': entry['stage'], 'text': entry['text']}}, ensure_ascii=False)}\n\n"

        # 思考过程：HyDE 假设答案（如有）
        hyde_answer = get_last_hyde_answer()
        if hyde_answer:
            yield f"data: {json.dumps({'type': 'thinking', 'data': {'step': 'hyde_answer', 'text': f'HyDE 假设答案: {hyde_answer[:250]}'}}, ensure_ascii=False)}\n\n"

        # 思考过程：检索结果汇总
        sources = self._extract_sources(docs)
        yield f"data: {json.dumps({'type': 'sources', 'data': [s.model_dump() for s in sources], 'rewrite': rewrite_result, 'route': actual_strategy}, ensure_ascii=False)}\n\n"

        if not docs:
            full_answer = "未在知识库中找到相关信息。"
            yield f"data: {json.dumps({'type': 'token', 'data': full_answer}, ensure_ascii=False)}\n\n"
        else:
            intent = self._classify_intent(question)
            yield f"data: {json.dumps({'type': 'thinking', 'data': {'step': 'synthesize', 'text': f'LLM 生成答案中（意图: {intent}, 参考 {len(docs)} 篇文档）...'}}, ensure_ascii=False)}\n\n"
            prompt = self._get_prompt(intent)
            messages = prompt.format_messages(
                chat_history=history_text,
                context=context,
                question=question,
            )
            full_answer = ""
            async for chunk in self.llm.astream(messages):
                token = chunk.content
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)}\n\n"

        session_service.add_message(session_id, "user", question)
        session_service.add_message(
            session_id,
            "assistant",
            full_answer,
            [s.model_dump() for s in sources],
        )

        session = session_service.get_session(session_id)
        if session and session.get("title") == "新对话":
            title = question[:30] + ("..." if len(question) > 30 else "")
            session_service.update_title(session_id, title)

        self._maybe_compress(session_id)
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"


qa_service = QAService()
