import json
from typing import List, AsyncIterator
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from backend.config import get_settings
from backend.models.schemas import Source
from backend.services.hybrid_retriever import hybrid_retriever
from backend.services.query_rewriter import QueryRewriter
from backend.services.session_service import session_service

settings = get_settings()

# RAG 系统的提示词模板
# 要求模型基于提供的上下文回答，并注明来源
PROMPT_TEMPLATE = """You are an enterprise knowledge base assistant. Answer questions strictly based on the provided document context and conversation history. If relevant information is not found, explicitly state "未在知识库中找到相关信息". Cite specific document sources when answering.

Previous conversation:
{chat_history}

Context:
{context}

Question: {question}

Answer:"""


class QAService:
    """
    问答服务
    核心服务，负责：
    - 文档检索（支持多种策略）
    - 基于检索结果的问答生成
    - 流式问答响应（SSE）
    - 多轮对话支持
    """

    # 检索策略映射表
    STRATEGIES = {
        "vector": hybrid_retriever.vector_search,           # 纯向量检索
        "hybrid": hybrid_retriever.hybrid_search,           # 混合检索（向量 + BM25）
        "hybrid_rerank": hybrid_retriever.hybrid_search_with_rerank,  # 混合检索 + 重排序
    }

    def __init__(self):
        """
        初始化问答服务
        - 配置 LLM（通义千问）
        - 初始化提示词模板
        """
        self.llm = ChatOpenAI(
            model=settings.qwen_model,
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            temperature=0.3,  # 中等随机性，平衡创造性和准确性
        )
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.rewriter = QueryRewriter()

    def _build_context(self, docs: List[Document]) -> str:
        """
        将检索到的文档列表构建为上下文字符串

        Args:
            docs: Document 对象列表

        Returns:
            格式化的上下文字符串，每个文档前标注来源
        """
        parts = []
        for i, doc in enumerate(docs):
            filename = doc.metadata.get("filename", "unknown")
            parts.append(f"[Source {i+1}: {filename}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, docs: List[Document]) -> List[Source]:
        """
        从 Document 对象中提取来源信息

        Args:
            docs: Document 对象列表

        Returns:
            Source 对象列表，包含文件名、内容摘要和相关性分数
        """
        return [
            Source(
                content=doc.page_content[:300],  # 截取前300字符作为摘要
                filename=doc.metadata.get("filename", "unknown"),
                score=round(doc.metadata.get("score", 0.0), 4),
            )
            for doc in docs
        ]

    def _format_history(self, messages: list) -> str:
        """
        将对话历史格式化为字符串

        Args:
            messages: LangChain 消息对象列表

        Returns:
            格式化的对话历史字符串，最多包含最近10条消息
        """
        if not messages:
            return "(无历史对话)"
        lines = []
        for msg in messages[-10:]:  # 只取最近10条消息
            role = "用户" if msg.type == "human" else "助手"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def search(self, query: str, strategy: str = "hybrid_rerank", top_k: int = 5, chat_history: str = "") -> List[Document]:
        """
        根据策略执行文档检索，支持查询改写

        Args:
            query: 查询文本
            strategy: 检索策略（vector/hybrid/hybrid_rerank）
            top_k: 返回的文档数量
            chat_history: 格式化的对话历史，有历史时触发查询改写

        Returns:
            Document 对象列表
        """
        if chat_history:
            rewrite_result = self.rewriter.rewrite(query, chat_history)
            queries = self.rewriter.get_queries(rewrite_result)
        else:
            queries = [query]

        if len(queries) == 1:
            return self._retrieve(queries[0], strategy, top_k)
        else:
            return self._multi_query_retrieve(queries, strategy, top_k)

    def _retrieve(self, query: str, strategy: str, top_k: int) -> List[Document]:
        """Execute a single retrieval against the chosen strategy"""
        retriever_fn = self.STRATEGIES.get(strategy, hybrid_retriever.hybrid_search_with_rerank)
        return retriever_fn(query, top_k=top_k)

    def _multi_query_retrieve(self, queries: List[str], strategy: str, top_k: int) -> List[Document]:
        """多查询检索：每个子查询独立检索，RRF 融合"""
        all_docs = []
        for q in queries:
            docs = self._retrieve(q, strategy, top_k)
            all_docs.append(docs)
        return hybrid_retriever.rrf_fusion(
            all_docs[0],
            [d for docs in all_docs[1:] for d in docs],
            top_k=top_k,
        )

    def ask(self, question: str, strategy: str = "hybrid_rerank", top_k: int = 5):
        """
        非流式问答接口
        执行检索 + LLM 生成，返回完整答案

        Args:
            question: 用户问题
            strategy: 检索策略
            top_k: 检索文档数量

        Returns:
            包含 answer（答案）和 sources（来源列表）的字典
        """
        docs = self.search(question, strategy, top_k)
        if not docs:
            return {"answer": "未在知识库中找到相关信息。", "sources": []}
        # 构建上下文
        context = self._build_context(docs)
        # 调用 LLM 生成答案
        messages = self.prompt.format_messages(chat_history="(无历史对话)", context=context, question=question)
        response = self.llm.invoke(messages)
        return {
            "answer": response.content,
            "sources": self._extract_sources(docs),
        }

    async def ask_stream(
        self, question: str, session_id: str,
        strategy: str = "hybrid_rerank", top_k: int = 5,
    ) -> AsyncIterator[str]:
        """
        流式问答接口（SSE）
        支持多轮对话，返回流式响应

        Args:
            question: 用户问题
            session_id: 会话 ID，用于关联对话历史
            strategy: 检索策略
            top_k: 检索文档数量

        Yields:
            SSE 格式的数据字符串，包含：
            - sources: 检索来源信息
            - token: LLM 生成的文本片段
            - done: 结束标识
        """
        # (1) 加载对话历史
        history = session_service.get_history(session_id)
        history_text = self._format_history(history.messages)

        # (2) 查询改写 + 文档检索
        if history.messages:
            rewrite_result = self.rewriter.rewrite(question, history_text)
            queries = self.rewriter.get_queries(rewrite_result)
        else:
            queries = [question]
            rewrite_result = {"original": question, "rewritten": question, "sub_queries": [], "changes": []}
        if len(queries) == 1:
            docs = self._retrieve(queries[0], strategy, top_k)
        else:
            docs = self._multi_query_retrieve(queries, strategy, top_k)
        context = self._build_context(docs) if docs else ""

        # (3) 首包：发送检索到的来源信息 + 改写信息
        sources = self._extract_sources(docs)
        yield f"data: {json.dumps({'type': 'sources', 'data': [s.model_dump() for s in sources], 'rewrite': rewrite_result}, ensure_ascii=False)}\n\n"

        # (4) 如果没有检索到文档，直接返回
        if not docs:
            full_answer = "未在知识库中找到相关信息。"
            yield f"data: {json.dumps({'type': 'token', 'data': full_answer}, ensure_ascii=False)}\n\n"
        else:
            # (5) 流式调用 LLM 生成答案
            messages = self.prompt.format_messages(
                chat_history=history_text, context=context, question=question
            )
            full_answer = ""
            async for chunk in self.llm.astream(messages):
                token = chunk.content
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)}\n\n"

        # (6) 持久化对话消息到数据库
        session_service.add_message(session_id, "user", question)
        session_service.add_message(session_id, "assistant", full_answer,
                                    [s.model_dump() for s in sources])

        # (7) 自动生成会话标题（取问题前30个字符）
        session = session_service.get_session(session_id)
        if session and session.get("title") == "新对话":
            title = question[:30] + ("..." if len(question) > 30 else "")
            session_service.update_title(session_id, title)

        # (8) 发送结束标识
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"


# 全局单例实例
qa_service = QAService()
