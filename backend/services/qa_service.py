import json
from typing import List, AsyncIterator
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from backend.config import get_settings
from backend.models.schemas import Source
from backend.services.hybrid_retriever import hybrid_retriever
from backend.services.session_service import session_service

settings = get_settings()

PROMPT_TEMPLATE = """You are an enterprise knowledge base assistant. Answer questions strictly based on the provided document context and conversation history. If relevant information is not found, explicitly state "未在知识库中找到相关信息". Cite specific document sources when answering.

Previous conversation:
{chat_history}

Context:
{context}

Question: {question}

Answer:"""


class QAService:
    STRATEGIES = {
        "vector": hybrid_retriever.vector_search,
        "hybrid": hybrid_retriever.hybrid_search,
        "hybrid_rerank": hybrid_retriever.hybrid_search_with_rerank,
    }

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.qwen_model,
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            temperature=0.3,
        )
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def _build_context(self, docs: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs):
            filename = doc.metadata.get("filename", "unknown")
            parts.append(f"[Source {i+1}: {filename}]\n{doc.page_content}")
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

    def _format_history(self, messages: list) -> str:
        if not messages:
            return "(无历史对话)"
        lines = []
        for msg in messages[-10:]:
            role = "用户" if msg.type == "human" else "助手"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def search(self, query: str, strategy: str = "hybrid_rerank", top_k: int = 5) -> List[Document]:
        retriever_fn = self.STRATEGIES.get(strategy, hybrid_retriever.hybrid_search_with_rerank)
        return retriever_fn(query, top_k=top_k)

    def ask(self, question: str, strategy: str = "hybrid_rerank", top_k: int = 5):
        docs = self.search(question, strategy, top_k)
        if not docs:
            return {"answer": "未在知识库中找到相关信息。", "sources": []}
        context = self._build_context(docs)
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
        # (1) 检索
        docs = self.search(question, strategy, top_k)
        context = self._build_context(docs) if docs else ""

        # (2) 加载历史
        history = session_service.get_history(session_id)
        history_text = self._format_history(history.messages)

        # (3) 首包：sources
        sources = self._extract_sources(docs)
        yield f"data: {json.dumps({'type': 'sources', 'data': [s.model_dump() for s in sources]}, ensure_ascii=False)}\n\n"

        if not docs:
            full_answer = "未在知识库中找到相关信息。"
            yield f"data: {json.dumps({'type': 'token', 'data': full_answer}, ensure_ascii=False)}\n\n"
        else:
            messages = self.prompt.format_messages(
                chat_history=history_text, context=context, question=question
            )
            full_answer = ""
            async for chunk in self.llm.astream(messages):
                token = chunk.content
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)}\n\n"

        # (4) 持久化
        session_service.add_message(session_id, "user", question)
        session_service.add_message(session_id, "assistant", full_answer,
                                    [s.model_dump() for s in sources])

        # (5) 自动标题
        session = session_service.get_session(session_id)
        if session and session.get("title") == "新对话":
            title = question[:30] + ("..." if len(question) > 30 else "")
            session_service.update_title(session_id, title)

        # (6) 结束
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"


qa_service = QAService()
