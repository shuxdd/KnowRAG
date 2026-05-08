from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from backend.config import settings
from backend.services.vector_service import get_hybrid_retriever
from backend.services.reranker import RerankedRetriever
from backend.models.schemas import AnswerResponse, SourceCitation, SessionHistory, HistoryMessage

RAG_SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请根据提供的文档片段和对话历史回答问题。

规则：
1. 仅根据提供的文档片段回答，不要使用外部知识
2. 如果文档片段不足以回答问题，请明确说明"根据提供的文档无法回答此问题"
3. 回答要简洁、准确，使用中文
4. 如果回答引用了文档内容，请注明来源
5. 如果用户的问题是对上一轮回答的追问或省略了主语，请结合对话历史理解问题意图

文档片段：
{context}"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

_llm = None
_session_histories: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_histories:
        _session_histories[session_id] = InMemoryChatMessageHistory()
    return _session_histories[session_id]


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.3,
            max_tokens=2048,
        )
    return _llm


def _format_docs(docs) -> str:
    return "\n\n".join(
        f"[来源: {d.metadata.get('filename', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


def build_rag_chain(retriever):
    """Build a RAG chain that supports chat history via MessagesPlaceholder."""
    rag_chain = (
        {
            "context": itemgetter("question") | retriever | _format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | RAG_PROMPT
        | get_llm()
        | StrOutputParser()
    )
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )


def get_session_history_messages(session_id: str) -> list[HistoryMessage]:
    history = _session_histories.get(session_id)
    if not history:
        return []
    messages = []
    for msg in history.messages:
        if msg.type == "human":
            messages.append(HistoryMessage(role="human", content=msg.content))
        elif msg.type == "ai":
            messages.append(HistoryMessage(role="ai", content=str(msg.content)))
    return messages


def clear_session_history(session_id: str) -> None:
    _session_histories.pop(session_id, None)


async def ask_question(
    question: str, top_k: int = 4, session_id: str = "default"
) -> AnswerResponse:
    fetch_k = top_k * 5
    hybrid_retriever = get_hybrid_retriever(fetch_k)
    reranked_retriever = RerankedRetriever(
        base_retriever=hybrid_retriever,
        top_n=top_k,
    )
    chain = build_rag_chain(reranked_retriever)

    answer = await chain.ainvoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )

    docs = await reranked_retriever.ainvoke(question)
    sources = [
        SourceCitation(
            doc_id=doc.metadata.get("doc_id", ""),
            filename=doc.metadata.get("filename", "unknown"),
            content_snippet=doc.page_content[:200],
        )
        for doc in docs
    ]

    return AnswerResponse(
        question=question, answer=answer, sources=sources, session_id=session_id
    )
