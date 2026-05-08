from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from backend.config import settings
from backend.services.vector_service import get_hybrid_retriever
from backend.models.schemas import AnswerResponse, SourceCitation

RAG_SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请根据提供的文档片段回答问题。

规则：
1. 仅根据提供的文档片段回答，不要使用外部知识
2. 如果文档片段不足以回答问题，请明确说明"根据提供的文档无法回答此问题"
3. 回答要简洁、准确，使用中文
4. 如果回答引用了文档内容，请注明来源

文档片段：
{context}"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "{question}"),
])


_llm = None


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
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | get_llm()
        | StrOutputParser()
    )
    return chain


async def ask_question(question: str, top_k: int = 4) -> AnswerResponse:
    retriever = get_hybrid_retriever(top_k)
    chain = build_rag_chain(retriever)

    answer = await chain.ainvoke(question)

    docs = await retriever.ainvoke(question)
    sources = [
        SourceCitation(
            doc_id=doc.metadata.get("doc_id", ""),
            filename=doc.metadata.get("filename", "unknown"),
            content_snippet=doc.page_content[:200],
        )
        for doc in docs
    ]

    return AnswerResponse(question=question, answer=answer, sources=sources)
