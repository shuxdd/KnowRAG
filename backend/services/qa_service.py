from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from backend.config import get_settings
from backend.models.schemas import Source
from backend.services.hybrid_retriever import hybrid_retriever

settings = get_settings()

PROMPT_TEMPLATE = """You are an enterprise knowledge base assistant. Answer questions strictly based on the provided document context. If relevant information is not found in the context, explicitly state "未在知识库中找到相关信息". Cite specific document sources when answering.

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

    def search(self, query: str, strategy: str = "hybrid_rerank", top_k: int = 5) -> List[Document]:
        retriever_fn = self.STRATEGIES.get(strategy, hybrid_retriever.hybrid_search_with_rerank)
        return retriever_fn(query, top_k=top_k)

    def ask(self, question: str, strategy: str = "hybrid_rerank", top_k: int = 5):
        docs = self.search(question, strategy, top_k)
        if not docs:
            return {
                "answer": "未在知识库中找到相关信息。",
                "sources": [],
            }
        context = self._build_context(docs)
        messages = self.prompt.format_messages(context=context, question=question)
        response = self.llm.invoke(messages)
        return {
            "answer": response.content,
            "sources": self._extract_sources(docs),
        }


qa_service = QAService()
