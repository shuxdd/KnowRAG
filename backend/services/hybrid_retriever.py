import hashlib
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI

from backend.config import settings

HYDE_PROMPT = """You are a knowledge base assistant. Write a short passage (2-3 sentences) that answers the following question. Be factual and concise. Write in the same language as the question.

Question: {query}

Passage:"""


def _content_id(doc: Document) -> str:
    """Stable identity for deduplication across retrievers."""
    return hashlib.md5(doc.page_content.encode()).hexdigest()


def rrf_fusion(doc_lists: list[list[Document]], k: int = 60, top_n: int = 4) -> list[Document]:
    """Reciprocal Rank Fusion: merge N ranked lists into one."""
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for docs in doc_lists:
        for rank, doc in enumerate(docs):
            cid = _content_id(doc)
            scores[cid] += 1.0 / (k + rank + 1)
            doc_map[cid] = doc

    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [doc_map[cid] for cid in sorted_ids[:top_n]]


class HybridRetriever(BaseRetriever):
    """Combines vector similarity (dense), BM25 (sparse), and HyDE via RRF."""

    vector_retriever: BaseRetriever
    bm25_retriever: BM25Retriever
    rrf_k: int = 60
    fetch_k: int = 0  # how many to fetch from each, 0 = same as top_n

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._hyde_llm = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.3,
            request_timeout=10,
        )

    def _hyde_search(self, query: str, top_k: int) -> list[Document]:
        """HyDE retrieval: LLM generates hypothetical answer, embed concat(query, answer)."""
        try:
            prompt = ChatPromptTemplate.from_template(HYDE_PROMPT)
            messages = prompt.format_messages(query=query)
            response = self._hyde_llm.invoke(messages)
            hyde_answer = response.content.strip()
            if not hyde_answer:
                return []
            combined = f"{query}\n{hyde_answer}"
            orig_k = self.vector_retriever.search_kwargs.get("k", 4)
            self.vector_retriever.search_kwargs["k"] = top_k
            docs = self.vector_retriever.invoke(combined)
            self.vector_retriever.search_kwargs["k"] = orig_k
            return docs
        except Exception:
            return []

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        orig_k = self.vector_retriever.search_kwargs.get("k", 4)
        fetch_k = self.fetch_k or orig_k * 2

        # temporarily widen vector k to fetch more candidates
        self.vector_retriever.search_kwargs["k"] = fetch_k
        vec_docs = self.vector_retriever.invoke(query)
        self.vector_retriever.search_kwargs["k"] = orig_k

        bm25_docs = self.bm25_retriever.invoke(query)[:fetch_k]
        hyde_docs = self._hyde_search(query, top_k=fetch_k)

        return rrf_fusion([vec_docs, bm25_docs, hyde_docs], k=self.rrf_k, top_n=orig_k)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        orig_k = self.vector_retriever.search_kwargs.get("k", 4)
        fetch_k = self.fetch_k or orig_k * 2

        self.vector_retriever.search_kwargs["k"] = fetch_k
        vec_docs = await self.vector_retriever.ainvoke(query)
        self.vector_retriever.search_kwargs["k"] = orig_k

        bm25_docs = self.bm25_retriever.invoke(query)[:fetch_k]
        hyde_docs = self._hyde_search(query, top_k=fetch_k)

        return rrf_fusion([vec_docs, bm25_docs, hyde_docs], k=self.rrf_k, top_n=orig_k)
