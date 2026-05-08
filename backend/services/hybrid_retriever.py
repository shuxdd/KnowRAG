import hashlib
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever


def _content_id(doc: Document) -> str:
    """Stable identity for deduplication across retrievers."""
    return hashlib.md5(doc.page_content.encode()).hexdigest()


def rrf_fusion(results_a: list[Document], results_b: list[Document], k: int = 60, top_n: int = 4) -> list[Document]:
    """Reciprocal Rank Fusion: merge two ranked lists into one."""
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(results_a):
        cid = _content_id(doc)
        scores[cid] += 1.0 / (k + rank + 1)
        doc_map[cid] = doc

    for rank, doc in enumerate(results_b):
        cid = _content_id(doc)
        scores[cid] += 1.0 / (k + rank + 1)
        doc_map[cid] = doc

    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [doc_map[cid] for cid in sorted_ids[:top_n]]


class HybridRetriever(BaseRetriever):
    """Combines vector similarity (dense) and BM25 (sparse) via RRF."""

    vector_retriever: BaseRetriever
    bm25_retriever: BM25Retriever
    rrf_k: int = 60
    fetch_k: int = 0  # how many to fetch from each, 0 = same as top_n

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        fetch_k = self.fetch_k or self.vector_retriever.search_kwargs.get("k", 4) * 2

        # temporarily widen vector k to fetch more candidates
        orig_k = self.vector_retriever.search_kwargs.get("k", 4)
        self.vector_retriever.search_kwargs["k"] = fetch_k
        vec_docs = self.vector_retriever.invoke(query)
        self.vector_retriever.search_kwargs["k"] = orig_k

        bm25_docs = self.bm25_retriever.invoke(query)[:fetch_k]

        top_n = orig_k
        return rrf_fusion(vec_docs, bm25_docs, k=self.rrf_k, top_n=top_n)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        fetch_k = self.fetch_k or self.vector_retriever.search_kwargs.get("k", 4) * 2

        orig_k = self.vector_retriever.search_kwargs.get("k", 4)
        self.vector_retriever.search_kwargs["k"] = fetch_k
        vec_docs = await self.vector_retriever.ainvoke(query)
        self.vector_retriever.search_kwargs["k"] = orig_k

        bm25_docs = self.bm25_retriever.invoke(query)[:fetch_k]

        return rrf_fusion(vec_docs, bm25_docs, k=self.rrf_k, top_n=orig_k)
