from typing import List
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from backend.services.vector_service import vector_service
from backend.services.reranker import reranker


class HybridRetriever:
    def __init__(self):
        self._bm25 = None
        self._corpus_texts: List[str] = []
        self._corpus_docs: List[Document] = []

    def _ensure_bm25(self):
        all_docs = vector_service.get_all_chunks()
        if not all_docs:
            return
        current_ids = {d.metadata.get("doc_id", "") for d in all_docs}
        cached_ids = {d.metadata.get("doc_id", "") for d in self._corpus_docs}
        if current_ids != cached_ids or not self._bm25:
            self._corpus_docs = all_docs
            self._corpus_texts = [d.page_content for d in all_docs]
            if self._corpus_texts:
                tokenized = [text.split() for text in self._corpus_texts]
                self._bm25 = BM25Okapi(tokenized)

    def vector_search(self, query: str, top_k: int = 10) -> List[Document]:
        return vector_service.similarity_search(query, k=top_k)

    def _dedup_docs(self, docs: List[Document]) -> List[Document]:
        seen = set()
        unique = []
        for doc in docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    def hybrid_search(self, query: str, top_k: int = 10) -> List[Document]:
        vector_docs = vector_service.similarity_search(query, k=top_k)
        self._ensure_bm25()
        bm25_docs = []
        if self._bm25:
            tokenized_query = query.split()
            bm25_scores = self._bm25.get_scores(tokenized_query)
            scored = sorted(
                zip(self._corpus_docs, bm25_scores),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]
            for doc, score in scored:
                doc.metadata["score"] = float(score)
                bm25_docs.append(doc)
        return self._dedup_docs(vector_docs + bm25_docs)[:top_k]

    def hybrid_search_with_rerank(
        self, query: str, top_k: int = 10, top_n: int = 5
    ) -> List[Document]:
        candidates = self.hybrid_search(query, top_k=top_k)
        return reranker.rerank(query, candidates, top_n=top_n)


hybrid_retriever = HybridRetriever()
