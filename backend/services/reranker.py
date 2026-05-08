import os

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import run_in_executor
from pydantic import Field


class Reranker:
    """Cross-encoder reranker using BGE-reranker-v2-m3 via sentence_transformers."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[Document], top_n: int) -> list[Document]:
        if len(documents) <= top_n:
            return documents

        model = self._load()
        pairs = [[query, doc.page_content] for doc in documents]
        scores = model.predict(pairs, show_progress_bar=False)

        if isinstance(scores, float):
            scores = [scores]

        scored = list(zip(scores, documents))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_n]]


class RerankedRetriever(BaseRetriever):
    """Fetches candidates via a base retriever, then re-scores with a cross-encoder.

    The base retriever should already be configured to return more candidates
    than top_n (e.g. top_n * 3). The reranker will trim results down to top_n.
    """

    base_retriever: BaseRetriever
    reranker: Reranker = Field(default_factory=Reranker)
    top_n: int = 4

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        candidates = self.base_retriever.invoke(query)
        return self.reranker.rerank(query, candidates, self.top_n)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        candidates = await self.base_retriever.ainvoke(query)
        return await run_in_executor(None, self.reranker.rerank, query, candidates, self.top_n)
