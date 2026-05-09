import os
from typing import List
from langchain_core.documents import Document
from backend.config import get_settings

settings = get_settings()


class Reranker:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(settings.reranker_model)
        return self._model

    def rerank(
        self, query: str, docs: List[Document], top_n: int = 5
    ) -> List[Document]:
        if not docs:
            return []
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.model.predict(pairs)
        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_docs = []
        for doc, score in scored[:top_n]:
            doc.metadata["score"] = float(score)
            top_docs.append(doc)
        return top_docs


reranker = Reranker()
