"""Tests for hybrid_retriever — jieba tokenization, RRF fusion, and retrieval strategies."""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from backend.services.hybrid_retriever import HybridRetriever, rrf_fusion


class _MockRetriever(BaseRetriever):
    """Test double that returns preset documents and records call count."""
    docs: list[Document] = []
    call_count: int = 0
    search_kwargs: dict = {"k": 3}

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        self.call_count += 1
        return self.docs

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


def make_doc(content: str) -> Document:
    return Document(page_content=content, metadata={})


class TestJiebaTokenization:
    """Verify jieba correctly tokenizes Chinese text in BM25 index building."""

    def test_chinese_text_is_properly_segmented(self):
        import jieba
        tokens = jieba.lcut("企业知识库测试")
        assert len(tokens) >= 2  # Chinese should be segmented into multiple tokens
        assert "企业" in tokens or "知识" in tokens or "知识库" in tokens

    def test_query_segmentation(self):
        import jieba
        tokens = jieba.lcut("如何优化检索质量")
        assert len(tokens) >= 3


class TestRRFFusion:
    """Verify RRF fusion correctly ranks documents from multiple retrieval sources."""

    def test_empty_both_returns_empty(self):
        result = rrf_fusion([[]], top_n=5)
        assert result == []

    def test_only_vector_docs_returns_them_in_order(self):
        docs = [make_doc("doc A"), make_doc("doc B"), make_doc("doc C")]
        result = rrf_fusion([docs], top_n=3)
        assert result == docs

    def test_duplicate_doc_in_both_lists_gets_combined_rrf_score(self):
        shared = make_doc("shared content")
        vector_docs = [shared, make_doc("vec only")]
        bm25_docs = [shared, make_doc("bm25 only")]
        result = rrf_fusion([vector_docs, bm25_docs], top_n=5)
        # shared doc should appear once and rank first (combined score)
        assert len(result) == 3
        assert result[0] == shared

    def test_top_n_truncates_correctly(self):
        vector_docs = [make_doc(f"vec_{i}") for i in range(10)]
        bm25_docs = [make_doc(f"bm25_{i}") for i in range(10)]
        result = rrf_fusion([vector_docs, bm25_docs], top_n=5)
        assert len(result) == 5

    def test_ranking_stable_with_different_k(self):
        docs = [make_doc(f"doc_{i}") for i in range(3)]
        result_k60 = rrf_fusion([docs], k=60, top_n=3)
        result_k10 = rrf_fusion([docs], k=10, top_n=3)
        # Both should return same order when no competition
        assert result_k60 == result_k10 == docs


class TestRetrieveStrategies:
    """Verify three retrieval strategies return correct results and call correct components."""

    def _make_doc(self, content: str, parent_id: str = "p1", score: float = 0.9) -> Document:
        return Document(page_content=content, metadata={"parent_id": parent_id, "score": score})

    def _build_retriever(self, vec_docs=None, bm25_docs=None):
        vec = _MockRetriever(docs=vec_docs or [])
        bm25 = _MockRetriever(docs=bm25_docs or [])
        retriever = HybridRetriever(
            vector_retriever=vec,
            bm25_retriever=bm25,
            fetch_k=5,
        )
        return retriever, vec, bm25

    def test_fast_retrieve_returns_list(self):
        """_fast_retrieve returns a list of Documents."""
        doc = self._make_doc("fast content")
        retriever, vec, bm25 = self._build_retriever(vec_docs=[doc])
        with patch.object(retriever, "_expand_to_parents", return_value=[doc]):
            result = retriever._fast_retrieve("test", top_k=3)
            assert isinstance(result, list)
            assert len(result) >= 1
            assert result[0].page_content == "fast content"

    def test_fast_retrieve_calls_only_vector(self):
        """_fast_retrieve only calls vector_retriever, not BM25."""
        doc = self._make_doc("fast content")
        retriever, vec, bm25 = self._build_retriever(vec_docs=[doc])
        with patch.object(retriever, "_expand_to_parents", return_value=[doc]):
            retriever._fast_retrieve("test", top_k=3)
            assert vec.call_count >= 1
            assert bm25.call_count == 0

    def test_fast_retrieve_restores_orig_k_after(self):
        """_fast_retrieve restores vector_retriever.search_kwargs k after invocation."""
        doc = self._make_doc("content")
        retriever, vec, bm25 = self._build_retriever(vec_docs=[doc])
        orig_k = vec.search_kwargs.get("k", 4)
        with patch.object(retriever, "_expand_to_parents", return_value=[doc]):
            retriever._fast_retrieve("test", top_k=3)
        assert vec.search_kwargs.get("k") == orig_k

    def test_precise_retrieve_calls_both_vector_and_bm25(self):
        """_precise_retrieve calls both vector and BM25 retrievers."""
        doc = self._make_doc("precise content")
        retriever, vec, bm25 = self._build_retriever(vec_docs=[doc], bm25_docs=[doc])
        with patch.object(retriever, "_expand_to_parents", return_value=[doc]):
            retriever._precise_retrieve("test", top_k=3)
            assert vec.call_count >= 1
            assert bm25.call_count >= 1

    def test_precise_retrieve_returns_list(self):
        """_precise_retrieve returns a list of Documents."""
        doc = self._make_doc("precise content")
        retriever, vec, bm25 = self._build_retriever(vec_docs=[doc], bm25_docs=[doc])
        with patch.object(retriever, "_expand_to_parents", return_value=[doc]):
            result = retriever._precise_retrieve("test", top_k=3)
            assert isinstance(result, list)
            assert len(result) >= 1

    def test_precise_retrieve_restores_orig_k(self):
        """_precise_retrieve restores vector_retriever.search_kwargs k."""
        doc = self._make_doc("content")
        retriever, vec, bm25 = self._build_retriever(vec_docs=[doc], bm25_docs=[doc])
        orig_k = vec.search_kwargs.get("k", 4)
        with patch.object(retriever, "_expand_to_parents", return_value=[doc]):
            retriever._precise_retrieve("test", top_k=3)
        assert vec.search_kwargs.get("k") == orig_k

    def test_deep_retrieve_is_full_pipeline(self):
        """_deep_retrieve uses vector + BM25 + Reranker."""
        doc = self._make_doc("deep content")
        retriever, vec, bm25 = self._build_retriever(vec_docs=[doc], bm25_docs=[doc])
        with patch.object(retriever, "_expand_to_parents", return_value=[doc]), \
             patch("backend.services.reranker.reranker") as mock_reranker, \
             patch("backend.services.hybrid_retriever.retrieval_cache.get", return_value=None):
            mock_reranker.rerank.return_value = [doc]
            result = retriever._deep_retrieve("test", top_k=3)
            assert vec.call_count >= 1
            assert bm25.call_count >= 1
            assert isinstance(result, list)

    def test_deep_retrieve_restores_orig_k(self):
        """_deep_retrieve restores vector_retriever.search_kwargs k."""
        doc = self._make_doc("content")
        retriever, vec, bm25 = self._build_retriever(vec_docs=[doc], bm25_docs=[doc])
        orig_k = vec.search_kwargs.get("k", 4)
        with patch.object(retriever, "_expand_to_parents", return_value=[doc]), \
             patch("backend.services.reranker.reranker") as mock_reranker, \
             patch("backend.services.hybrid_retriever.retrieval_cache.get", return_value=None):
            mock_reranker.rerank.return_value = [doc]
            retriever._deep_retrieve("test", top_k=3)
        assert vec.search_kwargs.get("k") == orig_k
