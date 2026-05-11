"""Tests for hybrid_retriever — jieba tokenization and RRF fusion."""
import pytest
from langchain_core.documents import Document
from backend.services.hybrid_retriever import HybridRetriever


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
    """Verify RRF fusion correctly ranks documents from two retrieval sources."""

    def test_empty_both_returns_empty(self):
        retriever = HybridRetriever()
        result = retriever.rrf_fusion([], [], top_k=5)
        assert result == []

    def test_only_vector_docs_returns_them_in_order(self):
        retriever = HybridRetriever()
        docs = [make_doc("doc A"), make_doc("doc B"), make_doc("doc C")]
        result = retriever.rrf_fusion(docs, [], top_k=3)
        assert result == docs

    def test_duplicate_doc_in_both_lists_gets_combined_rrf_score(self):
        retriever = HybridRetriever()
        shared = make_doc("shared content")
        vector_docs = [shared, make_doc("vec only")]
        bm25_docs = [shared, make_doc("bm25 only")]
        result = retriever.rrf_fusion(vector_docs, bm25_docs, top_k=5)
        # shared doc should appear once and rank first (combined score)
        assert len(result) == 3
        assert result[0] == shared

    def test_top_k_truncates_correctly(self):
        retriever = HybridRetriever()
        vector_docs = [make_doc(f"vec_{i}") for i in range(10)]
        bm25_docs = [make_doc(f"bm25_{i}") for i in range(10)]
        result = retriever.rrf_fusion(vector_docs, bm25_docs, top_k=5)
        assert len(result) == 5

    def test_ranking_stable_with_different_k(self):
        retriever = HybridRetriever()
        docs = [make_doc(f"doc_{i}") for i in range(3)]
        result_k60 = retriever.rrf_fusion(docs, [], k=60, top_k=3)
        result_k10 = retriever.rrf_fusion(docs, [], k=10, top_k=3)
        # Both should return same order when no competition
        assert result_k60 == result_k10 == docs
