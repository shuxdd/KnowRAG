from typing import List
import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from backend.services.vector_service import vector_service
from backend.services.reranker import reranker


class HybridRetriever:
    """
    混合检索器
    结合向量检索和 BM25 全文检索两种方法
    提供三种检索策略：纯向量、纯 BM25、混合 + 重排序
    """

    def __init__(self):
        """
        初始化混合检索器
        - _bm25: BM25 索引对象，懒加载
        - _corpus_texts: 语料库的文本列表，用于构建 BM25 索引
        - _corpus_docs: 语料库的 Document 对象列表
        """
        self._bm25 = None
        self._corpus_texts: List[str] = []
        self._corpus_docs: List[Document] = []

    def _ensure_bm25(self):
        """
        确保 BM25 索引已构建且为最新
        当文档集合发生变化时会重新构建索引
        采用懒加载策略，只在首次使用时构建
        """
        all_docs = vector_service.get_all_chunks()
        if not all_docs:
            return
        # 检查当前文档 ID 集合与缓存的文档 ID 集合是否一致
        current_ids = {d.metadata.get("doc_id", "") for d in all_docs}
        cached_ids = {d.metadata.get("doc_id", "") for d in self._corpus_docs}
        if current_ids != cached_ids or not self._bm25:
            # 文档有更新，重新构建 BM25 索引
            self._corpus_docs = all_docs
            self._corpus_texts = [d.page_content for d in all_docs]
            if self._corpus_texts:
                # 对文本进行分词（使用 jieba 中文分词）
                tokenized = [jieba.lcut(text) for text in self._corpus_texts]
                self._bm25 = BM25Okapi(tokenized)

    def vector_search(self, query: str, top_k: int = 10) -> List[Document]:
        """
        纯向量检索策略
        使用 embedding 模型将查询转换为向量，然后进行相似度搜索

        Args:
            query: 查询文本
            top_k: 返回的最相似文档数量

        Returns:
            按相似度降序排列的 Document 列表
        """
        return vector_service.similarity_search(query, k=top_k)

    def rrf_fusion(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        k: int = 60,
        top_k: int = 10,
    ) -> List[Document]:
        scores: dict[str, tuple[float, Document]] = {}

        for rank, doc in enumerate(vector_docs):
            key = doc.page_content[:100]
            rrf = 1.0 / (k + rank + 1)
            if key in scores:
                scores[key] = (scores[key][0] + rrf, scores[key][1])
            else:
                scores[key] = (rrf, doc)

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content[:100]
            rrf = 1.0 / (k + rank + 1)
            if key in scores:
                scores[key] = (scores[key][0] + rrf, scores[key][1])
            else:
                scores[key] = (rrf, doc)

        sorted_docs = sorted(scores.values(), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in sorted_docs[:top_k]]

    def hybrid_search(self, query: str, top_k: int = 10) -> List[Document]:
        """
        混合检索策略
        同时执行向量检索和 BM25 检索，然后使用 RRF 算法融合

        Args:
            query: 查询文本
            top_k: 每种检索方法返回的文档数量

        Returns:
            混合检索结果（通过 RRF 算法融合向量和 BM25 排名，按 RRF 分数降序取前 top_k 个）
        """
        # 1. 向量检索
        vector_docs = vector_service.similarity_search(query, k=top_k)
        # 2. 确保 BM25 索引已构建
        self._ensure_bm25()
        bm25_docs = []
        if self._bm25:
            # 对查询进行分词
            tokenized_query = jieba.lcut(query)
            # 计算查询与所有文档的 BM25 分数
            bm25_scores = self._bm25.get_scores(tokenized_query)
            # 按分数降序排列
            scored = sorted(
                zip(self._corpus_docs, bm25_scores),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]
            # 提取文档并保存 BM25 分数到元数据
            for doc, score in scored:
                doc.metadata["score"] = float(score)
                bm25_docs.append(doc)
        # 3. RRF 融合两种检索结果
        return self.rrf_fusion(vector_docs, bm25_docs, top_k=top_k)

    def hybrid_search_with_rerank(
        self, query: str, top_k: int = 10, top_n: int = 3
    ) -> List[Document]:
        """
        混合检索 + 重排序策略
        先进行混合检索获取候选文档，然后用交叉编码器重排序

        Args:
            query: 查询文本
            top_k: 混合检索阶段返回的候选文档数量
            top_n: 重排序后返回的最终文档数量

        Returns:
            最终排序结果（top_n 个最相关的 Document）
        """
        # 1. 先进行混合检索获取候选文档
        candidates = self.hybrid_search(query, top_k=top_k)
        # 2. 使用重排序模型对候选文档进行精细排序
        return reranker.rerank(query, candidates, top_n=top_n)


# 全局单例实例
hybrid_retriever = HybridRetriever()
