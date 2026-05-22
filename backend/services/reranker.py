"""
重排序服务模块

使用交叉编码器（Cross-Encoder）对候选文档进行精细排序。

特点：
- 比双编码器（Bi-Encoder）精度更高
- 同时编码查询和文档，计算相关性分数
- 使用 BGE Reranker 模型

使用场景：
- deep 检索策略的最后一步
- 对混合检索结果进行二次排序
"""

import os
from typing import List
from langchain_core.documents import Document
from backend.config import get_settings

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
settings = get_settings()


class Reranker:
    """
    重排序服务

    使用交叉编码器对候选文档进行精细排序。
    模型懒加载，首次使用时才从 HuggingFace 加载。
    """

    def __init__(self):
        """
        初始化重排序器
        _model 属性在首次使用时懒加载
        """
        self._model = None

    @property
    def model(self):
        """
        懒加载重排序模型
        首次访问时从 HuggingFace 加载 BGE Reranker 模型

        Returns:
            CrossEncoder 模型实例
        """
        if self._model is None:
            os.environ["HF_ENDPOINT"] = settings.hf_endpoint
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(settings.reranker_model)
        return self._model

    def rerank(
        self, query: str, docs: List[Document], top_n: int = 3
    ) -> List[Document]:
        """
        对候选文档进行重排序

        Args:
            query: 查询文本
            docs: 候选文档列表（通常来自混合检索）
            top_n: 返回的最相关文档数量

        Returns:
            重排序后的 Document 列表（取前 top_n 个）
        """
        if not docs:
            return []
        # 构建查询-文档对列表
        pairs = [[query, doc.page_content] for doc in docs]
        # 使用交叉编码器预测相关性分数
        scores = self.model.predict(pairs)
        # 将文档与分数配对并按分数降序排列
        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        # 取前 top_n 个文档
        top_docs = []
        for doc, score in scored[:top_n]:
            doc.metadata["score"] = float(score)  # 将分数存入元数据
            top_docs.append(doc)
        return top_docs


# 全局单例实例
reranker = Reranker()
