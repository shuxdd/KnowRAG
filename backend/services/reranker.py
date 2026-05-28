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
import logging
from typing import List
from langchain_core.documents import Document

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logger = logging.getLogger(__name__)

from backend.config import get_settings

settings = get_settings()


class Reranker:
    """
    重排序服务

    使用交叉编码器对候选文档进行精细排序。
    模型懒加载，首次调用 rerank() 时才加载。
    """

    def __init__(self):
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        device_str = settings.reranker_device if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading reranker model: {settings.reranker_model} on {device_str}")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self._device = torch.device(device_str)
        self._tokenizer = AutoTokenizer.from_pretrained(
            settings.reranker_model, local_files_only=True
        )
        dtype = torch.float16 if device_str == "cuda" else torch.float32
        self._model = AutoModelForSequenceClassification.from_pretrained(
            settings.reranker_model, local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self._device)
        self._model.eval()

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
        self._ensure_model()
        import torch
        # 构建查询-文档对并批量编码
        pairs = [[query, doc.page_content] for doc in docs]
        inputs = self._tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        scores = outputs.logits.squeeze(-1).tolist()
        if not isinstance(scores, list):
            scores = [scores]
        # 将文档与分数配对并按分数降序排列
        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        # 取前 top_n 个文档
        top_docs = []
        for doc, score in scored[:top_n]:
            doc.metadata["score"] = float(score)
            top_docs.append(doc)
        return top_docs


# 全局单例（模型尚未加载，首次 rerank() 时才加载）
reranker = Reranker()
