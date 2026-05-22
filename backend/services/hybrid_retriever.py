"""
混合检索器模块

实现多种检索策略的混合检索：
- 向量检索（Vector Search）：使用 embedding 模型进行语义相似度检索
- BM25 检索：基于词频的经典全文检索算法
- HyDE 检索：使用 LLM 生成假设答案辅助检索
- RRF 融合：倒数排名融合算法合并多检索器结果

检索策略：
- fast: 仅向量检索
- precise: 向量 + BM25，RRF 融合
- deep: 向量 + BM25，RRF 融合 + CrossEncoder 重排序
- HyDE: 条件触发（问题简短或模糊时），作为第三条检索支路与向量、BM25 结果 RRF 融合

缓存：
- 使用 Redis 缓存检索结果，避免重复检索
"""

import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
import redis

from backend.config import get_settings

settings = get_settings()
from backend.services.parent_store import parent_store

logger = logging.getLogger(__name__)

CACHE_PREFIX = "retrieval:"


class RetrievalCache:
    """
    Redis 检索结果缓存

    使用 MD5 哈希作为缓存键，缓存检索到的文档列表。
    文档变更时自动清除缓存。
    """

    def __init__(self, redis_url: str, ttl: int = 600):
        self._client = redis.Redis.from_url(redis_url, decode_responses=False)
        self._ttl = ttl

    def build_cache_key(
        self,
        *,
        namespace: str,
        query: str,
        strategy: str | None = None,
        top_k: int | None = None,
        chat_history: str = "",
        extra: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "namespace": namespace,
            "query": query.strip(),
            "strategy": strategy,
            "top_k": top_k,
            "chat_history": chat_history.strip(),
            "extra": extra or {},
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return CACHE_PREFIX + hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, cache_key: str, *, label: str = "") -> list[Document] | None:
        try:
            raw = self._client.get(cache_key)
            if raw is None:
                logger.debug(f"Cache MISS  query='{label[:60]}'")
                return None
            data = json.loads(raw)
            logger.debug(f"Cache HIT  ({len(data)} docs)")
            return [Document(page_content=d["c"], metadata=d["m"]) for d in data]
        except Exception as e:
            logger.warning(f"Cache GET error: {e}")
            return None

    def set(self, cache_key: str, docs: list[Document], *, label: str = ""):
        try:
            data = [{"c": d.page_content, "m": d.metadata} for d in docs]
            self._client.setex(cache_key, self._ttl, json.dumps(data, ensure_ascii=False))
            logger.debug(f"Cache SET  ({len(docs)} docs, ttl={self._ttl}s, query='{label[:60]}')")
        except Exception as e:
            logger.warning(f"Cache SET error: {e}")

    def invalidate_all(self):
        """Clear all retrieval cache entries on document change."""
        try:
            keys = list(self._client.scan_iter(match=CACHE_PREFIX + "*"))
            if keys:
                self._client.delete(*keys)
        except Exception:
            logger.warning("Cache invalidation failed (Redis unreachable?)", exc_info=True)


retrieval_cache = RetrievalCache(settings.redis_url, settings.retrieval_cache_ttl)

# ==================== HyDE 提示词模板 ====================
# HyDE (Hypothetical Document Embeddings) 的核心思想：
# 让 LLM 先生成一个"假设答案"，这个假设答案包含了回答问题所需的关键信息
# 然后将"原始问题 + 假设答案"一起进行 embedding 检索
# 这样可以让检索更准确，因为假设答案的语义与真实文档更接近
HYDE_PROMPT = """You are a knowledge base assistant. Write a short passage (2-3 sentences) that answers the following question. Be factual and concise. Write in the same language as the question.

Question: {query}

Passage:"""


def _content_id(doc: Document) -> str:
    """
    生成文档内容的稳定唯一标识符

    用于跨检索器去重，基于文档内容的 MD5 哈希值
    即使文档来自不同的检索器，只要内容相同就会有相同的 ID

    Args:
        doc: LangChain Document 对象

    Returns:
        文档内容的 MD5 哈希值（十六进制字符串）
    """
    return hashlib.md5(doc.page_content.encode()).hexdigest()


def rrf_fusion(doc_lists: list[list[Document]], k: int = 60, top_n: int = 4) -> list[Document]:
    """
    倒数排名融合算法（Reciprocal Rank Fusion, RRF）

    RRF 是一种经典的多检索结果融合算法，用于合并多个排序列表。
    核心思想：对于同一个文档，如果在多个排序列表中都排名靠前，
    说明它更可能是相关的。

    公式：score(d) = Σ 1/(k + rank(d))
    - k: 融合参数，通常设为 60，k 越小排名靠前的文档优势越大
    - rank(d): 文档在某个排序列表中的排名（从 0 开始）

    Args:
        doc_lists: 多个检索器返回的文档列表
        k: RRF 融合参数，控制排名靠前文档的权重（默认 60）
        top_n: 返回融合后的前 N 个文档

    Returns:
        融合排序后的文档列表

    Example:
        假设向量检索返回 [Doc_A, Doc_B]，BM25 返回 [Doc_B, Doc_C]
        RRF 会综合两个列表的排名，计算每个文档的总得分
    """
    # 累计每个文档的 RRF 分数
    scores: dict[str, float] = defaultdict(float)
    # 存储文档 ID 到文档对象的映射
    doc_map: dict[str, Document] = {}

    for docs in doc_lists:
        for rank, doc in enumerate(docs):
            cid = _content_id(doc)  # 获取文档唯一标识
            # RRF 公式：1/(k + rank+1)，rank+1 是因为 rank 从 0 开始
            scores[cid] += 1.0 / (k + rank + 1)
            doc_map[cid] = doc

    # 按 RRF 分数降序排列
    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    # 返回前 top_n 个文档
    return [doc_map[cid] for cid in sorted_ids[:top_n]]


def _max_leaf_score_per_parent(leaves: list[Document]) -> dict[str, float]:
    scores: dict[str, list[float]] = {}
    for leaf in leaves:
        pid = leaf.metadata.get("parent_id", "")
        score = leaf.metadata.get("score", 0.0)
        scores.setdefault(pid, []).append(score)
    return {pid: max(scores) for pid, scores in scores.items()}


class HybridRetriever(BaseRetriever):
    """
    混合检索器

    结合三种检索策略：
    1. 向量检索（Dense）：使用 embedding 模型将文本转为向量，进行语义相似度搜索
    2. BM25 检索（Sparse）：基于词频的经典全文检索算法
    3. HyDE 检索：让 LLM 生成假设答案，用假设答案辅助检索

    三种检索结果通过 RRF（倒数排名融合）算法合并，
    最终返回一个排序后的文档列表。

    继承自 LangChain 的 BaseRetriever，可以直接与 LangChain 生态集成。
    """

    # 向量检索器（需要在初始化时注入）
    vector_retriever: BaseRetriever
    # BM25 检索器（需要在初始化时注入）
    bm25_retriever: BaseRetriever
    # RRF 融合参数 k，控制排名靠前文档的优势程度
    rrf_k: int = 60
    # 从每个检索器获取的候选文档数量
    fetch_k: int = 10

    parent_crop_paragraphs: int = 3

    def __init__(self, **kwargs):
        """
        初始化混合检索器

        初始化 HyDE 专用的 LLM（用于生成假设答案）
        """
        super().__init__(**kwargs)
        self.parent_crop_paragraphs = kwargs.get("parent_crop_paragraphs", 3)
        self._hyde_llm = ChatOpenAI(
            model=settings.qwen_model,
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            temperature=0.3,
            request_timeout=10,
        )

    def rebuild_bm25(self):
        """Rebuild BM25 index from current ChromaDB chunks. Call after document add/delete."""
        from backend.services.vector_service import vector_service

        docs = vector_service.get_all_chunks()
        if docs:
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            logger.info(f"BM25 index rebuilt with {len(docs)} chunks")
        else:
            self.bm25_retriever = BM25Retriever.from_documents(
                [Document(page_content="__placeholder__", metadata={})]
            )
            logger.info("BM25 index reset (no chunks)")

    def _hyde_search(self, query: str, top_k: int) -> list[Document]:
        """
        HyDE 检索策略

        HyDE (Hypothetical Document Embeddings) 的工作流程：
        1. 用 LLM 根据问题生成一个简短的假设答案
        2. 将"原始问题 + 假设答案"拼接在一起
        3. 对拼接后的文本进行 embedding
        4. 返回与拼接文本最相似的真实文档

        原理：假设答案中包含了回答问题所需的关键信息和上下文，
        这些内容与真实文档的 embedding 更接近，因此检索更准确。

        Args:
            query: 原始查询问题
            top_k: 要返回的文档数量

        Returns:
            与假设答案最相似的文档列表
        """
        try:
            # 构建 HyDE 提示词
            prompt = ChatPromptTemplate.from_template(HYDE_PROMPT)
            messages = prompt.format_messages(query=query)
            # 调用 LLM 生成假设答案
            response = self._hyde_llm.invoke(messages)
            hyde_answer = response.content.strip()

            if not hyde_answer:
                return []

            # 将原始问题与假设答案拼接
            # 拼接后的文本既包含查询意图，又包含答案的语义信息
            combined = f"{query}\n{hyde_answer}"

            # 临时扩大向量检索的 k 值，获取更多候选文档
            orig_k = self.vector_retriever.search_kwargs.get("k", 4)
            self.vector_retriever.search_kwargs["k"] = top_k
            # 执行向量检索
            docs = self.vector_retriever.invoke(combined)
            # 恢复原来的 k 值
            self.vector_retriever.search_kwargs["k"] = orig_k

            return docs
        except Exception:
            logger.warning("HyDE search failed, skipping HyDE branch", exc_info=True)
            return []

    def _crop_parent(self, query: str, content: str) -> str:
        """Keep only the top paragraphs by reranker score against query."""
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if len(paragraphs) <= self.parent_crop_paragraphs:
            return content

        from backend.services.reranker import reranker

        para_docs = [Document(page_content=p) for p in paragraphs]
        scored = reranker.rerank(query, para_docs, top_n=len(paragraphs))
        top_paras = sorted(
            scored,
            key=lambda d: d.metadata.get("score", 0),
            reverse=True,
        )[:self.parent_crop_paragraphs]
        # Restore original order
        top_paras.sort(key=lambda d: paragraphs.index(d.page_content))
        return "\n\n".join(d.page_content for d in top_paras)

    @staticmethod
    def _should_hyde(query: str) -> bool:
        """HyDE is triggered for short or vague queries that lack semantic signals."""
        if len(query) < 20:
            return True
        vague = ["是什么", "怎么", "如何", "什么叫", "什么是", "怎么做", "怎样", "干啥", "干吗"]
        return any(w in query for w in vague)

    def _expand_to_parents(self, leaves: list[Document], top_n: int, query: str = "") -> list[Document]:
        parent_ids_ordered: list[str] = []
        seen: set[str] = set()
        for leaf in leaves:
            pid = leaf.metadata.get("parent_id")
            if pid and pid not in seen:
                parent_ids_ordered.append(pid)
                seen.add(pid)

        parents = parent_store.get_by_ids(parent_ids_ordered)
        parent_map = {p.id: p for p in parents}
        leaf_scores = _max_leaf_score_per_parent(leaves)

        ordered_docs = []
        for pid in parent_ids_ordered:
            if pid not in parent_map:
                continue
            p = parent_map[pid]
            content = self._crop_parent(query, p.content) if query else p.content
            ordered_docs.append(Document(
                page_content=content,
                metadata={
                    "doc_id": pid,
                    "filename": p.filename,
                    "heading_path": p.heading_path,
                    "page_start": p.page_start,
                    "page_end": p.page_end,
                    "score": leaf_scores.get(pid, 0.0),
                },
            ))
        return ordered_docs[:top_n]

    def _fast_retrieve(self, query: str, top_k: int) -> list[Document]:
        """Fast strategy: vector retrieval only, no BM25/HyDE/Reranker."""
        orig_k = self.vector_retriever.search_kwargs.get("k", 4)
        self.vector_retriever.search_kwargs["k"] = top_k * 2
        docs = self.vector_retriever.invoke(query)
        self.vector_retriever.search_kwargs["k"] = orig_k
        return self._expand_to_parents(docs, top_n=top_k, query=query)

    def _precise_retrieve(self, query: str, top_k: int) -> list[Document]:
        """Precise strategy: vector + BM25 -> RRF fusion, HyDE on demand."""
        orig_k = self.vector_retriever.search_kwargs.get("k", 4)
        fetch_k = self.fetch_k if self.fetch_k else 10

        self.vector_retriever.search_kwargs["k"] = fetch_k
        vec_docs = self.vector_retriever.invoke(query)
        self.vector_retriever.search_kwargs["k"] = orig_k

        bm25_docs = self.bm25_retriever.invoke(query)[:fetch_k]

        doc_lists = [vec_docs, bm25_docs]
        if self._should_hyde(query):
            hyde_docs = self._hyde_search(query, top_k=fetch_k)
            doc_lists.append(hyde_docs)

        fused = rrf_fusion(doc_lists, k=self.rrf_k, top_n=top_k)
        return self._expand_to_parents(fused, top_n=top_k, query=query)

    def _deep_retrieve(self, query: str, top_k: int) -> list[Document]:
        """Deep strategy: vector + BM25 -> RRF -> Reranker, HyDE on demand."""
        use_hyde = self._should_hyde(query)
        cache_key = retrieval_cache.build_cache_key(
            namespace="hybrid_deep",
            query=query,
            strategy="deep",
            top_k=top_k,
            extra={"hyde": use_hyde},
        )
        cached = retrieval_cache.get(cache_key, label=query)
        if cached:
            return cached[:top_k]

        from backend.services.reranker import reranker

        orig_k = self.vector_retriever.search_kwargs.get("k", 4)
        fetch_k = self.fetch_k if self.fetch_k else 10

        self.vector_retriever.search_kwargs["k"] = fetch_k
        vec_docs = self.vector_retriever.invoke(query)
        self.vector_retriever.search_kwargs["k"] = orig_k

        bm25_docs = self.bm25_retriever.invoke(query)[:fetch_k]

        doc_lists = [vec_docs, bm25_docs]
        if use_hyde:
            hyde_docs = self._hyde_search(query, top_k=fetch_k)
            doc_lists.append(hyde_docs)

        fused = rrf_fusion(doc_lists, k=self.rrf_k, top_n=10)
        reranked = reranker.rerank(query, fused, top_n=top_k)
        result = self._expand_to_parents(reranked, top_n=top_k, query=query)
        retrieval_cache.set(cache_key, result, label=query)
        return result

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        orig_k = self.vector_retriever.search_kwargs.get("k", 4)
        return self._deep_retrieve(query, top_k=orig_k)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        use_hyde = self._should_hyde(query)
        cache_key = retrieval_cache.build_cache_key(
            namespace="hybrid_deep",
            query=query,
            strategy="deep",
            top_k=self.vector_retriever.search_kwargs.get("k", 4),
            extra={"hyde": use_hyde},
        )
        cached = retrieval_cache.get(cache_key, label=query)
        if cached:
            return cached[:self.vector_retriever.search_kwargs.get("k", 4)]

        from backend.services.reranker import reranker

        orig_k = self.vector_retriever.search_kwargs.get("k", 4)
        fetch_k = self.fetch_k if self.fetch_k else 10

        self.vector_retriever.search_kwargs["k"] = fetch_k
        try:
            async def _vec():
                return await self.vector_retriever.ainvoke(query)

            async def _bm25():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.bm25_retriever.invoke, query)

            async def _hyde():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._hyde_search, query, fetch_k)

            if use_hyde:
                vec_docs, bm25_raw, hyde_docs = await asyncio.gather(
                    _vec(), _bm25(), _hyde()
                )
            else:
                vec_docs, bm25_raw = await asyncio.gather(_vec(), _bm25())
                hyde_docs = []
        finally:
            self.vector_retriever.search_kwargs["k"] = orig_k

        bm25_docs = bm25_raw[:fetch_k]

        doc_lists = [vec_docs, bm25_docs]
        if hyde_docs:
            doc_lists.append(hyde_docs)

        fused = rrf_fusion(doc_lists, k=self.rrf_k, top_n=10)
        reranked = reranker.rerank(query, fused, top_n=orig_k)
        result = self._expand_to_parents(reranked, top_n=orig_k, query=query)
        retrieval_cache.set(cache_key, result, label=query)
        return result


class _VectorServiceRetriever(BaseRetriever):
    """Thin BaseRetriever adapter around VectorService.similarity_search."""

    vector_service: Any = None
    search_kwargs: dict = {}

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        k = self.search_kwargs.get("k", 4)
        return self.vector_service.similarity_search(query, k=k)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


def _build_hybrid_retriever() -> HybridRetriever:
    from backend.services.vector_service import vector_service

    vec_retriever = _VectorServiceRetriever(
        vector_service=vector_service,
        search_kwargs={"k": 3},
    )

    retriever = HybridRetriever(
        vector_retriever=vec_retriever,
        bm25_retriever=BM25Retriever.from_documents(
            [Document(page_content="__placeholder__", metadata={})]
        ),
    )
    retriever.rebuild_bm25()
    return retriever


hybrid_retriever: HybridRetriever = _build_hybrid_retriever()
