"""
混合检索器模块

实现多种检索策略的混合检索：
- 向量检索（Vector Search）：使用 embedding 模型进行语义相似度检索
- BM25 检索：基于词频的经典全文检索算法
- RRF 融合：倒数排名融合算法合并多检索器结果

检索策略：
- fast: 仅向量检索
- precise: 向量 + BM25，RRF 融合
- deep: 向量 + BM25，RRF 融合 + CrossEncoder 重排序

缓存：
- 使用 Redis 缓存检索结果，避免重复检索
"""

import asyncio
import concurrent.futures
import contextvars
import hashlib
import json
import logging
from collections import defaultdict
from typing import Any

import numpy as np

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
import redis

from backend.config import get_settings

settings = get_settings()
from backend.db import SessionFactory
from backend.models.db_models import RetrievalStatsORM
from backend.services.parent_store import parent_store
from backend.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

CACHE_PREFIX = "retrieval:"

# 用于向外部暴露检索中间过程的 contextvar
_retrieval_progress: contextvars.ContextVar[list[dict]] = contextvars.ContextVar(
    "_retrieval_progress", default=[]
)


def _record_progress(stage: str, text: str):
    """Append a progress entry for the current retrieval."""
    entries = _retrieval_progress.get()
    entries.append({"stage": stage, "text": text})
    _retrieval_progress.set(entries)


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


def mmr_select(
    docs: list[Document],
    embeddings: list[list[float]],
    scores: list[float] | None = None,
    top_n: int = 5,
    lambda_mult: float = 0.6,
) -> list[Document]:
    """
    最大边际相关性（MMR）去重选择

    在选下一个结果时同时考虑与 query 的相关性和与已选结果的差异性，
    避免返回高度相似的冗余内容。

    公式：MMR(d) = λ × relevance(d) - (1-λ) × max sim(d, selected)

    Args:
        docs: 候选文档列表
        embeddings: 每个文档对应的 embedding 向量
        scores: 每个文档的相关性分数（如 cosine similarity / rerank score），
                若为 None 则用 embedding 间的余弦相似度
        top_n: 返回文档数
        lambda_mult: 相关性权重，0~1，越大越重视相关性，越小越重视多样性

    Returns:
        MMR 选择后的文档列表
    """
    if len(docs) <= top_n:
        return docs

    emb_matrix = np.array(embeddings, dtype=np.float32)
    # 归一化
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    emb_matrix = emb_matrix / norms

    # 文档间相似度矩阵
    sim_matrix = emb_matrix @ emb_matrix.T

    # 相关性分数
    if scores is not None:
        relevance = np.array(scores, dtype=np.float32)
    else:
        # 用 embedding 的范数作为 proxy（已归一化，全部为 1，退化为等权）
        # 改用每个 doc 与其他 doc 的平均相似度作为 relevance
        relevance = np.array(scores if scores else [1.0] * len(docs), dtype=np.float32)

    selected: list[int] = []
    remaining = list(range(len(docs)))

    for _ in range(top_n):
        if not remaining:
            break
        if not selected:
            # 第一轮：选相关性最高的
            best = remaining[int(np.argmax(relevance[remaining]))]
        else:
            # MMR: λ × relevance - (1-λ) × max_sim_to_selected
            best_score = -np.inf
            best = remaining[0]
            for idx in remaining:
                max_sim = max(sim_matrix[idx, s] for s in selected)
                mmr_score = lambda_mult * relevance[idx] - (1 - lambda_mult) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best = idx
        selected.append(best)
        remaining.remove(best)

    return [docs[i] for i in selected]


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

    结合两种检索策略：
    1. 向量检索（Dense）：使用 embedding 模型将文本转为向量，进行语义相似度搜索
    2. BM25 检索（Sparse）：基于词频的经典全文检索算法

    检索结果通过 RRF（倒数排名融合）算法合并，
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
    # MMR 多样性权重，0~1，越大越重视相关性，越小越重视多样性
    mmr_lambda: float = 0.6

    def __init__(self, **kwargs):
        """初始化混合检索器"""
        super().__init__(**kwargs)

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

    def _expand_to_parents(self, leaves: list[Document], top_n: int, query: str = "", user_id: int | None = None) -> list[Document]:
        parent_ids_ordered: list[str] = []
        seen: set[str] = set()
        for leaf in leaves:
            pid = leaf.metadata.get("parent_id")
            if pid and pid not in seen:
                parent_ids_ordered.append(pid)
                seen.add(pid)

        # Track retrieval stats for on-demand KG extraction
        if user_id is not None:
            import threading
            threading.Thread(
                target=self._update_retrieval_stats,
                args=(parent_ids_ordered, user_id),
                daemon=True,
            ).start()

        parents = parent_store.get_by_ids(parent_ids_ordered)
        parent_map = {p.id: p for p in parents}
        leaf_scores = _max_leaf_score_per_parent(leaves)

        ordered_docs = []
        for pid in parent_ids_ordered:
            if pid not in parent_map:
                continue
            p = parent_map[pid]
            content = p.content
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

    def _fast_retrieve(self, query: str, top_k: int, user_id: int | None = None) -> list[Document]:
        """Fast strategy: vector retrieval only, no BM25/Reranker."""
        _retrieval_progress.set([])
        _record_progress("vector", f"向量检索: 搜索与问题语义相似的叶子块...")
        docs = self.vector_retriever.search_with_k(query, k=top_k * 2, user_id=user_id)
        _record_progress("vector_result", f"向量检索: 找到 {len(docs)} 个候选块")

        # MMR 去重：利用 Milvus 返回的 embedding
        if len(docs) > top_k:
            embeddings = [d.metadata.pop("_embedding") for d in docs]
            scores = [d.metadata.get("score", 0.0) for d in docs]
            if embeddings[0] is not None:
                docs = mmr_select(docs, embeddings, scores, top_n=top_k, lambda_mult=self.mmr_lambda)
                _record_progress("mmr", f"MMR 去重: 从 {top_k * 2} 个候选中选出 {len(docs)} 个多样化结果")
            else:
                docs = docs[:top_k]
        else:
            for d in docs:
                d.metadata.pop("_embedding", None)

        result = self._expand_to_parents(docs, top_n=top_k, query=query, user_id=user_id)
        _record_progress("expand", f"展开到父块: {len(docs)} 个叶子块 → {len(result)} 个父块")
        return result

    def _precise_retrieve(self, query: str, top_k: int, user_id: int | None = None) -> list[Document]:
        """Precise strategy: vector + BM25 -> RRF fusion (parallel)."""
        _retrieval_progress.set([])
        fetch_k = self.fetch_k if self.fetch_k else 10

        _record_progress("start", f"开始检索 (策略: precise, 获取 {fetch_k} 候选)")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_vec = ex.submit(self.vector_retriever.search_with_k, query, fetch_k, user_id)
            _record_progress("vector", "向量检索: 并行执行语义搜索...")
            f_bm25 = ex.submit(self.bm25_retriever.invoke, query)
            _record_progress("bm25", "BM25 检索: 并行执行关键词搜索...")

            vec_docs = f_vec.result()
            bm25_raw = f_bm25.result()

        _record_progress("vector_result", f"向量检索完成: {len(vec_docs)} 个候选块")
        _record_progress("bm25_result", f"BM25 检索完成: {len(bm25_raw)} 个候选块")

        # 清理 _embedding（RRF 不需要，MMR 会重新编码）
        for d in vec_docs:
            d.metadata.pop("_embedding", None)

        if user_id is not None:
            bm25_raw = [d for d in bm25_raw if d.metadata.get("user_id") == user_id]
        bm25_docs = bm25_raw[:fetch_k]

        # Add graph retrieval as third path
        graph_docs = self._graph_retrieve(query, user_id or 0, top_k=fetch_k)
        doc_lists = [vec_docs, bm25_docs, graph_docs]

        total_candidates = sum(len(dl) for dl in doc_lists)
        _record_progress("rrf", f"RRF 融合: 合并 {len(doc_lists)} 路共 {total_candidates} 个候选块 → Top {top_k}")
        fused = rrf_fusion(doc_lists, k=self.rrf_k, top_n=top_k)
        _record_progress("rrf_done", f"RRF 融合完成: {len(fused)} 个结果")

        # MMR 去重：重新编码 fused docs 的 embedding
        if len(fused) > 1:
            texts = [d.page_content for d in fused]
            embeddings = embedding_service.embed(texts)
            scores = [d.metadata.get("score", 0.0) for d in fused]
            fused = mmr_select(fused, embeddings, scores, top_n=top_k, lambda_mult=self.mmr_lambda)
            _record_progress("mmr", f"MMR 去重: 选出 {len(fused)} 个多样化结果")

        result = self._expand_to_parents(fused, top_n=top_k, query=query, user_id=user_id)
        _record_progress("expand", f"展开到父块: {len(fused)} 个叶子块 → {len(result)} 个父块")
        return result

    def _deep_retrieve(self, query: str, top_k: int, user_id: int | None = None) -> list[Document]:
        """Deep strategy: vector + BM25 -> RRF -> Reranker (parallel)."""
        _retrieval_progress.set([])
        cache_key = retrieval_cache.build_cache_key(
            namespace="hybrid_deep",
            query=query,
            strategy="deep",
            top_k=top_k,
        )
        cached = retrieval_cache.get(cache_key, label=query)
        if cached:
            _record_progress("cache", "缓存命中，直接返回上次结果")
            return cached[:top_k]

        from backend.services.reranker import reranker

        fetch_k = self.fetch_k if self.fetch_k else 10
        _record_progress("start", f"开始深度检索 (获取 {fetch_k} 候选)")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_vec = ex.submit(self.vector_retriever.search_with_k, query, fetch_k, user_id)
            _record_progress("vector", "向量检索: 并行执行语义搜索...")
            f_bm25 = ex.submit(self.bm25_retriever.invoke, query)
            _record_progress("bm25", "BM25 检索: 并行执行关键词搜索...")

            vec_docs = f_vec.result()
            bm25_raw = f_bm25.result()

        _record_progress("vector_result", f"向量检索完成: {len(vec_docs)} 个候选块")
        _record_progress("bm25_result", f"BM25 检索完成: {len(bm25_raw)} 个候选块")

        # 清理 _embedding（RRF 不需要，MMR 会重新编码）
        for d in vec_docs:
            d.metadata.pop("_embedding", None)

        if user_id is not None:
            bm25_raw = [d for d in bm25_raw if d.metadata.get("user_id") == user_id]
        bm25_docs = bm25_raw[:fetch_k]

        # Add graph retrieval as third path
        graph_docs = self._graph_retrieve(query, user_id or 0, top_k=fetch_k)
        doc_lists = [vec_docs, bm25_docs, graph_docs]

        total_candidates = sum(len(dl) for dl in doc_lists)
        _record_progress("rrf", f"RRF 融合: 合并 {len(doc_lists)} 路共 {total_candidates} 个候选块 → Top 10")
        fused = rrf_fusion(doc_lists, k=self.rrf_k, top_n=10)
        _record_progress("rerank", f"CrossEncoder 重排序: 对 {len(fused)} 个候选块重新打分排序")
        reranked = reranker.rerank(query, fused, top_n=top_k)
        _record_progress("rerank_done", f"重排序完成: {len(reranked)} 个结果")

        # MMR 去重：重新编码 reranked docs 的 embedding
        if len(reranked) > 1:
            texts = [d.page_content for d in reranked]
            embeddings = embedding_service.embed(texts)
            scores = [d.metadata.get("score", 0.0) for d in reranked]
            reranked = mmr_select(reranked, embeddings, scores, top_n=top_k, lambda_mult=self.mmr_lambda)
            _record_progress("mmr", f"MMR 去重: 选出 {len(reranked)} 个多样化结果")

        result = self._expand_to_parents(reranked, top_n=top_k, query=query, user_id=user_id)
        _record_progress("expand", f"展开到父块: {len(reranked)} 个叶子块 → {len(result)} 个父块")
        retrieval_cache.set(cache_key, result, label=query)
        return result

    def _graph_retrieve(self, query: str, user_id: int, top_k: int = 10) -> list[Document]:
        """Graph retrieval: extract entities from query, traverse Neo4j graph."""
        from backend.services.graph_service import graph_service
        from backend.services.entity_extractor import entity_extractor

        entities = entity_extractor.extract_query_entities(query)
        if not entities:
            return []

        chunk_ids = graph_service.search_by_entities(entities, user_id=user_id, top_k=top_k)
        if not chunk_ids:
            return []

        parents = parent_store.get_by_ids(chunk_ids)
        return [
            Document(
                page_content=p.content,
                metadata={"doc_id": p.id, "filename": p.filename, "heading_path": p.heading_path, "score": 1.0},
            )
            for p in parents
        ]

    def _update_retrieval_stats(self, parent_ids: list[str], user_id: int) -> None:
        """Async-safe: increment hit_count for retrieved parent chunks."""
        from datetime import datetime, timezone, timedelta

        if not parent_ids:
            return
        try:
            window = timedelta(days=settings.kg_hit_window_days)
            cutoff = datetime.now(timezone.utc) - window
            uuids = []
            for pid in parent_ids:
                try:
                    import uuid as _uuid
                    uuids.append(_uuid.UUID(pid))
                except ValueError:
                    continue
            if not uuids:
                return

            with SessionFactory() as session:
                for uid in uuids:
                    row = session.query(RetrievalStatsORM).filter_by(chunk_id=uid).first()
                    if row:
                        if row.last_hit_at and row.last_hit_at < cutoff:
                            row.hit_count = 1
                        else:
                            row.hit_count = row.hit_count + 1
                        row.last_hit_at = datetime.now(timezone.utc)
                    else:
                        session.add(RetrievalStatsORM(
                            chunk_id=uid, hit_count=1,
                            last_hit_at=datetime.now(timezone.utc), extracted=False,
                            user_id=user_id,
                        ))
                session.commit()

                # Trigger extraction for chunks that hit threshold
                to_extract = (
                    session.query(RetrievalStatsORM)
                    .filter(
                        RetrievalStatsORM.chunk_id.in_(uuids),
                        RetrievalStatsORM.hit_count >= settings.kg_hit_threshold,
                        RetrievalStatsORM.extracted == False,
                    )
                    .all()
                )
                if to_extract:
                    import threading
                    extract_ids = [str(r.chunk_id) for r in to_extract]
                    extract_user_id = to_extract[0].user_id
                    threading.Thread(
                        target=self._trigger_extraction, args=(extract_ids, extract_user_id), daemon=True
                    ).start()
        except Exception as e:
            logger.warning(f"Failed to update retrieval stats: {e}")

    def _trigger_extraction(self, chunk_ids: list[str], user_id: int) -> None:
        """Background: extract entities from chunks and write to Neo4j."""
        from backend.services.graph_service import graph_service
        from backend.services.entity_extractor import entity_extractor

        parents = parent_store.get_by_ids(chunk_ids)
        for p in parents:
            try:
                result = entity_extractor.extract_from_chunk(p.content)
                if result["entities"]:
                    import json as _json
                    graph_service.build_graph_for_chunk(
                        chunk_id=p.id,
                        filename=p.filename,
                        heading_path=_json.dumps(p.heading_path, ensure_ascii=False),
                        entities=result["entities"],
                        relations=result["relations"],
                        user_id=user_id,
                    )
                    logger.info(f"Extracted {len(result['entities'])} entities from chunk {p.id}")

                import uuid as _uuid
                with SessionFactory() as session:
                    row = session.query(RetrievalStatsORM).filter_by(chunk_id=_uuid.UUID(p.id)).first()
                    if row:
                        row.extracted = True
                        session.commit()
            except Exception as e:
                logger.warning(f"Extraction failed for chunk {p.id}: {e}")

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        orig_k = self.vector_retriever.search_kwargs.get("k", 4)
        return self._deep_retrieve(query, top_k=orig_k)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        orig_k = self.vector_retriever.search_kwargs.get("k", 4)
        cache_key = retrieval_cache.build_cache_key(
            namespace="hybrid_deep",
            query=query,
            strategy="deep",
            top_k=orig_k,
        )
        cached = retrieval_cache.get(cache_key, label=query)
        if cached:
            return cached[:orig_k]

        from backend.services.reranker import reranker

        fetch_k = self.fetch_k if self.fetch_k else 10

        async def _vec():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.vector_retriever.search_with_k, query, fetch_k)

        async def _bm25():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.bm25_retriever.invoke, query)

        vec_docs, bm25_raw = await asyncio.gather(_vec(), _bm25())

        bm25_docs = bm25_raw[:fetch_k]

        # 清理 _embedding
        for d in vec_docs:
            d.metadata.pop("_embedding", None)

        doc_lists = [vec_docs, bm25_docs]

        fused = rrf_fusion(doc_lists, k=self.rrf_k, top_n=10)
        reranked = reranker.rerank(query, fused, top_n=orig_k)

        # MMR 去重
        if len(reranked) > 1:
            loop = asyncio.get_event_loop()
            texts = [d.page_content for d in reranked]
            embeddings = await loop.run_in_executor(None, embedding_service.embed, texts)
            scores = [d.metadata.get("score", 0.0) for d in reranked]
            reranked = mmr_select(reranked, embeddings, scores, top_n=orig_k, lambda_mult=self.mmr_lambda)

        result = self._expand_to_parents(reranked, top_n=orig_k, query=query, user_id=None)
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

    def search_with_k(self, query: str, k: int, user_id: int | None = None) -> list[Document]:
        """Thread-safe search with explicit k, bypassing shared search_kwargs."""
        return self.vector_service.similarity_search(query, k=k, user_id=user_id)


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


# 导出 progress/thinking 工具供外部读取
def get_retrieval_progress() -> list[dict]:
    return _retrieval_progress.get()
