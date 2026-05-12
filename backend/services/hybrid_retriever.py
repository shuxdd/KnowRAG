import hashlib
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI

from backend.config import get_settings

settings = get_settings()
from backend.services.parent_store import parent_store

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
    bm25_retriever: BM25Retriever
    # RRF 融合参数 k，控制排名靠前文档的优势程度
    rrf_k: int = 60
    # 从每个检索器获取的候选文档数量，0 表示使用 top_n 的两倍
    fetch_k: int = 0

    def __init__(self, **kwargs):
        """
        初始化混合检索器

        初始化 HyDE 专用的 LLM（用于生成假设答案）
        """
        super().__init__(**kwargs)
        # 初始化用于 HyDE 的 LLM（使用 DeepSeek 模型）
        self._hyde_llm = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.3,       # 中等随机性，生成的事实性假设答案
            request_timeout=10,    # 请求超时时间（秒）
        )

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
            # 如果 HyDE 检索失败，返回空列表
            return []

    def _expand_to_parents(self, leaves: list[Document], top_n: int) -> list[Document]:
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
            ordered_docs.append(Document(
                page_content=p.content,
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

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        from backend.services.reranker import reranker

        orig_k = self.vector_retriever.search_kwargs.get("k", 4)
        fetch_k = self.fetch_k or orig_k * 2

        self.vector_retriever.search_kwargs["k"] = fetch_k
        vec_docs = self.vector_retriever.invoke(query)
        self.vector_retriever.search_kwargs["k"] = orig_k

        bm25_docs = self.bm25_retriever.invoke(query)[:fetch_k]
        hyde_docs = self._hyde_search(query, top_k=fetch_k)

        fused = rrf_fusion([vec_docs, bm25_docs, hyde_docs], k=self.rrf_k, top_n=orig_k * 2)
        reranked = reranker.rerank(query, fused, top_n=orig_k)
        return self._expand_to_parents(reranked, top_n=orig_k)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        from backend.services.reranker import reranker

        orig_k = self.vector_retriever.search_kwargs.get("k", 4)
        fetch_k = self.fetch_k or orig_k * 2

        self.vector_retriever.search_kwargs["k"] = fetch_k
        vec_docs = await self.vector_retriever.ainvoke(query)
        self.vector_retriever.search_kwargs["k"] = orig_k

        bm25_docs = self.bm25_retriever.invoke(query)[:fetch_k]
        hyde_docs = self._hyde_search(query, top_k=fetch_k)

        fused = rrf_fusion([vec_docs, bm25_docs, hyde_docs], k=self.rrf_k, top_n=orig_k * 2)
        reranked = reranker.rerank(query, fused, top_n=orig_k)
        return self._expand_to_parents(reranked, top_n=orig_k)
