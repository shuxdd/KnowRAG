"""
分层分块器模块

将结构化元素列表分块为（父块，叶子块）对。

分块策略：
1. 父块边界：h1 或 h2 标题创建新的父块
2. h3-h6 标题保留在当前父块内
3. 超出大小限制的父块：先按 h3 分割，再按段落分割，最后按文本分割

叶子块构建：
- 表格、代码块和列表：preserve=True，保留不合并
- 普通文本：优先语义分块（句子 embedding 相似度骤降处切分），失败时回退到 RecursiveCharacterTextSplitter
- 过大的语义块：二次递归切分
- 过小的叶子块：合并到前一个叶子块
"""

import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import get_settings
from backend.services.parsing.base import StructuredElement
from backend.models.chunk_types import ParentChunk, LeafChunk

logger = logging.getLogger(__name__)

settings = get_settings()


class HierarchicalChunker:
    """
    分层分块器

    将结构化元素转换为父块-叶子块层次结构。
    父块代表文档的逻辑章节，叶子块用于向量检索。
    """

    def __init__(self):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.leaf_chunk_size,
            chunk_overlap=settings.leaf_chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        self._semantic_model = None  # lazy init
        self._llm = None  # lazy init

    @property
    def semantic_model(self):
        if self._semantic_model is None:
            from sentence_transformers import SentenceTransformer

            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            self._semantic_model = SentenceTransformer(
                settings.embedding_model,
                device=settings.embedding_device,
            )
        return self._semantic_model

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=settings.mimo_model,
                api_key=settings.mimo_api_key,
                base_url=settings.mimo_base_url,
                temperature=0.0,
                request_timeout=10,
            )
        return self._llm

    def chunk(
        self, elements: list[StructuredElement], filename: str
    ) -> Tuple[list[ParentChunk], list[LeafChunk]]:
        parents: list[ParentChunk] = []
        leaves: list[LeafChunk] = []
        heading_stack: list[tuple[int, str]] = []
        current_parent_elements: list[StructuredElement] = []
        current_parent_path: list[str] = []

        def flush_parent():
            nonlocal current_parent_elements, current_parent_path
            if not current_parent_elements:
                return
            # Skip parent chunks that contain only headings (no body content)
            if all(el.element_type == "heading" for el in current_parent_elements):
                current_parent_elements = []
                current_parent_path = []
                return
            parent = self._build_parent(
                current_parent_elements, current_parent_path[:], filename
            )
            if len(parent.content) > settings.parent_max_chars:
                sub_parents, sub_leaves = self._split_oversized(
                    parent, current_parent_elements, filename
                )
                parents.extend(sub_parents)
                leaves.extend(sub_leaves)
            else:
                leaf_list = self._build_leaves(parent, current_parent_elements)
                parents.append(parent)
                leaves.extend(leaf_list)

        for el in elements:
            if el.element_type == "heading":
                while heading_stack and heading_stack[-1][0] >= el.heading_level:
                    heading_stack.pop()
                heading_stack.append((el.heading_level, el.content))

                if el.heading_level <= 2:
                    flush_parent()
                    current_parent_elements = []
                    current_parent_path = [h[1] for h in heading_stack]
                    current_parent_elements.append(el)
                else:
                    current_parent_elements.append(el)
            else:
                if not current_parent_path:
                    current_parent_path = [filename]
                current_parent_elements.append(el)

        flush_parent()
        return parents, leaves

    def _build_parent(
        self,
        elements: list[StructuredElement],
        heading_path: list[str],
        filename: str,
    ) -> ParentChunk:
        content = "\n\n".join(el.content for el in elements)
        pages = [el.page for el in elements if el.page is not None]
        created_at = None
        for el in elements:
            ca = el.metadata.get("created_at")
            if ca:
                created_at = ca
                break
        return ParentChunk(
            id=str(uuid.uuid4()),
            content=content,
            heading_path=heading_path,
            filename=filename,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            created_at=created_at,
        )

    def _generate_description(
        self,
        element: StructuredElement,
        heading_path: list[str],
        prev_text: str,
        next_text: str,
    ) -> str:
        """Generate a one-line description for a preserved element via LLM."""
        type_label = {"code": "代码块", "table": "表格", "list": "列表"}.get(
            element.element_type, "内容块"
        )
        path_str = " > ".join(heading_path) if heading_path else "(无章节路径)"
        ctx_parts = []
        if prev_text:
            ctx_parts.append(f"前文：{prev_text[:200]}")
        if next_text:
            ctx_parts.append(f"后文：{next_text[:200]}")
        context_str = "\n".join(ctx_parts) if ctx_parts else "(无相邻上下文)"

        content_preview = element.content[:500]

        prompt = [
            SystemMessage(content="你是一个文档分析助手。根据上下文为内容块生成一句简短的中文描述，不超过50字，说明其内容和作用。只输出描述，不要其他内容。"),
            HumanMessage(content=f"章节路径：{path_str}\n{context_str}\n\n{type_label}内容：\n{content_preview}"),
        ]
        try:
            resp = self.llm.invoke(prompt)
            return resp.content.strip()
        except Exception as e:
            logger.warning(f"LLM description generation failed: {e}")
            return ""

    def _build_leaves(
        self,
        parent: ParentChunk,
        elements: list[StructuredElement],
    ) -> list[LeafChunk]:
        leaves: list[LeafChunk] = []
        text_parts: list[tuple[int, str]] = []
        preserve_items: list[tuple[int, StructuredElement]] = []

        for idx, el in enumerate(elements):
            if el.element_type in ("table", "code", "list"):
                preserve_items.append((idx, el))
            else:
                text_parts.append((idx, el.content))

        preserve_items.sort(key=lambda x: x[0])

        if text_parts:
            all_text = "\n\n".join(t[1] for t in text_parts)
            texts = self._split_text_semantic(all_text)
            if texts is None:
                split_docs = self._splitter.create_documents([all_text])
                texts = [doc.page_content for doc in split_docs]
            for ci, text in enumerate(texts):
                leaves.append(
                    LeafChunk(
                        id=str(uuid.uuid4()),
                        content=text,
                        heading_path=parent.heading_path,
                        parent_id=parent.id,
                        filename=parent.filename,
                        chunk_index=ci,
                        preserve=False,
                    )
                )

        last_used_ci = len(leaves)
        preserve_indices = {idx for idx, _ in preserve_items}

        def _gen_one(idx: int, el: StructuredElement) -> tuple[int, str]:
            prev_text, next_text = "", ""
            for j in range(idx - 1, -1, -1):
                if j not in preserve_indices:
                    prev_text = elements[j].content
                    break
            for j in range(idx + 1, len(elements)):
                if j not in preserve_indices:
                    next_text = elements[j].content
                    break
            desc = self._generate_description(el, parent.heading_path, prev_text, next_text)
            return idx, desc

        descriptions: dict[int, str] = {}
        if preserve_items:
            max_workers = min(len(preserve_items), 4)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(_gen_one, idx, el): idx
                    for idx, el in preserve_items
                }
                for future in as_completed(futures):
                    try:
                        idx, desc = future.result()
                        descriptions[idx] = desc
                    except Exception as e:
                        logger.warning(f"Description generation task failed: {e}")

        for idx, el in preserve_items:
            desc = descriptions.get(idx, "")
            content = f"{desc}\n\n{el.content}" if desc else el.content
            leaves.append(
                LeafChunk(
                    id=str(uuid.uuid4()),
                    content=content,
                    heading_path=parent.heading_path,
                    parent_id=parent.id,
                    filename=parent.filename,
                    chunk_index=last_used_ci,
                    preserve=True,
                )
            )
            last_used_ci += 1

        self._merge_undersized_leaves(leaves)
        leaves.sort(key=lambda l: l.chunk_index)
        return leaves

    @staticmethod
    def _merge_undersized_leaves(leaves: list[LeafChunk], min_chars: int = 100) -> None:
        """Merge leaf chunks below `min_chars` into the previous leaf.

        Skips preserved chunks (tables/code) — they stay atomic.
        """
        i = 0
        while i < len(leaves):
            leaf = leaves[i]
            if leaf.preserve or len(leaf.content) >= min_chars:
                i += 1
                continue
            # Find a merge target: prefer previous, fall back to next
            if i > 0 and not leaves[i - 1].preserve:
                target = leaves[i - 1]
                target.content = target.content + "\n\n" + leaf.content
                leaves.pop(i)
                # don't advance i — next leaf slides into current position
            elif i + 1 < len(leaves) and not leaves[i + 1].preserve:
                target = leaves[i + 1]
                target.content = leaf.content + "\n\n" + target.content
                leaves.pop(i)
                # advance past the merged target
                i += 1
            else:
                i += 1

    def _split_oversized(
        self,
        parent: ParentChunk,
        elements: list[StructuredElement],
        filename: str,
    ) -> Tuple[list[ParentChunk], list[LeafChunk]]:
        h3_indices = [
            i
            for i, el in enumerate(elements)
            if el.element_type == "heading" and el.heading_level == 3
        ]

        if not h3_indices:
            return self._split_by_paragraphs(elements, parent.heading_path, filename)

        all_parents: list[ParentChunk] = []
        all_leaves: list[LeafChunk] = []
        start = 0

        for h3_idx in h3_indices + [len(elements)]:
            sub = elements[start:h3_idx]
            if not sub:
                start = h3_idx
                continue

            sub_path = parent.heading_path[:]
            h3_els = [el for el in sub if el.element_type == "heading"]
            if h3_els:
                sub_path.append(h3_els[0].content)

            sub_parent = self._build_parent(sub, sub_path, filename)
            if len(sub_parent.content) > settings.parent_max_chars:
                sub_params, sub_leaves = self._split_by_paragraphs(
                    sub, sub_path, filename
                )
                all_parents.extend(sub_params)
                all_leaves.extend(sub_leaves)
            else:
                sub_leaves = self._build_leaves(sub_parent, sub)
                all_parents.append(sub_parent)
                all_leaves.extend(sub_leaves)

            start = h3_idx

        return all_parents, all_leaves

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences, Chinese-aware."""
        parts = re.split(r"(?<=[。！？.!?\n])\s*", text)
        return [p.strip() for p in parts if p.strip()]

    def _split_text_semantic(self, text: str) -> list[str] | None:
        """Split plain text at semantic boundaries. Returns None if insufficient boundaries."""
        sents = self._split_sentences(text)
        if len(sents) < 4:
            return None

        embeds = self.semantic_model.encode(sents, convert_to_numpy=True)
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeds = embeds / norms

        similarities = [float(np.dot(embeds[i], embeds[i + 1])) for i in range(len(embeds) - 1)]

        threshold = float(np.mean(similarities) - 0.5 * np.std(similarities))
        boundaries = [i for i, s in enumerate(similarities) if s < threshold]

        if not boundaries:
            return None

        groups: list[str] = []
        start = 0
        for b in boundaries:
            if b >= start:
                groups.append("".join(sents[start : b + 1]))
                start = b + 1
        remaining = "".join(sents[start:])
        if remaining.strip():
            groups.append(remaining)

        # Split oversized chunks with recursive splitter
        result: list[str] = []
        for chunk in groups:
            if len(chunk) > settings.leaf_chunk_size * 2:
                sub_docs = self._splitter.create_documents([chunk])
                result.extend(doc.page_content for doc in sub_docs)
            else:
                result.append(chunk)

        # Merge undersized chunks (< 80 chars) into neighbors
        merged: list[str] = []
        for chunk in result:
            if len(chunk) < 80 and merged:
                merged[-1] = merged[-1] + "\n\n" + chunk
            else:
                merged.append(chunk)
        if len(merged) >= 2 and len(merged[0]) < 80:
            merged[1] = merged[0] + "\n\n" + merged[1]
            merged.pop(0)

        return merged if len(merged) >= 2 else None

    def _split_semantic(
        self,
        elements: list[StructuredElement],
        heading_path: list[str],
        filename: str,
    ) -> Tuple[list[ParentChunk], list[LeafChunk]] | None:
        """Semantic chunking by sentence embedding similarity drops.

        Returns (parents, leaves) on success, None when no good boundaries found.
        """
        full_text = "\n\n".join(el.content for el in elements)
        sents = self._split_sentences(full_text)
        if len(sents) < 4:
            return None

        embeds = self.semantic_model.encode(sents, convert_to_numpy=True)
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeds = embeds / norms

        similarities = [float(np.dot(embeds[i], embeds[i + 1])) for i in range(len(embeds) - 1)]

        threshold = float(np.mean(similarities) - 0.5 * np.std(similarities))
        boundaries = [i for i, s in enumerate(similarities) if s < threshold]

        if not boundaries:
            return None

        # Group sentences at boundaries
        groups: list[str] = []
        start = 0
        for b in boundaries:
            if b >= start:
                groups.append("".join(sents[start : b + 1]))
                start = b + 1
        remaining = "".join(sents[start:])
        if remaining.strip():
            groups.append(remaining)

        # Merge undersized groups (< 200 chars) into neighbors
        merged: list[str] = []
        i = 0
        while i < len(groups):
            chunk = groups[i]
            if len(chunk) >= 200 or not merged:
                merged.append(chunk)
                i += 1
            else:
                # Merge small chunk with previous
                merged[-1] = merged[-1] + "\n\n" + chunk
                i += 1

        if len(merged) < 2:
            return None

        all_parents = []
        all_leaves = []
        for idx, text in enumerate(merged):
            sub_path = heading_path[:]
            if idx > 0:
                sub_path = heading_path[:] + [f"§{idx + 1}"]
            pseudo = StructuredElement(
                content=text,
                element_type="paragraph",
            )
            sub_parent = self._build_parent([pseudo], sub_path, filename)
            sub_leaves = self._build_leaves(sub_parent, [pseudo])
            all_parents.append(sub_parent)
            all_leaves.extend(sub_leaves)
        return all_parents, all_leaves

    def _split_by_paragraphs(
        self,
        elements: list[StructuredElement],
        heading_path: list[str],
        filename: str,
    ) -> Tuple[list[ParentChunk], list[LeafChunk]]:
        total_len = sum(len(el.content) for el in elements)
        if total_len <= settings.parent_max_chars:
            p = self._build_parent(elements, heading_path, filename)
            return [p], self._build_leaves(p, elements)

        # Try semantic split first, fall back to even splits
        result = self._split_semantic(elements, heading_path, filename)
        if result is not None:
            return result

        target = max(1, total_len // settings.parent_max_chars)
        # Group all elements by approximate chunk count
        chunk_size = max(1, len(elements) // target)
        all_parents: list[ParentChunk] = []
        all_leaves: list[LeafChunk] = []
        for i in range(0, len(elements), chunk_size):
            sub = elements[i : i + chunk_size]
            sub_path = heading_path[:]
            if i > 0:
                sub_path = heading_path[:] + ["(continued)"]
            sub_parent = self._build_parent(sub, sub_path, filename)
            sub_leaves = self._build_leaves(sub_parent, sub)
            all_parents.append(sub_parent)
            all_leaves.extend(sub_leaves)
        return all_parents, all_leaves
