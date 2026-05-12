import uuid
from typing import Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import get_settings
from backend.services.parsing.base import StructuredElement
from backend.models.chunk_types import ParentChunk, LeafChunk

settings = get_settings()


class HierarchicalChunker:
    """Split StructuredElement lists into (parent, leaf) pairs.

    Parent boundaries are created at heading level 1 or 2.
    Level 3+ headings stay within the current parent.
    Oversized parents are recursively split by h3 boundaries,
    then by paragraph groups, then by text.
    """

    def __init__(self):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.leaf_chunk_size,
            chunk_overlap=settings.leaf_chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

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
        return ParentChunk(
            id=str(uuid.uuid4()),
            content=content,
            heading_path=heading_path,
            filename=filename,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
        )

    def _build_leaves(
        self,
        parent: ParentChunk,
        elements: list[StructuredElement],
    ) -> list[LeafChunk]:
        leaves: list[LeafChunk] = []
        text_parts: list[tuple[int, str]] = []
        preserve_items: list[tuple[int, StructuredElement]] = []

        for idx, el in enumerate(elements):
            if el.element_type in ("table", "code"):
                preserve_items.append((idx, el))
            else:
                text_parts.append((idx, el.content))

        preserve_items.sort(key=lambda x: x[0])

        if text_parts:
            all_text = "\n\n".join(t[1] for t in text_parts)
            split_docs = self._splitter.create_documents([all_text])
            for ci, doc in enumerate(split_docs):
                leaves.append(
                    LeafChunk(
                        id=str(uuid.uuid4()),
                        content=doc.page_content,
                        heading_path=parent.heading_path,
                        parent_id=parent.id,
                        filename=parent.filename,
                        chunk_index=ci,
                        preserve=False,
                    )
                )

        last_used_ci = len(leaves)
        for idx, el in preserve_items:
            leaves.append(
                LeafChunk(
                    id=str(uuid.uuid4()),
                    content=el.content,
                    heading_path=parent.heading_path,
                    parent_id=parent.id,
                    filename=parent.filename,
                    chunk_index=last_used_ci,
                    preserve=True,
                )
            )
            last_used_ci += 1

        leaves.sort(key=lambda l: l.chunk_index)
        return leaves

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

    def _split_by_paragraphs(
        self,
        elements: list[StructuredElement],
        heading_path: list[str],
        filename: str,
    ) -> Tuple[list[ParentChunk], list[LeafChunk]]:
        # Build the full text content to check its size
        full_content = "\n\n".join(el.content for el in elements)

        # If it fits, short-circuit
        if len(full_content) <= settings.parent_max_chars:
            p = self._build_parent(elements, heading_path, filename)
            return [p], self._build_leaves(p, elements)

        paras = [el for el in elements if el.element_type == "paragraph"]

        # Single oversized paragraph: split at the text level
        if len(paras) <= 1:
            split_docs = self._splitter.create_documents([full_content])
            all_parents: list[ParentChunk] = []
            all_leaves: list[LeafChunk] = []

            for ci, doc in enumerate(split_docs):
                sub_path = heading_path[:]
                if ci > 0:
                    sub_path = heading_path[:] + ["(continued)"]

                sub_el = StructuredElement(
                    content=doc.page_content, element_type="paragraph"
                )
                sub_parent = self._build_parent([sub_el], sub_path, filename)
                sub_leaves = self._build_leaves(sub_parent, [sub_el])
                all_parents.append(sub_parent)
                all_leaves.extend(sub_leaves)

            return all_parents, all_leaves

        # Multiple paragraphs: group them to stay under parent_max_chars
        total_len = sum(len(el.content) for el in elements)
        target = max(
            1, len(paras) // max(1, total_len // settings.parent_max_chars)
        )
        all_parents: list[ParentChunk] = []
        all_leaves: list[LeafChunk] = []

        for i in range(0, len(paras), target):
            sub = paras[i : i + target]
            sub_path = heading_path[:]
            if i > 0:
                sub_path = heading_path[:] + ["(continued)"]
            sub_parent = self._build_parent(sub, sub_path, filename)
            sub_leaves = self._build_leaves(sub_parent, sub)
            all_parents.append(sub_parent)
            all_leaves.extend(sub_leaves)

        return all_parents, all_leaves
