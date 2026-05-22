"""
文档服务模块

提供文档处理的完整流程：
1. 文件保存：将上传文件保存到本地存储
2. 文档解析：根据文件类型选择解析器（PDF/DOCX/TXT/Markdown）
3. 分块处理：使用 HierarchicalChunker 进行分层分块
4. 数据存储：将父块存入 PostgreSQL，叶子块存入 ChromaDB
5. 文档删除：删除时同时清理两个数据库

支持的文档格式：
- PDF: 使用 PdfParser（MinerU 优先，PyMuPDF 回退）
- DOCX: 使用 DocxParser
- TXT: 使用 TxtParser
- Markdown: 使用 MarkdownParser
"""

import os
import uuid
import logging
from typing import Tuple
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.config import get_settings
from backend.services.parsing.markdown_parser import MarkdownParser
from backend.services.parsing.docx_parser import DocxParser
from backend.services.parsing.pdf_parser import PdfParser
from backend.services.parsing.txt_parser import TxtParser
from backend.services.chunking.hierarchical_chunker import HierarchicalChunker
from backend.services.vector_service import vector_service
from backend.services.parent_store import parent_store
from backend.services.hybrid_retriever import retrieval_cache, hybrid_retriever
from backend.models.chunk_types import ParentChunk, LeafChunk
from backend.services.parsing.markdown_parser import _file_mtime

logger = logging.getLogger(__name__)
settings = get_settings()


def _extract_file_created(filepath: str, ext: str) -> str | None:
    """Extract document creation time: PDF→metadata, DOCX→core.xml, other→mtime."""
    if ext == ".pdf":
        try:
            import fitz
            pdf = fitz.open(filepath)
            meta = pdf.metadata or {}
            raw = meta.get("creationDate") or meta.get("modDate")
            pdf.close()
            if raw:
                from backend.services.parsing.pdf_parser import _parse_pdf_date
                return _parse_pdf_date(raw)
        except Exception:
            logger.debug(f"PDF date extraction failed for {filepath}", exc_info=True)
    elif ext == ".docx":
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(filepath)
            c = doc.core_properties.created
            if c:
                from datetime import timezone
                return c.astimezone(timezone.utc).isoformat()
        except Exception:
            logger.debug(f"DOCX date extraction failed for {filepath}", exc_info=True)
    return _file_mtime(filepath)

_PARSERS = {
    ".md": MarkdownParser(),
    ".markdown": MarkdownParser(),
    ".pdf": PdfParser(),
    ".docx": DocxParser(),
    ".txt": TxtParser(),
}

def _normalize_ext(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return ".md" if ext in (".md", ".markdown") else ext

def _get_parser(ext: str):
    parser = _PARSERS.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return parser


class DocumentService:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        self.chunker = HierarchicalChunker()

    def process_file(self, filepath: str, filename: str) -> str:
        # Dedup: remove existing chunks before re-processing the same filename
        parent_store.delete_by_filename(filename)
        vector_service.delete_by_filename(filename)

        ext = _normalize_ext(filename)
        parser = _get_parser(ext)

        try:
            elements = parser.parse(filepath)
            parents, leaves = self.chunker.chunk(elements, filename)
        except Exception as e:
            logger.warning(f"{filename}: structured parsing failed ({e}), using legacy fallback")
            parents, leaves = self._legacy_fallback(filepath, filename)

        # 1) PG first
        parent_store.add(parents)

        # 2) ChromaDB second; rollback PG on failure
        try:
            vector_service.add_leaves(leaves)
        except Exception as e:
            logger.error(f"{filename}: chroma insert failed, rolling back PG parents")
            parent_store.delete_by_ids([p.id for p in parents])
            raise

        retrieval_cache.invalidate_all()
        hybrid_retriever.rebuild_bm25()
        return leaves[0].id if leaves else ""

    def _legacy_fallback(self, filepath: str, filename: str) -> Tuple[list[ParentChunk], list[LeafChunk]]:
        ext = _normalize_ext(filename)
        if ext == ".pdf":
            loader = PyPDFLoader(filepath)
        elif ext == ".docx":
            loader = Docx2txtLoader(filepath)
        elif ext == ".txt":
            loader = TextLoader(filepath, encoding="utf-8")
        else:
            raise ValueError(f"Legacy fallback: unsupported file type: {ext}")

        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.leaf_chunk_size,
            chunk_overlap=settings.leaf_chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        chunks = splitter.split_documents(raw_docs)

        created_at = _extract_file_created(filepath, ext)
        all_text = "\n\n".join(doc.page_content for doc in raw_docs)
        parent = ParentChunk(
            id=str(uuid.uuid4()),
            content=all_text,
            heading_path=[filename],
            filename=filename,
            created_at=created_at,
        )

        leaves = [
            LeafChunk(
                id=str(uuid.uuid4()),
                content=chunk.page_content,
                heading_path=[filename],
                parent_id=parent.id,
                filename=filename,
                chunk_index=i,
                preserve=False,
            )
            for i, chunk in enumerate(chunks)
        ]

        return [parent], leaves

    def save_upload(self, file_content: bytes, filename: str) -> str:
        os.makedirs(settings.upload_dir, exist_ok=True)
        unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
        filepath = os.path.join(settings.upload_dir, unique_name)
        with open(filepath, "wb") as f:
            f.write(file_content)
        return filepath

    def get_file_size(self, filepath: str) -> int:
        return os.path.getsize(filepath)

    def delete_file(self, filename: str) -> dict:
        parent_count = parent_store.delete_by_filename(filename)
        leaf_count = vector_service.delete_by_filename(filename)
        for f in os.listdir(settings.upload_dir):
            if f.endswith("_" + filename):
                os.remove(os.path.join(settings.upload_dir, f))
                break
        retrieval_cache.invalidate_all()
        hybrid_retriever.rebuild_bm25()
        return {"parents": parent_count, "leaves": leaf_count}


document_service = DocumentService()
