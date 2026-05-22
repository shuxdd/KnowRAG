"""
MinerU PDF 解析器模块

MinerU 是一个强大的 PDF 解析库，特别适合处理扫描件和复杂版面的 PDF。
本模块使用 MinerU Flash 模式（免费，无需 API Token）解析 PDF，
然后将结果转换为结构化元素列表。

限制：
- 文件大小 ≤ 10MB
- 页数 ≤ 20 页

使用场景：
- 当 PDF 字符密度过低（疑似扫描件）时自动调用
"""

import logging
import os

from backend.services.parsing.base import BaseParser, StructuredElement

logger = logging.getLogger(__name__)


class MinerUParser(BaseParser):
    """
    MinerU PDF 解析器

    使用 MinerU Flash 模式解析扫描件和复杂版面 PDF，
    然后复用 MarkdownParser 转换为结构化元素。
    """

    def parse(self, filepath: str) -> list[StructuredElement]:
        """
        使用 MinerU 解析 PDF 文件

        Args:
            filepath: PDF 文件路径

        Returns:
            结构化元素列表

        Raises:
            FileNotFoundError: 文件不存在
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)

        from langchain_mineru import MinerULoader

        file_size = os.path.getsize(filepath)
        if file_size > 10 * 1024 * 1024:
            logger.warning(
                "File %s exceeds 10 MB, MinerU Flash mode may fail", filepath
            )

        loader = MinerULoader(source=filepath, mode="flash")
        docs = loader.load()

        if not docs:
            logger.warning("MinerU returned no content for %s", filepath)
            return []

        # Try to get PDF creation date via PyMuPDF
        created_at = None
        try:
            import fitz
            pdf = fitz.open(filepath)
            pdf_meta = pdf.metadata or {}
            raw = pdf_meta.get("creationDate") or pdf_meta.get("modDate")
            if raw:
                from backend.services.parsing.pdf_parser import _parse_pdf_date
                created_at = _parse_pdf_date(raw)
            pdf.close()
        except Exception:
            pass
        if not created_at:
            from backend.services.parsing.markdown_parser import _file_mtime
            created_at = _file_mtime(filepath)

        elem_meta = {"created_at": created_at} if created_at else {}
        md_content = "\n\n".join(doc.page_content for doc in docs)

        from backend.services.parsing.markdown_parser import MarkdownParser

        elements = MarkdownParser().parse_string(md_content)
        for el in elements:
            el.metadata = elem_meta
        return elements
