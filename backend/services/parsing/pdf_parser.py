"""
PDF 解析器模块

解析策略：
1. MinerU（主路径）：优先使用 MinerU Flash 模式解析所有 PDF
2. PyMuPDF（回退路径）：MinerU 不可用或失败时，使用 PyMuPDF 提取文本块、
   基于字体大小识别标题级别（h1/h2/h3）、提取表格

PyMuPDF 回退路径支持：
- 文本提取和布局分析
- 标题检测（基于字体大小和加粗）
- 表格提取并转换为 Markdown 格式
"""

import logging
import re
from datetime import datetime, timezone

import fitz
from collections import Counter
from backend.config import get_settings
from backend.services.parsing.base import BaseParser, StructuredElement

settings = get_settings()
logger = logging.getLogger(__name__)


def _parse_pdf_date(raw: str) -> str | None:
    """
    解析 PDF 日期字符串

    支持格式：
    - D:YYYYMMDDHHMMSS+HH'MM'（PDF 内部格式）
    - ISO 格式

    Args:
        raw: 原始日期字符串

    Returns:
        ISO 格式日期字符串，解析失败返回 None
    """
    if not raw:
        return None
    match = re.search(r"(\d{4}\d{2}\d{2}\d{2}\d{2}\d{2})", raw)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            ).isoformat()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(raw.replace("'", "")).isoformat()
    except (ValueError, TypeError):
        return None


class PdfParser(BaseParser):
    """
    PDF 文档解析器

    使用 PyMuPDF (fitz) 库解析 PDF，提取文本、标题、表格等结构。
    支持自动检测扫描件并回退到 MinerU 解析。
    """

    def parse(self, filepath: str) -> list[StructuredElement]:
        """
        解析 PDF 文件

        优先使用 MinerU 解析，失败时回退到 PyMuPDF。

        Args:
            filepath: PDF 文件路径

        Returns:
            结构化元素列表
        """
        try:
            from backend.services.parsing.mineru_parser import MinerUParser
            return MinerUParser().parse(filepath)
        except ImportError:
            logger.warning("MinerU not installed, falling back to PyMuPDF")
        except Exception:
            logger.warning(
                "MinerU parsing failed for %s, falling back to PyMuPDF",
                filepath,
                exc_info=True,
            )
        return self._parse_pymupdf(filepath)

    def _parse_pymupdf(self, filepath: str) -> list[StructuredElement]:
        """PyMuPDF fallback parsing with font-based heading detection and table extraction."""
        doc = fitz.open(filepath)
        page_count = doc.page_count
        if page_count == 0:
            doc.close()
            return []

        pdf_meta = doc.metadata or {}
        created_raw = pdf_meta.get("creationDate") or pdf_meta.get("modDate")
        created_at = _parse_pdf_date(created_raw) if created_raw else None
        doc.close()

        doc = fitz.open(filepath)
        sizes: list[float] = []
        pages_text: list[list[tuple[fitz.Rect, str, float, bool]]] = []
        elem_meta = {"created_at": created_at} if created_at else {}

        for page_num in range(doc.page_count):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            blocks_on_page: list[tuple[fitz.Rect, str, float, bool]] = []
            for block in blocks:
                if block["type"] != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        sizes.append(span["size"])
                        text = span["text"].strip()
                        if text:
                            blocks_on_page.append(
                                (
                                    fitz.Rect(span["bbox"]),
                                    text,
                                    span["size"],
                                    bool(span["flags"] & 2),
                                )
                            )
            pages_text.append(blocks_on_page)

        if not sizes:
            doc.close()
            return []

        base_size = Counter(round(s, 1) for s in sizes).most_common(1)[0][0]
        h1_threshold = base_size * settings.pdf_h1_ratio
        h2_low = base_size * settings.pdf_h2_ratio
        h3_low = base_size * settings.pdf_h3_ratio

        page_width = doc[0].rect.width if doc.page_count > 0 else 595
        margin = page_width * 0.02

        def is_heading(bbox: fitz.Rect, size: float, bold: bool) -> int | None:
            if bbox.x0 < margin or bbox.x1 > page_width - margin:
                return None
            if size > h1_threshold:
                return 1
            if size > h2_low:
                return 2
            if size > h3_low and bold:
                return 3
            return None

        result: list[StructuredElement] = []

        for page_num, spans in enumerate(pages_text):
            page = doc[page_num]
            tables = page.find_tables()
            table_regions = []
            for tbl in tables:
                rows = []
                for row in tbl.extract():
                    cells = [str(c).strip() for c in row]
                    rows.append(" | ".join(cells))
                md_table = "\n".join(rows)
                table_regions.append(tbl.bbox)
                result.append(
                    StructuredElement(
                        content=md_table, element_type="table", page=page_num,
                        metadata=elem_meta,
                    )
                )

            for bbox, text, size, bold in spans:
                if table_regions:
                    in_table = False
                    for tbl_bbox in table_regions:
                        if (
                            bbox.x0 >= tbl_bbox[0]
                            and bbox.x1 <= tbl_bbox[2]
                            and bbox.y0 >= tbl_bbox[1]
                            and bbox.y1 <= tbl_bbox[3]
                        ):
                            in_table = True
                            break
                    if in_table:
                        continue

                if text.replace(".", "").replace("-", "").strip().isdigit():
                    continue

                h_level = is_heading(bbox, size, bold)
                if h_level is not None:
                    if len(text) > 100:
                        result.append(
                            StructuredElement(
                                content=text, element_type="paragraph", page=page_num,
                                metadata=elem_meta,
                            )
                        )
                    else:
                        result.append(
                            StructuredElement(
                                content=text,
                                element_type="heading",
                                heading_level=h_level,
                                page=page_num,
                                metadata=elem_meta,
                            )
                        )
                else:
                    result.append(
                        StructuredElement(
                            content=text, element_type="paragraph", page=page_num,
                            metadata=elem_meta,
                        )
                    )

        doc.close()
        return result
