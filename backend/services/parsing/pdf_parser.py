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
        if doc.page_count == 0:
            doc.close()
            return []

        pdf_meta = doc.metadata or {}
        created_raw = pdf_meta.get("creationDate") or pdf_meta.get("modDate")
        created_at = _parse_pdf_date(created_raw) if created_raw else None
        elem_meta = {"created_at": created_at} if created_at else {}

        page_width = doc[0].rect.width if doc.page_count > 0 else 595
        page_height = doc[0].rect.height if doc.page_count > 0 else 842
        margin = page_width * 0.02
        header_zone = page_height * 0.06    # top 6% — likely header / page number
        footer_zone = page_height * 0.92    # bottom 8 % — likely footer / annotation

        sizes: list[float] = []
        page_lines: list[list[tuple[fitz.Rect, str, float, bool]]] = []

        # --- First pass: collect font sizes & group spans into lines ---
        for page_num in range(doc.page_count):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            lines_on_page: list[tuple[fitz.Rect, str, float, bool]] = []
            for block in blocks:
                if block["type"] != 0:
                    continue
                for line_data in block.get("lines", []):
                    line_spans: list[tuple[fitz.Rect, str, float, bool]] = []
                    for span in line_data.get("spans", []):
                        sizes.append(span["size"])
                        text = span["text"].strip()
                        if not text:
                            continue
                        bbox = span["bbox"]
                        # Skip header / footer / annotation regions
                        if bbox[1] < header_zone or bbox[1] > footer_zone:
                            continue
                        line_spans.append(
                            (fitz.Rect(bbox), text, span["size"],
                             bool(span["flags"] & 2))
                        )
                    if line_spans:
                        merged_text = "".join(s[1] for s in line_spans)
                        merged_bbox = fitz.Rect(
                            min(s[0].x0 for s in line_spans),
                            min(s[0].y0 for s in line_spans),
                            max(s[0].x1 for s in line_spans),
                            max(s[0].y1 for s in line_spans),
                        )
                        max_size = max(s[2] for s in line_spans)
                        any_bold = any(s[3] for s in line_spans)
                        lines_on_page.append(
                            (merged_bbox, merged_text, max_size, any_bold)
                        )
            page_lines.append(lines_on_page)

        if not sizes:
            doc.close()
            return []

        base_size = Counter(round(s, 1) for s in sizes).most_common(1)[0][0]
        h1_threshold = base_size * settings.pdf_h1_ratio
        h2_low = base_size * settings.pdf_h2_ratio
        h3_low = base_size * settings.pdf_h3_ratio

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

        # --- Watermark / repeated-text detection ---
        # Text that appears on >= 80% of pages and is short is likely a
        # header, footer, watermark, or annotation artifact.
        text_page_count: dict[str, set[int]] = {}
        for page_num, lines in enumerate(page_lines):
            for _, text, _, _ in lines:
                text_page_count.setdefault(text, set()).add(page_num)
        watermark_texts = {
            text for text, pages in text_page_count.items()
            if len(pages) >= doc.page_count * 0.8 and len(text) < 60
        }

        result: list[StructuredElement] = []

        # --- Second pass: build structured elements ---
        for page_num, lines in enumerate(page_lines):
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

            for bbox, text, size, bold in lines:
                # Skip watermark / repeated annotation text
                if text in watermark_texts:
                    continue

                # Skip text inside table regions (already extracted)
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
                                content=text, element_type="paragraph",
                                page=page_num, metadata=elem_meta,
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
                            content=text, element_type="paragraph",
                            page=page_num, metadata=elem_meta,
                        )
                    )

        doc.close()
        return result
