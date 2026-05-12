import fitz
from collections import Counter
from backend.config import get_settings
from backend.services.parsing.base import BaseParser, StructuredElement

settings = get_settings()


class PdfParser(BaseParser):
    def parse(self, filepath: str) -> list[StructuredElement]:
        doc = fitz.open(filepath)
        if doc.page_count == 0:
            return []

        sizes: list[float] = []
        pages_text: list[list[tuple[fitz.Rect, str, float, bool]]] = []

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
                        content=md_table, element_type="table", page=page_num
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
                                content=text, element_type="paragraph", page=page_num
                            )
                        )
                    else:
                        result.append(
                            StructuredElement(
                                content=text,
                                element_type="heading",
                                heading_level=h_level,
                                page=page_num,
                            )
                        )
                else:
                    result.append(
                        StructuredElement(
                            content=text, element_type="paragraph", page=page_num
                        )
                    )

        doc.close()
        return result
