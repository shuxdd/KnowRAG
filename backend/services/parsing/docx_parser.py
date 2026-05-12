from docx import Document as DocxDocument
from backend.services.parsing.base import BaseParser, StructuredElement


class DocxParser(BaseParser):
    def parse(self, filepath: str) -> list[StructuredElement]:
        doc = DocxDocument(filepath)
        result: list[StructuredElement] = []
        for p in doc.paragraphs:
            style = p.style.name
            text = p.text.strip()
            if not text:
                continue
            if style.startswith("Heading "):
                level = int(style.split()[1])
                result.append(
                    StructuredElement(
                        content=text,
                        element_type="heading",
                        heading_level=min(level, 6),
                    )
                )
            else:
                result.append(
                    StructuredElement(content=text, element_type="paragraph")
                )
        for tbl in doc.tables:
            rows = []
            for row in tbl.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(" | ".join(cells))
            md_table = "\n".join(rows)
            result.append(StructuredElement(content=md_table, element_type="table"))
        return result
