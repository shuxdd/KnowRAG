import re
from backend.services.parsing.base import BaseParser, StructuredElement


class MarkdownParser(BaseParser):
    """Parse Markdown text into StructuredElement list.

    Parses headings (h1-h6), code blocks (fenced), and paragraphs.
    Nested headings produce a flat sequence with heading_level set appropriately.
    """

    def parse_string(self, text: str) -> list[StructuredElement]:
        return self._parse(text)

    def parse(self, filepath: str) -> list[StructuredElement]:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return self._parse(text)

    def _parse(self, text: str) -> list[StructuredElement]:
        result: list[StructuredElement] = []
        lines = text.split("\n")
        i = 0
        buffer: list[str] = []

        def flush_buffer():
            if buffer:
                para = "\n".join(buffer).strip()
                if para:
                    stripped = para.strip()
                    if stripped == "```":
                        buffer.clear()
                        return
                    result.append(
                        StructuredElement(content=para, element_type="paragraph")
                    )
                buffer.clear()

        while i < len(lines):
            line = lines[i]

            # Fenced code block
            if line.startswith("```"):
                flush_buffer()
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                # Include closing fence line for stripping later
                if i < len(lines):
                    code_lines.append(lines[i])
                code_text = "\n".join(code_lines).strip("\n ")
                result.append(
                    StructuredElement(content=code_text, element_type="code")
                )
                i += 1
                continue

            # Heading line
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading_match:
                flush_buffer()
                level = len(heading_match.group(1))
                text_content = heading_match.group(2).strip()
                result.append(
                    StructuredElement(
                        content=text_content,
                        element_type="heading",
                        heading_level=level,
                    )
                )
                i += 1
                continue

            # Empty line marks a paragraph boundary
            if line.strip() == "":
                flush_buffer()
                i += 1
                continue

            # Regular line — accumulate in paragraph buffer
            buffer.append(line)
            i += 1

        flush_buffer()
        return result
