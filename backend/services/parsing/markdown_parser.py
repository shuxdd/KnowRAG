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

        # Split the text into top-level blocks separated by blank lines.
        # This preserves heading/paragraph/ code-block structure.
        raw_blocks = re.split(r"\n\n+", text)

        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue

            # Heading line ?  Must be at the start of the block.
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", block, re.MULTILINE)
            if heading_match and block == heading_match.group(0):
                # The entire block is a single heading line.
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                result.append(
                    StructuredElement(
                        content=heading_text,
                        element_type="heading",
                        heading_level=level,
                    )
                )
            elif block.startswith("```"):
                # Fenced code block
                code_content = block.strip("`").strip()
                result.append(
                    StructuredElement(content=code_content, element_type="code")
                )
            else:
                # Plain text paragraph – check for inline code fences
                # and split them out.
                parts = re.split(r"(```[\s\S]*?```)", block)
                for part in parts:
                    if not part.strip():
                        continue
                    if part.startswith("```"):
                        code_text = part.strip("`").strip()
                        result.append(
                            StructuredElement(
                                content=code_text, element_type="code"
                            )
                        )
                    else:
                        result.append(
                            StructuredElement(
                                content=part.strip(), element_type="paragraph"
                            )
                        )

        return result
