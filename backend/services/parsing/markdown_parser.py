"""
Markdown 解析器模块

解析 Markdown 文本，提取：
- 标题（h1-h6）
- 代码块（fenced code blocks）
- 段落（由空行分隔的连续文本行）

支持两种解析方式：
- parse(): 从文件路径读取并解析
- parse_string(): 直接解析文本字符串
"""

import os
import re
from datetime import datetime, timezone

from backend.services.parsing.base import BaseParser, StructuredElement


def _file_mtime(filepath: str) -> str | None:
    """
    获取文件的修改时间

    Args:
        filepath: 文件路径

    Returns:
        ISO 格式的时间字符串，失败返回 None
    """
    try:
        mtime = os.path.getmtime(filepath)
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except OSError:
        return None


class MarkdownParser(BaseParser):
    """
    Markdown 文档解析器

    解析 Markdown 文本，提取标题、代码块和段落。
    嵌套标题产生扁平序列，每个标题的 heading_level 反映其层级。
    """

    def parse_string(self, text: str) -> list[StructuredElement]:
        """
        解析 Markdown 文本字符串

        Args:
            text: Markdown 文本内容

        Returns:
            结构化元素列表
        """
        return self._parse(text)

    def parse(self, filepath: str) -> list[StructuredElement]:
        """
        解析 Markdown 文件

        Args:
            filepath: Markdown 文件路径

        Returns:
            结构化元素列表
        """
        created_at = _file_mtime(filepath)
        elem_meta = {"created_at": created_at} if created_at else {}
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        elements = self._parse(text)
        for el in elements:
            el.metadata = elem_meta
        return elements

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

        def _is_table_row(line: str) -> bool:
            return "|" in line

        def _is_separator_row(line: str) -> bool:
            return bool(re.match(r"^\|?[\s:]*[-]{3,}[\s:|:-]*\|", line))

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

            # Table detection: consecutive pipe-containing lines with separator in between
            if _is_table_row(line):
                flush_buffer()
                table_lines = []
                has_separator = False
                j = i
                while j < len(lines) and _is_table_row(lines[j]):
                    if _is_separator_row(lines[j]):
                        has_separator = True
                    table_lines.append(lines[j])
                    j += 1
                if has_separator and len(table_lines) >= 2:
                    result.append(
                        StructuredElement(
                            content="\n".join(table_lines),
                            element_type="table",
                        )
                    )
                    i = j
                    continue
                # Not a valid table — push back to buffer
                buffer.extend(table_lines)
                i = j
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
