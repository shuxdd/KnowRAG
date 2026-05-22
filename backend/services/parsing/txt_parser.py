"""
文本文件解析器模块

解析纯文本文件（.txt），将文本按段落（双换行）分割，
每个段落作为一个 paragraph 类型的结构化元素。

支持：
- parse(): 从文件路径读取并解析
- parse_string(): 直接解析文本字符串
"""

from backend.services.parsing.base import BaseParser, StructuredElement
from backend.services.parsing.markdown_parser import _file_mtime


class TxtParser(BaseParser):
    """
    纯文本文件解析器

    将文本文件按段落分割，每个段落作为一个结构化元素。
    """

    def parse(self, filepath: str) -> list[StructuredElement]:
        """
        解析文本文件

        Args:
            filepath: 文本文件路径

        Returns:
            结构化元素列表
        """
        created_at = _file_mtime(filepath)
        elem_meta = {"created_at": created_at} if created_at else {}
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        elements = self.parse_string(text)
        for el in elements:
            el.metadata = elem_meta
        return elements

    def parse_string(self, text: str) -> list[StructuredElement]:
        """
        解析文本字符串

        Args:
            text: 文本内容

        Returns:
            结构化元素列表
        """
        result = []
        for para in text.split("\n\n"):
            para = para.strip()
            if para:
                result.append(
                    StructuredElement(content=para, element_type="paragraph")
                )
        return result
