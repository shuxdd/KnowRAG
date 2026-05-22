"""
解析器基类模块

定义文档解析器的基类和结构化元素数据结构。
所有文档解析器（PDF、DOCX、Markdown 等）都继承自 BaseParser。

结构化元素（StructuredElement）表示文档中的一个逻辑单元：
- heading: 标题（h1-h6）
- paragraph: 段落文本
- table: 表格
- code: 代码块
- list: 列表
"""

from dataclasses import dataclass, field
from typing import Literal

ElementType = Literal["heading", "paragraph", "table", "code", "list"]


@dataclass
class StructuredElement:
    """
    结构化元素数据类

    表示文档解析后的一个逻辑单元。

    属性:
        content: 元素文本内容
        element_type: 元素类型（heading/paragraph/table/code/list）
        heading_level: 标题级别（1-6，仅 heading 类型有效）
        page: 页码（仅 PDF 有效）
        metadata: 额外元数据（如 created_at）
    """
    content: str
    element_type: ElementType
    heading_level: int | None = None
    page: int | None = None
    metadata: dict = field(default_factory=dict)


class BaseParser:
    """
    文档解析器基类

    所有文档解析器都应继承此类并实现 parse 方法。
    """

    def parse(self, filepath: str) -> list[StructuredElement]:
        """
        解析文档文件

        Args:
            filepath: 文档文件路径

        Returns:
            结构化元素列表
        """
        raise NotImplementedError
