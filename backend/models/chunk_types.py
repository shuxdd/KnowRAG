"""
数据结构模块

定义文档分块（Chunk）的数据结构，用于在内存中表示分块内容。
包含两种分块类型：
- ParentChunk: 父块，代表文档的完整章节内容
- LeafChunk: 叶子块，通过分块得到的较小文本单元
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ParentChunk:
    """
    父块数据结构

    父块是文档分块的第一层，代表文档中的一个完整章节。
    包含章节的完整内容、标题路径、页码范围等信息。
    检索时返回父块以提供完整的上下文。

    属性:
        id: 唯一标识符
        content: 父块的完整文本内容
        heading_path: 标题路径列表（如 ["第一章", "1.1 简介"]）
        filename: 所属文件名
        page_start: 起始页码（可选）
        page_end: 结束页码（可选）
        created_at: 创建时间
    """
    id: str
    content: str
    heading_path: list[str]
    filename: str
    page_start: int | None = None
    page_end: int | None = None
    created_at: datetime | None = None


@dataclass
class LeafChunk:
    """
    叶子块数据结构

    叶子块是文档分块的第二层，通过对父块进行更细粒度的分块得到。
    每个叶子块是一个较小的文本单元，适合用于向量检索。

    属性:
        id: 唯一标识符
        content: 叶子块文本内容
        heading_path: 所属父块的标题路径
        parent_id: 所属父块的 ID
        filename: 所属文件名
        page: 页码（可选）
        chunk_index: 在父块内的分块索引
        preserve: 是否被保留不参与合并（表格、代码等特殊内容标记为 True）
    """
    id: str
    content: str
    heading_path: list[str]
    parent_id: str
    filename: str
    page: int | None = None
    chunk_index: int = 0
    preserve: bool = False
