from dataclasses import dataclass, field
from typing import Literal

ElementType = Literal["heading", "paragraph", "table", "code", "list"]


@dataclass
class StructuredElement:
    content: str
    element_type: ElementType
    heading_level: int | None = None  # 1-6, only for heading
    page: int | None = None  # page number, only for PDF
    metadata: dict = field(default_factory=dict)


class BaseParser:
    def parse(self, filepath: str) -> list[StructuredElement]:
        raise NotImplementedError
