from dataclasses import dataclass


@dataclass
class ParentChunk:
    id: str
    content: str
    heading_path: list[str]
    filename: str
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class LeafChunk:
    id: str
    content: str
    heading_path: list[str]
    parent_id: str
    filename: str
    page: int | None = None
    chunk_index: int = 0
    preserve: bool = False
