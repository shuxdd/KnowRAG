import uuid
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import settings

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": Docx2txtLoader,
}

CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def get_loader_class(file_type: str):
    loader_cls = LOADER_MAP.get(file_type.lower())
    if loader_cls is None:
        raise ValueError(f"不支持的文件类型: {file_type}，支持的格式: {list(LOADER_MAP.keys())}")
    return loader_cls


def load_and_split_document(file_path: str, file_type: str) -> tuple[list, str]:
    doc_id = uuid.uuid4().hex[:12]
    filename = Path(file_path).name

    loader_cls = get_loader_class(file_type)
    if loader_cls is TextLoader:
        loader = loader_cls(file_path, encoding="utf-8")
    else:
        loader = loader_cls(file_path)
    raw_docs = loader.load()

    if not raw_docs:
        raise ValueError(f"文档内容为空: {filename}")

    # inject doc-level metadata to every page/segment before splitting
    for doc in raw_docs:
        doc.metadata["doc_id"] = doc_id
        doc.metadata["filename"] = filename
        doc.metadata["file_type"] = file_type

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=CHINESE_SEPARATORS,
    )
    chunks = splitter.split_documents(raw_docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks, doc_id
