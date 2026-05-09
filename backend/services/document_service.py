import os
import uuid
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.config import get_settings
from backend.services.vector_service import vector_service

settings = get_settings()


class DocumentService:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def process_file(self, filepath: str, filename: str) -> str:
        ext = os.path.splitext(filename)[1].lower()
        ext = ".txt" if ext == ".md" else ext

        if ext == ".pdf":
            loader = PyPDFLoader(filepath)
        elif ext == ".docx":
            loader = Docx2txtLoader(filepath)
        elif ext == ".txt":
            loader = TextLoader(filepath, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        raw_docs = loader.load()
        chunks = self.splitter.split_documents(raw_docs)

        for i, chunk in enumerate(chunks):
            chunk.metadata["filename"] = filename
            chunk.metadata["chunk_index"] = i
            chunk.metadata["source"] = filename
            if "page" not in chunk.metadata:
                chunk.metadata["page"] = 0

        doc_ids = vector_service.add_documents(chunks)
        return doc_ids[0] if doc_ids else ""

    def save_upload(self, file_content: bytes, filename: str) -> str:
        os.makedirs(settings.upload_dir, exist_ok=True)
        unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
        filepath = os.path.join(settings.upload_dir, unique_name)
        with open(filepath, "wb") as f:
            f.write(file_content)
        return filepath

    def get_file_size(self, filepath: str) -> int:
        return os.path.getsize(filepath)


document_service = DocumentService()
