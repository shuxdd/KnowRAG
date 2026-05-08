import os
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import settings
from backend.models.schemas import DocumentResponse

_embeddings = None
_vector_store = None
_doc_registry: dict[str, DocumentResponse] = {}


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        persist_dir = str(Path(settings.chroma_persist_dir).resolve())
        _vector_store = Chroma(
            collection_name="knowrag_documents",
            embedding_function=get_embeddings(),
            persist_directory=persist_dir,
        )
        _rebuild_registry()
    return _vector_store


def _rebuild_registry():
    """Rebuild in-memory document registry from ChromaDB metadata."""
    global _doc_registry
    _doc_registry.clear()
    try:
        store = Chroma(
            collection_name="knowrag_documents",
            embedding_function=get_embeddings(),
            persist_directory=str(Path(settings.chroma_persist_dir).resolve()),
        )
        results = store.get()
        if not results or not results["ids"]:
            return
        seen = set()
        for meta in results["metadatas"]:
            doc_id = meta.get("doc_id")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                _doc_registry[doc_id] = DocumentResponse(
                    doc_id=doc_id,
                    filename=meta.get("filename", "unknown"),
                    file_type=meta.get("file_type", "unknown"),
                    chunk_count=sum(
                        1 for m in results["metadatas"] if m.get("doc_id") == doc_id
                    ),
                    uploaded_at=datetime.now(),
                    size_bytes=0,
                )
    except Exception:
        pass


def add_documents(documents: list, doc_response: DocumentResponse) -> list[str]:
    store = get_vector_store()
    ids = store.add_documents(documents)
    _doc_registry[doc_response.doc_id] = doc_response
    return ids


def delete_document(doc_id: str) -> None:
    store = get_vector_store()
    collection = store._collection
    results = collection.get(where={"doc_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])
    _doc_registry.pop(doc_id, None)


def list_documents() -> list[DocumentResponse]:
    return list(_doc_registry.values())


def get_document(doc_id: str) -> DocumentResponse | None:
    return _doc_registry.get(doc_id)


def get_retriever(top_k: int = 4):
    store = get_vector_store()
    return store.as_retriever(search_kwargs={"k": top_k})


def get_document_count() -> int:
    try:
        store = get_vector_store()
        return store._collection.count()
    except Exception:
        return 0
