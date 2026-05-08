import os
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import settings
from backend.models.schemas import DocumentResponse
from backend.services.hybrid_retriever import HybridRetriever

_embeddings = None
_vector_store = None
_doc_registry: dict[str, DocumentResponse] = {}
_bm25_corpus: list[Document] = []
_bm25_retriever: BM25Retriever | None = None


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


def _rebuild_bm25_retriever() -> None:
    global _bm25_retriever, _bm25_corpus
    if _bm25_corpus:
        _bm25_retriever = BM25Retriever.from_documents(_bm25_corpus)
    else:
        _bm25_retriever = None


def _rebuild_registry():
    """Rebuild in-memory document registry and BM25 corpus from ChromaDB."""
    global _doc_registry, _bm25_corpus
    _doc_registry.clear()
    _bm25_corpus.clear()
    try:
        store = Chroma(
            collection_name="knowrag_documents",
            embedding_function=get_embeddings(),
            persist_directory=str(Path(settings.chroma_persist_dir).resolve()),
        )
        results = store.get(include=["documents", "metadatas"])
        if not results or not results["ids"]:
            return
        seen = set()
        for id_, meta, doc_text in zip(results["ids"], results["metadatas"], results["documents"]):
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
            _bm25_corpus.append(Document(page_content=doc_text or "", metadata=meta))
        _rebuild_bm25_retriever()
    except Exception:
        pass


def add_documents(documents: list, doc_response: DocumentResponse) -> list[str]:
    global _bm25_corpus
    store = get_vector_store()
    ids = store.add_documents(documents)
    _doc_registry[doc_response.doc_id] = doc_response
    _bm25_corpus.extend(documents)
    _rebuild_bm25_retriever()
    return ids


def delete_document(doc_id: str) -> None:
    global _bm25_corpus
    store = get_vector_store()
    collection = store._collection
    results = collection.get(where={"doc_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])
    _doc_registry.pop(doc_id, None)
    _bm25_corpus = [d for d in _bm25_corpus if d.metadata.get("doc_id") != doc_id]
    _rebuild_bm25_retriever()


def list_documents() -> list[DocumentResponse]:
    return list(_doc_registry.values())


def get_document(doc_id: str) -> DocumentResponse | None:
    return _doc_registry.get(doc_id)


def get_retriever(top_k: int = 4):
    store = get_vector_store()
    return store.as_retriever(search_kwargs={"k": top_k})


def get_hybrid_retriever(top_k: int = 4):
    """Hybrid retriever: vector + BM25 via Reciprocal Rank Fusion.

    Falls back to vector-only if the BM25 corpus is empty.
    """
    global _bm25_retriever
    if _bm25_retriever is None:
        return get_retriever(top_k)

    vector_retriever = get_vector_store().as_retriever(search_kwargs={"k": top_k})
    return HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=_bm25_retriever,
        rrf_k=60,
        fetch_k=0,
    )


def get_document_count() -> int:
    try:
        store = get_vector_store()
        return store._collection.count()
    except Exception:
        return 0
