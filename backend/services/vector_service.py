import os
import uuid
from typing import List
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from backend.config import get_settings

settings = get_settings()


class VectorService:
    def __init__(self):
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        self.client = PersistentClient(path=settings.chroma_persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, docs: List[Document]) -> List[str]:
        ids = [str(uuid.uuid4()) for _ in docs]
        self.collection.add(
            ids=ids,
            documents=[doc.page_content for doc in docs],
            metadatas=[doc.metadata for doc in docs],
        )
        return ids

    def similarity_search(self, query: str, k: int = 10) -> List[Document]:
        results = self.collection.query(query_texts=[query], n_results=k)
        docs = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                score = 1.0 - (distance / 2.0)
                docs.append(
                    Document(
                        page_content=results["documents"][0][i],
                        metadata={
                            **metadata,
                            "doc_id": doc_id,
                            "score": max(0.0, min(1.0, score)),
                        },
                    )
                )
        return docs

    def delete_by_filename(self, filename: str) -> int:
        results = self.collection.get(where={"filename": filename})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0

    def get_document_stats(self) -> List[dict]:
        results = self.collection.get()
        if not results["metadatas"]:
            return []
        stats = {}
        for meta in results["metadatas"]:
            fn = meta.get("filename", "unknown")
            if fn not in stats:
                stats[fn] = {"filename": fn, "chunks_count": 0}
            stats[fn]["chunks_count"] += 1
        return list(stats.values())

    def get_all_chunks(self) -> List[Document]:
        results = self.collection.get()
        if not results["ids"]:
            return []
        docs = []
        for i, doc_id in enumerate(results["ids"]):
            docs.append(
                Document(
                    page_content=results["documents"][i],
                    metadata=results["metadatas"][i] if results["metadatas"] else {},
                )
            )
        return docs

    def count(self) -> int:
        return self.collection.count()


vector_service = VectorService()
