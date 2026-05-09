import os
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.models.schemas import (
    UploadResponse,
    DocumentListResponse,
    DocumentInfo,
)
from backend.services.document_service import document_service
from backend.services.vector_service import vector_service
from backend.utils.file_utils import validate_file

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    validate_file(file)
    content = await file.read()

    filepath = document_service.save_upload(content, file.filename)
    doc_id = document_service.process_file(filepath, file.filename)

    collection_results = vector_service.collection.get(
        where={"filename": file.filename}
    )
    chunks_count = len(collection_results["ids"]) if collection_results["ids"] else 0

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        chunks_count=chunks_count,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    stats = vector_service.get_document_stats()
    documents = []
    for s in stats:
        filename = s["filename"]
        file_size = 0
        for f in os.listdir("data/uploads"):
            if f.endswith("_" + filename):
                filepath = os.path.join("data/uploads", f)
                file_size = os.path.getsize(filepath)
                break
        documents.append(
            DocumentInfo(
                doc_id=s.get("filename", ""),
                filename=filename,
                file_size=file_size,
                chunks_count=s["chunks_count"],
                uploaded_at=datetime.now().isoformat(),
            )
        )
    return DocumentListResponse(documents=documents)


@router.delete("/{doc_id:path}")
async def delete_document(doc_id: str):
    deleted = vector_service.delete_by_filename(doc_id)
    for f in os.listdir("data/uploads"):
        if f.endswith("_" + doc_id):
            os.remove(os.path.join("data/uploads", f))
            break
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"detail": f"Deleted {deleted} chunks"}
