import os
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.config import settings
from backend.services.document_service import load_and_split_document
from backend.services.vector_service import add_documents, list_documents, delete_document
from backend.models.schemas import DocumentResponse, DocumentListResponse

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    original_name = file.filename or "unknown"
    ext = Path(original_name).suffix.lower()
    if ext not in settings.supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型：{ext}，支持：{settings.supported_extensions}",
        )

    content = await file.read()
    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="文件大小超出限制")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_name = f"{uuid.uuid4().hex}_{original_name}"
    saved_path = upload_dir / saved_name
    saved_path.write_bytes(content)

    try:
        chunks, doc_id = load_and_split_document(str(saved_path), ext)
    except ValueError as e:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))

    doc_response = DocumentResponse(
        doc_id=doc_id,
        filename=original_name,
        file_type=ext,
        chunk_count=len(chunks),
        uploaded_at=datetime.now(),
        size_bytes=len(content),
    )
    add_documents(chunks, doc_response)

    return doc_response


@router.get("", response_model=DocumentListResponse)
async def get_documents():
    docs = list_documents()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.delete("/{doc_id}")
async def remove_document(doc_id: str):
    from backend.services.vector_service import _doc_registry

    if doc_id not in _doc_registry:
        raise HTTPException(status_code=404, detail="文档不存在")
    delete_document(doc_id)

    # clean up uploaded file
    upload_dir = Path(settings.upload_dir)
    for f in upload_dir.iterdir():
        if doc_id in f.name:
            f.unlink(missing_ok=True)

    return {"status": "deleted", "doc_id": doc_id}
