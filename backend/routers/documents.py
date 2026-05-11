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

# 创建 /api/documents 前缀的路由组
router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    文档上传接口

    Args:
        file: 上传的文件（支持 PDF、DOCX、TXT）

    Returns:
        上传结果，包含：
        - doc_id: 文档 ID
        - filename: 文件名
        - chunks_count: 分块数量

    Raises:
        HTTPException: 如果文件类型不支持或验证失败
    """
    validate_file(file)  # 验证文件类型和大小
    content = await file.read()

    # 保存文件到本地存储
    filepath = document_service.save_upload(content, file.filename)
    # 处理文件（加载、分块、存入向量数据库）
    doc_id = document_service.process_file(filepath, file.filename)

    # 获取该文件的分块数量
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
    """
    获取已上传文档列表接口

    Returns:
        文档列表，每个文档包含：
        - doc_id: 文档 ID（实际为文件名）
        - filename: 文件名
        - file_size: 文件大小（字节）
        - chunks_count: 分块数量
        - uploaded_at: 上传时间
    """
    stats = vector_service.get_document_stats()
    documents = []
    for s in stats:
        filename = s["filename"]
        file_size = 0
        # 查找对应的上传文件以获取文件大小
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
    """
    删除指定文档接口

    Args:
        doc_id: 要删除的文档 ID（文件名）

    Returns:
        删除结果，包含删除的分块数量

    Raises:
        HTTPException: 如果文档不存在，返回 404 错误
    """
    # 从向量数据库删除
    deleted = vector_service.delete_by_filename(doc_id)
    # 从本地存储删除文件
    for f in os.listdir("data/uploads"):
        if f.endswith("_" + doc_id):
            os.remove(os.path.join("data/uploads", f))
            break
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"detail": f"Deleted {deleted} chunks"}
