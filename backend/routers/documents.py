"""
文档管理路由模块

提供文档上传、列表查询、删除、分块预览等文档管理接口。

接口列表：
- POST /api/documents/upload: 上传单个文档
- POST /api/documents/upload/batch: 批量上传文档
- GET /api/documents: 获取文档列表
- DELETE /api/documents: 删除所有文档
- DELETE /api/documents/{doc_id}: 删除指定文档
- GET /api/documents/{doc_id}/chunks: 获取文档分块预览

支持的文件类型：PDF、DOCX、TXT、Markdown

文档处理流程：
1. 验证文件类型和大小
2. 保存文件到上传目录
3. 解析文档结构（标题、段落、表格等）
4. 分块处理（父块 + 叶子块）
5. 存入向量数据库（ChromaDB）和关系数据库（PostgreSQL）
"""

import os
import logging
from datetime import datetime
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.models.schemas import (
    UploadResponse,
    DocumentListResponse,
    DocumentInfo,
    ChunkPreviewResponse,
    ParentChunkPreview,
    LeafChunkPreview,
)
from backend.config import get_settings
from backend.services.document_service import document_service
from backend.services.vector_service import vector_service
from backend.utils.file_utils import validate_file, validate_file_size

logger = logging.getLogger(__name__)

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
    validate_file(file)
    content = await file.read()
    validate_file_size(len(content))  # fallback check if content-length header missing

    try:
        filepath = document_service.save_upload(content, file.filename)
        doc_id = document_service.process_file(filepath, file.filename)
    except Exception as e:
        logger.error(f"Document processing failed for '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

    chunks_count = len(vector_service.get_by_filename(file.filename))

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        chunks_count=chunks_count,
    )


@router.post("/upload/batch")
async def upload_documents(files: List[UploadFile] = File(...)):
    """批量文档上传接口"""
    results = []
    for file in files:
        try:
            validate_file(file)
            content = await file.read()
            filepath = document_service.save_upload(content, file.filename)
            doc_id = document_service.process_file(filepath, file.filename)

            chunks_count = len(vector_service.get_by_filename(file.filename))
            results.append({
                "doc_id": doc_id,
                "filename": file.filename,
                "chunks_count": chunks_count,
                "status": "ok",
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e),
            })
    return {"results": results, "total": len(results)}


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
    try:
        stats = vector_service.get_document_stats()
        upload_dir = get_settings().upload_dir

        # Build filename -> filesize lookup in one pass
        file_sizes: dict[str, int] = {}
        if os.path.isdir(upload_dir):
            for f in os.listdir(upload_dir):
                prefix, sep, name = f.partition("_")
                if sep and name:
                    file_sizes[name] = os.path.getsize(os.path.join(upload_dir, f))

        documents = []
        for s in stats:
            filename = s["filename"]
            file_size = file_sizes.get(filename, 0)
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
    except Exception as e:
        logger.error(f"List documents error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.delete("")
async def delete_all_documents():
    stats = vector_service.get_document_stats()
    if not stats:
        return {"detail": "No documents to delete"}
    total_parents = 0
    total_leaves = 0
    deleted_files = 0
    for s in stats:
        filename = s["filename"]
        result = document_service.delete_file(filename)
        total_parents += result["parents"]
        total_leaves += result["leaves"]
        if result["parents"] > 0 or result["leaves"] > 0:
            deleted_files += 1
    return {"detail": f"Deleted {deleted_files} files, {total_leaves} leaf chunks, {total_parents} parent chunks"}


@router.delete("/{doc_id:path}")
async def delete_document(doc_id: str):
    result = document_service.delete_file(doc_id)
    if result["leaves"] == 0 and result["parents"] == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"detail": f"Deleted {result['leaves']} leaf chunks and {result['parents']} parent chunks"}


@router.get("/{doc_id:path}/chunks", response_model=ChunkPreviewResponse)
async def get_document_chunks(doc_id: str):
    from backend.services.parent_store import parent_store

    parents = parent_store.get_by_filename(doc_id)
    if not parents:
        raise HTTPException(status_code=404, detail="Document not found")

    parent_previews: list[ParentChunkPreview] = []
    for p in parents:
        leaf_results = vector_service.get_by_parent_id(p.id)
        leaf_docs = []
        for i, leaf in enumerate(leaf_results):
            meta = leaf["metadata"]
            content = leaf["document"]
            char_count = len(content)
            leaf_docs.append(LeafChunkPreview(
                chunk_index=meta.get("chunk_index", i),
                char_count=char_count,
                preserve=bool(meta.get("preserve", False)),
                undersized=char_count < 100,
                content_preview=content[:150],
            ))
        leaf_docs.sort(key=lambda l: l.chunk_index)
        parent_previews.append(ParentChunkPreview(
            id=p.id,
            heading_path=p.heading_path,
            char_count=len(p.content),
            page_start=p.page_start,
            page_end=p.page_end,
            created_at=p.created_at.isoformat() if p.created_at else None,
            content_preview=p.content[:200],
            leaves=leaf_docs,
        ))

    return ChunkPreviewResponse(filename=doc_id, parents=parent_previews)
