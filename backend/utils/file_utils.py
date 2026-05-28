"""
文件工具模块

提供文件验证相关的工具函数：
- validate_file(): 验证上传文件的类型
- validate_file_size(): 验证上传文件的大小

支持的文件类型：.pdf, .docx, .txt, .md
最大文件大小：由 settings.max_upload_size_mb 配置
"""

import os
from fastapi import UploadFile, HTTPException
from backend.config import get_settings

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def validate_file(file: UploadFile) -> str:
    """
    验证上传文件的类型和大小

    Args:
        file: FastAPI 上传的文件对象

    Returns:
        文件扩展名

    Raises:
        HTTPException: 文件名无效、类型不支持、或文件过大时抛出
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Size check via content-length header before reading file into memory
    content_length = file.headers.get("content-length") if file.headers else None
    if content_length:
        validate_file_size(int(content_length))

    return ext


def validate_file_size(size: int) -> None:
    """
    验证上传文件的大小

    Args:
        size: 文件大小（字节）

    Raises:
        HTTPException: 文件大小超过限制时抛出
    """
    max_bytes = get_settings().max_upload_size_mb * 1024 * 1024
    if size > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size / 1024 / 1024:.1f}MB). Max: {max_bytes / 1024 / 1024:.0f}MB",
        )
