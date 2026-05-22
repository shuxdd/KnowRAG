"""
文件工具模块

提供文件验证相关的工具函数：
- validate_file(): 验证上传文件的类型
- validate_file_size(): 验证上传文件的大小

支持的的文件类型：.pdf, .docx, .txt, .md
最大文件大小：50MB
"""

import os
from fastapi import UploadFile, HTTPException

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024


def validate_file(file: UploadFile) -> str:
    """
    验证上传文件的类型

    Args:
        file: FastAPI 上传的文件对象

    Returns:
        文件扩展名

    Raises:
        HTTPException: 文件名无效或文件类型不支持时抛出
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    return ext


def validate_file_size(size: int) -> None:
    """
    验证上传文件的大小

    Args:
        size: 文件大小（字节）

    Raises:
        HTTPException: 文件大小超过限制时抛出
    """
    if size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size / 1024 / 1024:.1f}MB). Max: {MAX_UPLOAD_SIZE / 1024 / 1024:.0f}MB",
        )
