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
    """
    文档处理服务
    负责文档的加载、分块、存储等操作
    支持 PDF、DOCX、TXT 等格式
    """

    def __init__(self):
        """
        初始化文档分块器
        使用递归字符文本分割器，按指定大小和重叠量对文档进行切分
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def process_file(self, filepath: str, filename: str) -> str:
        """
        处理上传的文件，进行文档加载和分块

        Args:
            filepath: 文件的完整路径
            filename: 原始文件名

        Returns:
            返回第一个文档块的 ID，如果处理失败则返回空字符串

        Raises:
            ValueError: 当文件类型不支持时抛出
        """
        ext = os.path.splitext(filename)[1].lower()
        ext = ".txt" if ext == ".md" else ext

        # 根据文件扩展名选择合适的加载器
        if ext == ".pdf":
            loader = PyPDFLoader(filepath)
        elif ext == ".docx":
            loader = Docx2txtLoader(filepath)
        elif ext == ".txt":
            loader = TextLoader(filepath, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # 加载原始文档
        raw_docs = loader.load()
        # 对文档进行分块处理
        chunks = self.splitter.split_documents(raw_docs)

        # 为每个分块添加元数据信息
        for i, chunk in enumerate(chunks):
            chunk.metadata["filename"] = filename  # 原始文件名
            chunk.metadata["chunk_index"] = i      # 分块索引
            chunk.metadata["source"] = filename     # 来源标识
            if "page" not in chunk.metadata:
                chunk.metadata["page"] = 0          # 页码，默认为0

        # 将分块添加到向量数据库
        doc_ids = vector_service.add_documents(chunks)
        return doc_ids[0] if doc_ids else ""

    def save_upload(self, file_content: bytes, filename: str) -> str:
        """
        保存上传的文件到本地存储

        Args:
            file_content: 文件的字节内容
            filename: 原始文件名

        Returns:
            保存后的文件完整路径
        """
        os.makedirs(settings.upload_dir, exist_ok=True)
        # 生成唯一的文件名，防止冲突
        unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
        filepath = os.path.join(settings.upload_dir, unique_name)
        with open(filepath, "wb") as f:
            f.write(file_content)
        return filepath

    def get_file_size(self, filepath: str) -> int:
        """
        获取文件大小

        Args:
            filepath: 文件的完整路径

        Returns:
            文件大小（字节）
        """
        return os.path.getsize(filepath)


# 全局单例实例，供其他模块直接导入使用
document_service = DocumentService()
