# CLAUDE.md

本文件为 Claude Code 在此仓库中工作时提供指导。

## 常用命令

```bash
# 后端（需先启动 PostgreSQL Docker 容器，端口 5433）
docker-compose up -d
.venv/Scripts/uvicorn backend.main:app --reload --port 8000

# 前端（Vite 开发服务器，端口 5173，/api 代理到 localhost:8000）
cd frontend && npm run dev

# 测试（仅在涉及核心逻辑变更、用户明确要求或验证修复时运行）
.venv/Scripts/pytest tests/ -v
.venv/Scripts/pytest tests/path/to/test_file.py::test_name -v

# 评估 CLI（独立运行，不需要 API）
python -m backend.eval_cli --strategy hybrid_rerank --limit 10

# Alembic（启动时自动执行，也可手动迁移）
alembic upgrade head
```

## 架构

**父子双存**：细粒度叶子块（200-300 字）存入 ChromaDB 做向量检索，粗粒度父块（~1500 字）存入 PostgreSQL。检索时搜叶子块，然后 `expand_to_parents` 将父块完整内容返回给 LLM，提供更完整的上下文。

**检索流水线**（`hybrid_retriever.py`）：三路并行检索（向量检索 bge-small-zh-v1.5 + BM25 结合 jieba 分词 + HyDE 用 LLM 生成假设答案辅助检索）→ 各取 10 条 → RRF 融合（k=60，取 top 10）→ CrossEncoder 重排序（bge-reranker-base，取 top k）→ 展开到父块。

**分块策略**（`hierarchical_chunker.py`）：H1/H2 标题触发父块边界，H3 及以下留在当前父块内。叶子块优先语义分块（句子 embedding 相似度骤降处切分），失败时回退 RecursiveCharacterTextSplitter（300 字，30 字重叠）。表格和代码块原子保留。仅含标题无正文的父块被跳过。不足 100 字的叶子块合并到相邻块。超限父块无 H3 时尝试语义分块再降级均分。

**解析管道**（`services/parsing/`）：四种格式解析器（Markdown、DOCX、PDF 基于 PyMuPDF、TXT）产出 `StructuredElement` 列表 → `HierarchicalChunker` 产出 (ParentChunk, LeafChunk) 对 → `ParentStore`（PostgreSQL）+ `VectorService.add_leaves`（ChromaDB）。结构化解析失败时回退到 LangChain 加载器。

**Agentic RAG**（`agent_service.py`）：基于 LangGraph 的多步推理 Agent。LLM 拆解复杂问题为子问题 → 各子问题并行执行独立 ReAct 循环（LLM 自主调用 6 款工具：语义检索、带反馈增强检索、章节精读、多文档对比、文档列表、分段预览）→ 综合子答案 → 自反思节点评估质量，不通过则改写重搜（最多 2 轮）→ SSE 流推送 `decompose`、`step`、`tool`、`thinking`、`token`、`reflect`、`sources`、`done` 事件。

**流式问答**（`qa_service.py`）：从 SQLite 加载对话历史 → `QueryRewriter` 处理指代消解和子问题拆分 → `HybridRetriever` 检索文档 → SSE 流依次推送 `sources`、`token`、`done` 事件 → 消息持久化到 `data/sessions.db`。

**评估模块**（`eval_service.py`）：加载 `data/test_qa_pairs.json`（55 对 QA）→ 对每种策略（vector/hybrid/hybrid_rerank）：检索文档、生成答案、自定义 LLM 准确率评分 → 构建 RAGAS `SingleTurnSample` 列表 → 运行 `ragas.evaluate()`（Faithfulness、ContextRecall、ContextPrecision、AnswerCorrectness 四项指标）→ 结果存入 `data/eval_results.db`。

## 关键模式

- **所有服务为模块级单例**（导入时即初始化）：`qa_service`、`vector_service`、`reranker`、`hybrid_retriever`、`document_service`、`parent_store`、`session_service`、`eval_service`
- **三个数据库**：PostgreSQL 存父块（端口 5433），ChromaDB 存叶子向量（`data/chroma_db/`），SQLite 存会话和评估结果
- **CORS**：后端仅允许 `http://localhost:5173`。Vite 将 `/api` 代理到 `localhost:8000`
- **HuggingFace 镜像**：多处设置 `HF_ENDPOINT=https://hf-mirror.com` 用于国内网络
- **LLM**：通过 DashScope 调用 Qwen（OpenAI 兼容 API），在 `.env` 中配置 `QWEN_API_KEY`、`QWEN_BASE_URL`、`QWEN_MODEL`
- **启动时**：Alembic 自动迁移到最新版本，预加载模型（reranker CrossEncoder + embedding 预热）
- **配置**：`backend/config.py` 通过 pydantic-settings 读取 `.env`，所有配置项均设有本地开发默认值
- **测试原则**：仅在以下情况运行测试：1) 用户明确要求 2) 涉及核心检索/分块/Agent 逻辑的修改 3) 验证 bug 修复。小改动（格式调整、配置文件、前端样式等）无需跑测试，导入检查即可
