# KnowRAG 技术总结

## 技术亮点总览

### 检索层

- **两路并行检索引擎**：Dense 向量检索（bge-small-zh-v1.5）+ Sparse BM25 关键词检索（jieba 分词），两路并发执行各取 top-10，通过 RRF 倒数排名融合后由 CrossEncoder（bge-reranker-base）精细重排序，最终展开叶子块为父块完整上下文送入 LLM
- **两层查询路由**：第一层正则规则零成本匹配（聊天/简单事实/复杂问题），第二层 LLM 语义路由，自动在 fast（纯向量，<50ms）/ precise（向量+BM25）/ deep（全流水线）三种策略间切换
- **RRF + 重排序**：倒数排名融合合并多路异构检索结果，不依赖绝对分数；CrossEncoder 将 query-doc 对拼接送入 Transformer 做精细二分类相关度判断，弥补双塔 embedding 的交互缺失
- **Redis 检索缓存**：相同查询命中缓存直接返回，文档变更时批量清理，TTL 600s

### 分块层

- **父子双存分块策略**：H1/H2 标题触发父块边界（~1500 字 → PostgreSQL），父块内容经 RecursiveCharacterTextSplitter 切为细粒度叶子块（~300 字，30 字重叠 → ChromaDB）。表格和代码原子保留不切割，不足 100 字的叶子块合并到相邻块防止碎片化。检索时以叶子为入口，以父块为上下文返回，兼顾精度与完整度
- **语义分块兜底**：超限父块无 H3 边界可供切割时，先用 bge-small-zh-v1.5 编码每个句子，在相邻句子余弦相似度骤降处切分，实现话题级的分割精度，失败时自动降级到字数均分

### 工程层

- **SSE 流式响应**：检索→token 生成→反思→来源的完整流水线事件通过 SSE 逐条推送，前端实时展示子任务进度卡片和思考过程
- **多格式文档解析**：支持 Markdown、DOCX、PDF（基于 PyMuPDF）、TXT 四种格式，统一产出 `StructuredElement` 列表，结构化解析失败时自动回退到 LangChain 加载器

- **查询改写**：多轮对话场景中的指代消解与子问题拆分，从对话历史中将"它""那个"还原为具体实体
- **MinerU OCR 集成**：针对扫描版 PDF 提供 OCR 解析管线，提取结构化元素
- **双库协作**：PostgreSQL（父块、会话、消息）+ Milvus（向量索引），Redis 做检索缓存

---

## 一、普通 RAG 三种检索模式

项目在 `backend/services/hybrid_retriever.py` 中实现了三条检索流水线，通过 `backend/services/query_router.py` 自动选择策略。

### 1.1 fast（快速模式）

```
用户Query → Embedding(bge-small-zh-v1.5) → ChromaDB向量相似度搜索 → top_k个叶子块 → expand_to_parents → 返回
```

仅做向量语义检索，最快但可能漏掉关键词匹配的内容。适合简单事实型问题。

### 1.2 precise（精确模式）

```
用户Query ─┬─→ Embedding → ChromaDB向量检索（语义）──→ 各取10条 ─→ RRF融合(k=60) → top 10 → 返回
           │
           └─→ jieba分词 → BM25倒排检索（关键词）──→
```

向量 + BM25 两路并行，RRF 融合后返回。语义 + 关键词互补，大部分场景够用。

### 1.3 deep（深度模式）

```
用户Query ─┬─→ Embedding → ChromaDB向量检索 ────→
           ├─→ jieba分词 → BM25检索 ────────────→  三路各取10条
           └─→ LLM生成假设答案 → Embedding → 检索 →    │
                                                       ↓
                                                 RRF融合(k=60) → top 10
                                                       ↓
                                                 CrossEncoder重排序(bge-reranker-base)
                                                       ↓
                                                 取 top k → expand_to_parents → 返回
```

完整流水线：向量 + BM25 两路 → RRF 融合 → CrossEncoder 重排 → 父块展开。

### 1.4 策略路由（QueryRouter）

两层路由决策，代码在 `backend/services/query_router.py`：

```
第一层：正则规则匹配（零成本）
  ├─ 聊天检测（你好/hi/谢谢…）         → chat，不检索直接回答
  ├─ 简单事实检测（是多少/在哪/是谁…）   → fast，仅向量检索
  └─ 未命中                           → 进入第二层

第二层：LLM 路由提示
  从 QueryRewriter 的 rewrite 结果中提取 route 字段
  如果 LLM 返回 "fast"/"precise"/"deep" → 使用该策略

第三层：兜底
  以上都没命中 → 默认 deep（最全流水线）
```

用户可在请求中显式指定 `strategy` 参数（非 `"auto"`）直接跳过路由。

---

## 二、父子双存分块策略

代码在 `backend/services/chunking/hierarchical_chunker.py`。

### 2.1 整体流程

```
文档 → 解析器(Markdown/DOCX/PDF/TXT) → StructuredElement列表
                                          │
                                          ▼
                                   HierarchicalChunker
                                          │
                            ┌─────────────┴─────────────┐
                            ▼                           ▼
                    ParentChunk (粗粒度)           LeafChunk (细粒度)
                    1500字 / PostgreSQL          300字 / ChromaDB
```

### 2.2 父块构建

- **H1/H2 标题**触发父块边界（`flush_parent`），H3 及以下留在当前父块内
- 父块内容 = 该章节下所有元素的拼接文本，目标 ≤ 1500 字
- 仅含标题无正文的父块跳过

### 2.3 叶子块构建

1. 分离文本元素和保留元素（表格、代码 → `preserve=True`）
2. 文本用 `RecursiveCharacterTextSplitter` 切割（300 字，30 字重叠）
3. 保留元素原子追加，不切割
4. 不足 100 字的叶子块合并到相邻块（跳过 preserve 块）

### 2.4 超限父块处理

超 1500 字的父块走 `_split_oversized`：
- **有 H3**：按 H3 边界切，子父块的 heading_path 追加 H3 标题
- **无 H3**：先尝试语义分块，失败则按字数均分（heading_path 追加 `"(continued)"`）

### 2.5 检索时父子联动

```
用户Query → 向量检索Leaf块 → 命中叶子 → leaf.parent_id
                                                │
                                    expand_to_parents()
                                                │
                                    返回ParentChunk完整1500字内容
```

叶子块只是检索入口，LLM 最终看到的是父块的完整上下文。

### 2.6 语义分块（2026-05 新增）

在 `_split_by_paragraphs` 中，超限父块且无 H3 边界时，先尝试语义分块再降级均分：

1. 将所有文本按句子切分（中英文标点感知）
2. 用 bge-small-zh-v1.5 编码每个句子
3. 计算相邻句子余弦相似度
4. 在相似度低于 `均值 - 0.5×标准差` 的位置切分
5. 合并过小组（< 200 字），确保每组有意义
6. 为每组建一个父块及其叶子

语义模型惰性加载，不影响启动速度。

---

## 三、混合检索引擎核心技术栈

| 环节 | 技术 | 说明 |
|------|------|------|
| Dense 检索 | bge-small-zh-v1.5 + ChromaDB | 语义相似度搜索 |
| Sparse 检索 | BM25 + jieba 分词 | 关键词精确匹配 |
| 融合 | RRF (k=60) | 倒数排名融合，不依赖绝对分数 |
| 重排序 | bge-reranker-base (CrossEncoder) | Query-Doc 拼接精细判断 |
| 上下文 | expand_to_parents | 叶子块展开为完整父块 |

---

## 四、技术栈总览

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI |
| LLM | Mimo v2.5 (OpenAI 兼容 API) |
| 向量数据库 | Milvus (HNSW + COSINE) |
| 关系数据库 | PostgreSQL (父块、会话、消息、用户) |
| 缓存 | Redis |
| Embedding | bge-large-zh-v1.5 (HuggingFace) |
| Reranker | bge-reranker-base (CrossEncoder) |
| 分词 | jieba |
| 前端 | React + TypeScript + Vite |
| 流式 | SSE (Server-Sent Events) |
