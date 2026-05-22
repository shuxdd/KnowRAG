# KnowRAG 技术总结

## 技术亮点总览

### 检索层

- **三路并行检索引擎**：Dense 向量检索（bge-small-zh-v1.5）+ Sparse BM25 关键词检索（jieba 分词）+ HyDE 假设答案检索，三路并发执行各取 top-10，通过 RRF 倒数排名融合后由 CrossEncoder（bge-reranker-base）精细重排序，最终展开叶子块为父块完整上下文送入 LLM
- **两层查询路由**：第一层正则规则零成本匹配（聊天/简单事实/复杂问题），第二层 LLM 语义路由，自动在 fast（纯向量，<50ms）/ precise（向量+BM25）/ deep（全流水线）三种策略间切换
- **HyDE 假设答案检索**：LLM 先针对问题生成简短的假设答案，再将原始问题与假设答案拼接做 embedding 检索——用 LLM 的语言缩小用户问题与知识库文档之间的语义鸿沟
- **RRF + 重排序**：倒数排名融合合并多路异构检索结果，不依赖绝对分数；CrossEncoder 将 query-doc 对拼接送入 Transformer 做精细二分类相关度判断，弥补双塔 embedding 的交互缺失
- **Redis 检索缓存**：相同查询命中缓存直接返回，文档变更时批量清理，TTL 600s

### 分块层

- **父子双存分块策略**：H1/H2 标题触发父块边界（~1500 字 → PostgreSQL），父块内容经 RecursiveCharacterTextSplitter 切为细粒度叶子块（~300 字，30 字重叠 → ChromaDB）。表格和代码原子保留不切割，不足 100 字的叶子块合并到相邻块防止碎片化。检索时以叶子为入口，以父块为上下文返回，兼顾精度与完整度
- **语义分块兜底**：超限父块无 H3 边界可供切割时，先用 bge-small-zh-v1.5 编码每个句子，在相邻句子余弦相似度骤降处切分，实现话题级的分割精度，失败时自动降级到字数均分

### Agent 层

- **多步推理 Agent**（基于 LangGraph）：LLM 自主拆解复杂问题为 2-5 个子问题 → 各子问题并行执行独立的 ReAct 循环（LLM 决策调哪个工具 → 执行 → 看结果决定继续还是结束）→ 综合所有子答案生成完整回答 → 自反思节点评估质量，不通过则自动改写查询回环重搜（最多 2 轮）
- **六款工具自主调度**：语义检索、带反馈增强检索（质量评估 + 自动改写重搜）、章节级精读、多文档对比、文档列表、分段预览——LLM 在 ReAct 循环中自主决定调用哪些工具及调用顺序
- **并行子问题研究**：通过 asyncio.Queue 实现多子问题并发执行，子问题之间独立 ReAct，SSE 事件按完成顺序实时推送到前端，不相互阻塞

### 工程层

- **SSE 流式响应**：检索→token 生成→反思→来源的完整流水线事件通过 SSE 逐条推送，前端实时展示子任务进度卡片和思考过程
- **多格式文档解析**：支持 Markdown、DOCX、PDF（基于 PyMuPDF）、TXT 四种格式，统一产出 `StructuredElement` 列表，结构化解析失败时自动回退到 LangChain 加载器
- **完整评估体系**：55 对 QA 测试集 + RAGAS 五项指标（Faithfulness、ContextRecall、ContextPrecision、AnswerCorrectness、AnswerAccuracy），支持三种策略的对比评估，结果持久化到 SQLite
- **查询改写**：多轮对话场景中的指代消解与子问题拆分，从对话历史中将"它""那个"还原为具体实体
- **MinerU OCR 集成**：针对扫描版 PDF 提供 OCR 解析管线，提取结构化元素
- **三库协作**：PostgreSQL（父块）+ ChromaDB（向量索引）+ SQLite（会话与评估），Redis 做检索缓存

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

完整流水线：向量 + BM25 + HyDE 三路 → RRF 融合 → CrossEncoder 重排 → 父块展开。HyDE 的核心思想是让 LLM 先写一个假设答案，用假设答案去做向量检索，缩小 query 与文档之间的语义鸿沟。

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
| HyDE | LLM 生成假设答案辅助检索 | 缩小语义差距 |
| 融合 | RRF (k=60) | 倒数排名融合，不依赖绝对分数 |
| 重排序 | bge-reranker-base (CrossEncoder) | Query-Doc 拼接精细判断 |
| 上下文 | expand_to_parents | 叶子块展开为完整父块 |

---

## 四、Agentic RAG

代码在 `backend/services/agent_service.py`。

### 4.1 与普通 RAG 的本质区别

| | 普通 RAG | Agentic RAG |
|---|---|---|
| 检索次数 | 固定 1 次 | LLM 自主决定，可多次 |
| 策略选择 | 正则+LLM路由 | LLM 自己选 tool 传参 |
| 工具 | 只有检索 | 6 个工具 |
| 问题处理 | 单问题一次性 | 拆子问题 → 并行研究 → 自检回环 |
| 质量保证 | 无 | Reflect 节点自检 |

Agent 是决策层，普通 RAG 是执行层。`search_docs` 内部直接调用 `qa_service.search()`，即完整的三条检索流水线。

### 4.2 Agent 流程

```
步骤1: 问题拆解 (decompose)
  LLM 分析问题复杂度，拆解为 2-5 个子问题

步骤2: 并行研究 (research)
  每个子问题启动独立的 LangGraph ReAct 循环：
    agent_node: LLM 决定调哪个 Tool 还是直接回答
      ↓ (有 tool_calls)
    tools_node: 执行工具调用
      ↓
    agent_node: 再看结果，决定继续调 Tool 还是输出

步骤3: 答案综合 (synthesize)
  所有子问题结果喂给 LLM，综合成完整回答

步骤4: 自反思 (reflect)
  LLM 自检质量 → pass=true 结束 / pass=false 回环补搜（最多 2 轮）
```

### 4.3 六款工具

| 工具 | 价值 | 说明 |
|------|------|------|
| `search_docs` | 核心（90%调用） | 内部调 qa_service.search()，即三条检索流水线 |
| `search_with_feedback` | 高 | 检索→LLM评估质量→不满意改写重搜，最多 2 轮 |
| `list_docs` | 有用 | "知识库有什么"，语义检索答不了 |
| `read_section` | 有用 | 按 heading_path 精读章节，与语义检索互补 |
| `compare_docs` | 有用 | 取完整文档对比，非 top_k 片段对比 |
| `get_chunks` | 调试用 | 查看分块结构，终端用户基本不问 |

### 4.4 SSE 事件流

兼容普通 RAG 的 SSE 格式，新增事件类型：

```json
{"type": "decompose", "data": "拆解为3个子问题：..."}
{"type": "step",      "data": {"sub_q": 0, "text": "正在处理: ...", "status": "running"}}
{"type": "step",      "data": {"sub_q": 0, "text": "完成", "status": "done"}}
{"type": "tool",      "data": {"sub_q": 1, "text": "调用工具: search_docs..."}}
{"type": "thinking",  "data": {"sub_q": 1, "text": "根据检索结果..."}}
{"type": "token",     "data": "综合来看..."}
{"type": "reflect",   "data": "自检通过"}
{"type": "sources",   "data": [...]}
{"type": "done"}
```

---

## 五、评估体系

### 5.1 评估指标（RAGAS 框架）

- **Faithfulness**：答案是否忠于检索到的上下文
- **ContextRecall**：检索上下文覆盖标准答案的程度
- **ContextPrecision**：检索上下文中相关内容的比例
- **AnswerCorrectness**：与标准答案的对比
- **AnswerAccuracy**：LLM 自定义评判（覆盖关键事实的比例）

### 5.2 运行方式

```bash
python -m backend.eval_cli --strategy all --limit 10
# 策略可选: fast / precise / deep / all / vector / hybrid / hybrid_rerank
```

评估结果存入 `data/eval_results.db`（SQLite），可通过 API `/api/eval/results` 查看。

### 5.3 前提条件

- Docker PostgreSQL 运行中（父块存储）
- ChromaDB 有数据（需先上传文档）
- Redis 运行中（检索缓存）
- QWEN_API_KEY 有效

---

## 六、技术栈总览

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI |
| LLM | 通义千问 (DashScope, OpenAI 兼容 API) |
| Agent 框架 | LangGraph (StateGraph + Tool Use) |
| 向量数据库 | ChromaDB |
| 关系数据库 | PostgreSQL (父块) + SQLite (会话/评估) |
| 缓存 | Redis |
| Embedding | bge-small-zh-v1.5 (HuggingFace) |
| Reranker | bge-reranker-base (CrossEncoder) |
| 分词 | jieba |
| 前端 | React + TypeScript + Vite |
| 流式 | SSE (Server-Sent Events) |
