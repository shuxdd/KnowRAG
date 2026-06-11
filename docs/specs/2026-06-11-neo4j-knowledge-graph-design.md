# Neo4j 知识图谱集成设计

## 概述

在 KnowRAG 现有向量检索（Milvus）+ BM25 的双路检索架构上，引入 Neo4j 知识图谱作为第三路检索源。采用按需抽取策略——只对高频命中的父块进行实体关系抽取，降低 LLM 调用成本。

### 目标

1. **多跳推理**：回答需要跨越多个文档段落关联的问题（如"A 和 B 有什么关系"、"X 的上游依赖有哪些"）
2. **结构化知识展示**：让用户能浏览实体关系图谱，探索文档中的知识网络
3. **增强检索质量**：作为向量+BM25 的第三路检索源，通过实体关系找到纯语义搜索遗漏的相关内容

---

## 图数据模型

### 节点类型

```cypher
// 实体节点
(:Entity {
  id: UUID,
  name: String,         // 归一化后的实体名称
  type: String,         // LLM 推断的类型（概念/技术/组织/人物/...）
  description: String,  // LLM 生成的一句话描述
  user_id: Int,
  created_at: DateTime
})

// 父块节点（轻量引用，与 PostgreSQL parent_chunks 对应）
(:ParentChunk {
  id: UUID,             // 与 PostgreSQL 中的 id 一致
  filename: String,
  heading_path: String, // JSON 序列化
  user_id: Int
})
```

### 关系类型

```cypher
// 实体间的语义关系
(Entity)-[:RELATES_TO {
  relation: String,     // 关系类型，如 "依赖"、"属于"、"导致"
  context: String,      // 原文中支撑该关系的句子
  chunk_id: UUID        // 来源父块 id
}]->(Entity)

// 实体到父块的提及关系
(Entity)-[:MENTIONED_IN {
  frequency: Int        // 在该父块中出现的次数
}]->(ParentChunk)
```

### 设计要点

- ParentChunk 节点只存 id、filename、heading_path 等元数据，实际内容仍在 PostgreSQL，避免数据冗余
- 实体名称通过 LLM prompt 约束归一化（同义实体合并为一个标准名）
- 关系带 context 字段，检索时可作为额外证据返回给 LLM
- user_id 字段隔离多用户数据，与现有 Milvus/PostgreSQL 一致

---

## 按需抽取策略

不对所有文档做全量抽取，而是通过检索命中频率驱动，只对高频命中的父块触发实体抽取。

### 检索计数表（PostgreSQL）

```sql
CREATE TABLE parent_chunk_retrieval_stats (
    chunk_id UUID PRIMARY KEY REFERENCES parent_chunks(id),
    hit_count INTEGER DEFAULT 0,
    last_hit_at TIMESTAMP,
    extracted BOOLEAN DEFAULT FALSE
);
```

### 流程

```
用户查询 → 三路检索（Vector + BM25，Graph 初始为空）
         → expand_to_parents 返回结果
         → [async] 更新命中父块的 hit_count（仅统计 window_days 内的命中）
         → if hit_count >= threshold AND extracted == False:
              [async] LLM 抽取实体关系 → 写入 Neo4j → 标记 extracted = True
```

滑动窗口实现：查询时 `WHERE last_hit_at >= NOW() - INTERVAL '$window_days days'`，过期记录 hit_count 归零。每次命中时更新 `last_hit_at`。

### 配置项

```python
kg_hit_threshold: int = 3        # 命中几次后触发抽取
kg_hit_window_days: int = 30     # 统计窗口（天）
```

### 效果对比

| 场景 | 全量抽取 | 按需抽取 |
|---|---|---|
| 100 篇文档上传 | 100 次 LLM 调用 | 0 次 |
| 用户问 20 个问题（命中 15 个不重复父块，阈值=3） | 0 次 | ~5 次 LLM 调用 |
| 图谱覆盖 | 100% | 仅热点内容 |

### 手动触发

新增端点 `POST /api/kg/extract/{document_id}`，用户可主动标记重要文档进行全量抽取，绕过阈值限制。前端在文档列表中提供"加入图谱"按钮调用此接口。

---

## 实体关系抽取

### 抽取粒度

按父块（~1500 字）独立抽取。每个父块一次 LLM 调用。

### LLM Prompt

```
从以下文本中抽取实体和它们之间的关系。

要求：
1. 实体名称归一化（同义实体合并为一个标准名）
2. 每个实体给出类型和一句话描述
3. 关系要有明确的类型（如：依赖、属于、导致、对比、包含）
4. 标注支撑该关系的原文句子
5. 实体最多 20 个，关系最多 15 个

输出 JSON 格式：
{
  "entities": [
    {"name": "...", "type": "...", "description": "..."}
  ],
  "relations": [
    {"source": "...", "target": "...", "relation": "...", "context": "..."}
  ]
}

文本内容：
{parent_chunk_content}
```

### 查询实体提取

从用户问题中提取实体名，用于图检索：

```
从以下问题中提取关键实体名称，输出 JSON 数组。
要求：与知识图谱中的归一化实体名保持一致。

问题：{query}
输出：["实体1", "实体2"]
```

---

## 图写入逻辑

### 新建 `backend/services/graph_service.py`

```python
# 伪代码
async def build_graph_for_chunk(parent_chunk, entities, relations, user_id):
    # 1. 创建/合并父块节点
    MERGE (c:ParentChunk {id: $chunk_id})
      SET c.filename = $filename, c.heading_path = $heading_path

    # 2. 创建/合并实体节点 + MENTIONED_IN 关系
    FOR entity IN entities:
        MERGE (e:Entity {name: $name, user_id: $user_id})
          ON CREATE SET e.type = $type, e.description = $description
        MERGE (e)-[:MENTIONED_IN]->(c)

    # 3. 创建实体间关系
    FOR rel IN relations:
        MATCH (s:Entity {name: $source, user_id: $user_id})
        MATCH (t:Entity {name: $target, user_id: $user_id})
        MERGE (s)-[:RELATES_TO {relation: $relation, chunk_id: $chunk_id}]->(t)
```

用 `MERGE` 而非 `CREATE`——同一用户多篇文档中出现的同名实体自动合并，关系追加而非重复。

### 删除文档时的清理

```cypher
// 1. 删除该文档所有父块的 MENTIONED_IN 关系
MATCH (c:ParentChunk {filename: $filename, user_id: $user_id})<-[:MENTIONED_IN]-(e:Entity)
DETACH DELETE c

// 2. 删除这些父块产生的实体间关系
MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
WHERE r.chunk_id IN $chunk_ids AND s.user_id = $user_id
DELETE r

// 3. 清理孤立实体
MATCH (e:Entity {user_id: $user_id}) WHERE NOT (e)--() DELETE e
```

其中 `$chunk_ids` 是该文档所有父块 id 的列表，从 PostgreSQL 查询获得。

---

## 图检索与融合

### 检索流水线改动

```
用户 query
    ↓
QueryRewriter（现有）
    ↓
三路并行检索：
  ├─ Vector search（Milvus）→ leaf chunks
  ├─ BM25（jieba 分词）→ leaf chunks
  └─ Graph search（Neo4j）→ parent chunk ids  ← 新增
    ↓
RRF 融合（三路结果合并排序）
    ↓
expand_to_parents（现有逻辑）
```

### 图检索流程

```python
async def _graph_retrieve(self, query: str, user_id: int, top_k: int) -> list[str]:
    # 1. LLM 从 query 中提取实体名
    entities = await self._extract_query_entities(query)

    # 2. Neo4j 查找匹配实体
    # MATCH (e:Entity) WHERE e.name IN $entities AND e.user_id = $user_id

    # 3. 从匹配实体出发，1-2 跳遍历 RELATES_TO 关系
    # 收集沿途关联的 ParentChunk id
    # MATCH (e)-[:RELATES_TO*1..2]->(related)-[:MENTIONED_IN]->(c:ParentChunk)

    # 4. 按相关度排序（跳数越近越相关，关系数越多越相关）
    # 返回 parent chunk id 列表
```

### RRF 融合改动

现有 `rrf_fusion` 接收 `doc_lists: list[list]`，把图检索结果作为第三路传入：

```python
doc_lists = [vector_results, bm25_results, graph_results]
```

图检索返回 parent chunk id，需先通过 `parent_id` 字段映射到 leaf chunks，再走现有 RRF 逻辑。

### 策略覆盖

| 策略 | 改动 |
|---|---|
| `fast` | 不变（图检索有 LLM 调用开销，不适合 fast） |
| `precise` | Vector + BM25 + Graph → RRF |
| `deep` | Vector + BM25 + Graph → RRF → Rerank |

---

## 结构化知识浏览 API

### 新增路由：`backend/routers/knowledge_graph.py`

| 端点 | 功能 |
|---|---|
| `GET /api/kg/stats` | 图谱统计：实体总数、关系总数、实体类型分布 |
| `GET /api/kg/entities` | 实体列表：支持按类型筛选、按名称搜索、分页 |
| `GET /api/kg/entities/{id}` | 实体详情：信息 + 所有关系 + 关联父块 |
| `GET /api/kg/search` | 图谱搜索：输入关键词，返回匹配实体及 1-2 跳邻居 |
| `GET /api/kg/relations` | 关系列表：支持按关系类型筛选，分页 |
| `POST /api/kg/extract/{document_id}` | 手动触发：对指定文档全量抽取实体关系 |

### 响应示例

```json
// GET /api/kg/entities/{id}
{
  "entity": {
    "id": "abc-123",
    "name": "BGE-large-zh-v1.5",
    "type": "技术",
    "description": "智源发布的中文文本嵌入模型"
  },
  "relations": [
    {
      "direction": "outgoing",
      "relation": "属于",
      "target": {"id": "...", "name": "BGE 系列", "type": "技术"},
      "context": "BGE-large-zh-v1.5 是 BGE 系列中..."
    }
  ],
  "mentioned_in": [
    {"chunk_id": "...", "filename": "architecture.md", "heading_path": ["技术架构", "向量检索"]}
  ]
}
```

### 前端页面

新增 `frontend/src/pages/KGPage.tsx`：
- 实体搜索框
- 实体列表（按类型分组）
- 点击实体后展示详情面板（关系列表 + 关联文档段落）
- 关系可视化（可选，后续迭代用 D3.js 或 vis.js）

---

## 基础设施

### Docker Compose

```yaml
neo4j:
  image: neo4j:5-community
  container_name: knowrag-neo4j
  ports:
    - "7474:7474"
    - "7687:7687"
  environment:
    - NEO4J_AUTH=neo4j/knowrag123
    - NEO4J_PLUGINS=["apoc"]
  volumes:
    - neo4j_data:/data
  networks:
    - milvus
```

### 配置项（`backend/config.py`）

```python
# Neo4j
neo4j_uri: str = "bolt://localhost:7687"
neo4j_user: str = "neo4j"
neo4j_password: str = "knowrag123"

# 图谱抽取
kg_extract_model: str = ""            # 空则复用 mimo_model
kg_extract_concurrency: int = 3
kg_max_entities_per_chunk: int = 20
kg_max_relations_per_chunk: int = 15
kg_hit_threshold: int = 3
kg_hit_window_days: int = 30
```

### 依赖

```
neo4j>=5.0.0
```

---

## 文件改动清单

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `docker-compose.yml` | 修改 | 新增 Neo4j 服务 |
| `requirements.txt` | 修改 | 新增 `neo4j` |
| `backend/config.py` | 修改 | 新增 Neo4j + KG 配置项 |
| `backend/models/db_models.py` | 修改 | 新增 `RetrievalStatsORM` |
| `backend/models/schemas.py` | 修改 | 新增图谱相关 Pydantic 模型 |
| `backend/services/graph_service.py` | **新建** | Neo4j 客户端、图写入、图查询、清理 |
| `backend/services/entity_extractor.py` | **新建** | LLM 实体关系抽取 |
| `backend/services/hybrid_retriever.py` | 修改 | 新增 `_graph_retrieve`，RRF 三路融合，检索计数 |
| `backend/services/document_service.py` | 修改 | 删除文档时清理图谱 |
| `backend/routers/documents.py` | 修改 | 删除接口调用图谱清理 |
| `backend/routers/knowledge_graph.py` | **新建** | 图谱浏览 API |
| `frontend/src/pages/KGPage.tsx` | **新建** | 知识图谱浏览页面 |
| `frontend/src/App.tsx` | 修改 | 新增 KG 路由 |
| `frontend/src/components/Layout.tsx` | 修改 | 导航栏新增 KG 入口 |
| `alembic/versions/` | **新建** | 新增 retrieval_stats 表迁移 |
