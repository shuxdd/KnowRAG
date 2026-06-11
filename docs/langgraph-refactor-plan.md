# LangGraph 重构 QA 流水线 + 反思重试

## Context

当前 QA 流水线是硬编码的线性流程（`qa_service.py:388-524`）。用 LangGraph StateGraph 重构，新增反思节点：生成答案后 LLM 自评质量，不足则升级检索策略重试一次。所有现有服务（hybrid_retriever、query_router、query_rewriter 等）不改动。

## 文件变更

| 文件 | 操作 |
|------|------|
| `backend/services/rag_graph.py` | **新建** ~300 行，StateGraph 定义 + 9 个节点 |
| `backend/services/qa_service.py` | **修改** 仅替换 `ask_stream()` 方法体（388-524 行 → ~25 行） |
| `requirements.txt` | 不改（langgraph 已安装） |

## 图拓扑

```
START → route → [chat/rag 分支]
                  ├─ chat → chat_generate ─────────────┐
                  └─ rag  → rewrite → classify → retrieve → generate → reflect → [ok/retry]
                              ↑                                        │
                              └──── retry (升级策略) ──────────────────┘
                                                                            ↓
                                                                      stream_generate → persist → END
```

route 放最前面，用正则规则零成本判断。chat 直接跳过改写和检索。

## RAGState (TypedDict)

```python
class RAGState(TypedDict):
    # 输入
    question: str
    session_id: str
    user_id: int | None
    top_k: int
    strategy: str
    history_text: str
    # route 节点产出
    actual_strategy: str       # fast/precise/deep/chat
    # rewrite 节点产出
    queries: List[str]
    rewrite_result: dict
    # classify 节点产出
    intent: str
    # retrieve 节点产出
    docs: List[Document]
    sources: List[Source]
    context: str
    # generate 节点产出
    answer: str                # 第一次非流式生成的答案（供 reflect 评估）
    gen_messages: list         # 生成时用的 prompt messages（供 stream_generate 重用）
    # reflect 节点产出
    reflection: str            # "ok" / "insufficient"
    reflection_reason: str
    retry_count: int           # 上限 1
    needs_retry: bool
```

## 9 个节点

### 1. `route` — 策略路由
- 调用: `query_router.route(query)` 仅用正则规则，不依赖 LLM hint
- 用户显式指定 strategy 时直接使用
- SSE: thinking(route)

### 2. `rewrite` — 查询改写（仅 RAG 路径）
- 调用: `query_rewriter.rewrite(query, history_text)`
- 指代消解、拼写纠错、查询扩展、子问题拆分
- SSE: thinking(rewrite), thinking(sub_queries)

### 3. `classify` — 意图分类（仅 RAG 路径）
- 调用: `qa_service._classify_intent(question)`
- SSE: thinking(intent)

### 4. `retrieve` — 文档检索
- 调用: `hybrid_retriever._fast/_precise/_deep_retrieve()`，按 `actual_strategy` 分发
- 多查询时 `rrf_fusion` 合并
- SSE: thinking(retrieval progress), sources

### 5. `generate` — 非流式生成答案（缓冲，不发给客户端）
- 调用: `qa_service._get_prompt(intent)` + `llm.invoke()`
- 答案存到 `state["answer"]`，prompt messages 存到 `state["gen_messages"]`
- SSE: thinking(synthesize)

### 6. `chat_generate` — 闲聊路径（跳过检索和反思）
- 调用: `llm.astream()` 直接流式输出
- SSE: sources, token

### 7. `reflect` — 答案质量自评
- LLM 调用评估答案是否充分、是否基于上下文
- 升级策略映射: fast→precise, precise→deep, deep 不可升级
- 最多重试 1 次
- SSE: thinking(reflect), thinking(retry)

### 8. `stream_generate` — 用 `llm.astream()` 重新流式生成
- reflect 通过后，用 `state["gen_messages"]` 调 `llm.astream()` 重新生成
- 因为 prompt 相同，LLM prompt cache 命中，延迟极低
- 用户看到真正的逐 token 流式输出，体验和当前完全一致
- SSE: token

### 9. `persist` — 持久化会话
- 调用: `session_service.add_message()`, `_maybe_compress()`
- SSE: done

## SSE 流式桥接

使用 `get_stream_writer()` + `stream_mode="custom"`:
- 节点内调用 `writer({"type": "thinking", ...})` 直接发送 SSE 事件
- `qa_service.ask_stream()` 消费: `async for event in rag_graph.stream(state, stream_mode="custom"): yield f"data: {json.dumps(event)}\n\n"`
- 前端无感知，SSE 格式完全不变

## qa_service.py 改动

仅替换 `ask_stream()` 方法体（388-524 行），改为：
```python
async def ask_stream(self, question, session_id, strategy="auto", top_k=5, user_id=None):
    from backend.services.rag_graph import rag_graph  # 延迟导入避免循环
    history = session_service.get_history(session_id)
    history_text = self._format_history(session_id, history.messages)
    initial_state = { ... }  # 填充 RAGState 所有字段
    async for event in rag_graph.stream(initial_state, stream_mode="custom"):
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
```

其余所有方法（`ask()`, `search()`, `answer_from_docs()`, `_classify_intent`, `_build_context` 等）不变。

## 实施顺序

1. langgraph 已安装，无需操作
2. 新建 `backend/services/rag_graph.py`（RAGState + 9 节点 + 图构建）
3. 修改 `qa_service.py` 的 `ask_stream()` 方法
4. 启动后端验证：普通问答、闲聊、反思重试流程

## 验证

1. langgraph 已安装
2. 启动后端 `uvicorn backend.main:app --reload --port 8000`
3. 前端提问 → 验证 SSE 事件序列正确（thinking → sources → token → done）
4. 验证反思节点：构造一个 fast 策略返回不足答案的场景，确认自动升级到 precise 重试
5. 验证 chat 路径不受影响
6. 验证 retry 上限为 1（deep 策略不再升级）
