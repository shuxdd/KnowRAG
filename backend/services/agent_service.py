"""
Agent 服务模块

基于 LangGraph 的多步推理 Agent，采用编排图 + 子问题 ReAct 图两层架构。

可用工具（3 个）：
- search_docs: 搜索知识库文档，支持 fast/precise/deep/auto 四种策略
- list_docs: 列出知识库中所有文档及章节数、页码范围等元信息
- read_section: 精读指定文档的章节，支持章节路径匹配和语义搜索两种模式

编排流程（orchestration_graph）：
1. decompose: LLM 拆解复杂问题为多个子问题（简单问题保持原样）
2. parallel research: 各子问题并行执行独立 ReAct 循环（LLM 自主调用工具）
3. synthesize: 综合各子问题答案生成完整回答
4. reflect: 自反思节点评估回答质量，不通过则改写重搜（最多 2 轮）

ReAct 图（react_graph）：
- LLM 绑定 3 个工具，自主决定调用时机和参数
- 最多 10 轮工具调用，防止无限循环
- 所有工具调用有 30 秒超时保护

SSE 事件流：
- decompose: 子问题分解结果
- step: 子问题研究进度
- tool: 工具调用详情
- thinking: Agent 思考过程
- token: LLM 生成的文本片段
- reflect: 反思评估结果
- sources: 最终引用来源
- done: 结束标识
"""

import asyncio
import json
import logging
import os
import re
from typing import TypedDict, AsyncIterator

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

from backend.config import get_settings
from backend.models.schemas import Source
from backend.services.qa_service import qa_service
from backend.services.vector_service import vector_service
from backend.services.parent_store import parent_store
from backend.services.session_service import session_service

logger = logging.getLogger(__name__)

settings = get_settings()

SYSTEM_PROMPT = """你是一个企业知识库助手。你可以使用以下工具来回答问题：

- search_docs(query, strategy, top_k): 搜索知识库中的文档内容。
  这是你最主要的工具。遇到任何知识库相关问题都应先调用它。
  query: 搜索关键词或问题。如果首次搜索结果不理想，换个角度改写 query 再搜。
  strategy: "fast"（快速）/"precise"（混合）/"deep"（深度）/"auto"（自动，推荐）。
  top_k: 返回数量（1-20），默认5，问题范围广时可增大。

- list_docs(): 列出知识库中所有文档及章节数、页码范围等元信息。
  当用户问"有哪些文档"、"知识库里有什么"、或需要了解文档概况时使用。

- read_section(doc_filename, heading_path, query): 精读文档的某个章节。
  当 search_docs 返回的结果不够详细、需要查看原文时使用。
  doc_filename: 文档文件名（如"员工手册.pdf"）。
  两种用法（二选一）：
  1. heading_path: 按章节路径精确查找（如["休假政策", "年假"]）
  2. query: 在文档内语义搜索（如"年假天数规定"），返回最相关章节

规则：
- 知识库问题必须先调 search_docs，结果不够时再调 read_section 补充。
- 如果首次 search_docs 结果不理想，尝试改写 query 重新搜索，不要直接放弃。
- 对比类问题：先 search_docs 获取相关文档，再多次 read_section 获取详细内容，最后自己推理对比。
- 回答时注明引用的文档来源。
- 始终用中文回答，不要编造检索结果中没有的信息。"""

DECOMPOSE_PROMPT = """你是一个问题分析助手。分析用户的问题，判断其复杂度并拆解为子问题。

规则：
- 简单问题（问候、单一事实查询、简单定义）：返回单个子问题（即原始问题本身）。
- 复杂问题（跨文档对比、多条件分析、需要多个步骤推理）：拆解为2-5个子问题，每个子问题应独立可检索。
- 子问题应简洁、具体，便于知识库检索。

请严格按以下JSON格式输出，不要包含其他文字：
{"complexity": "simple"|"complex", "sub_questions": ["子问题1", "子问题2", ...]}"""

SYNTHESIZE_PROMPT = """你是一个知识整合助手。根据以下子问题的分析结果，综合生成完整、连贯的回答。

原始问题：{question}

子问题分析结果：
{results}

要求：
- 综合所有子问题的答案，不要简单罗列。
- 如果有子问题检索失败，诚实说明该部分信息缺失。
- 引用来用时注明文档来源（文件名）。
- 始终用中文回答。"""

REFLECT_PROMPT = """你是一个质量检查助手。评估以下回答的质量，判断是否需要补充检索。

原始问题：{question}

子问题列表：{sub_questions}

当前回答：{answer}

检查清单：
1. 回答是否覆盖了所有子问题？
2. 是否有"未找到信息"但可以尝试改写查询重新检索的部分？
3. 回答中是否存在事实矛盾？
4. 引用是否准确（引用的文档是否真的支持该结论）？

请严格按以下JSON格式输出：
{{"pass": true|false, "refinement_query": "补充检索的查询词（仅当pass为false时需要）", "reason": "简短说明（通过或未通过的原因）"}}

注意：如果回答已经完整、准确，pass应为true，不要为微小瑕疵要求补充检索。"""


class MultiStepState(TypedDict):
    session_id: str
    question: str
    chat_history: str
    messages: list[BaseMessage]
    sub_questions: list[str]
    current_step: int
    research_results: list[dict]
    final_answer: str
    reflection_count: int
    needs_refinement: bool


class MultiStepAgentService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.mimo_model,
            api_key=settings.mimo_api_key,
            base_url=settings.mimo_base_url,
            temperature=0.3,
        )

    # ---- 工具实现（接受 user_id / sources_container 参数，不使用 contextvars）----

    def _search_docs_impl(
        self, query: str, strategy: str, top_k: int,
        user_id: int, sources_container: list[Source],
    ) -> str:
        """搜索知识库中的文档内容。"""
        docs = qa_service.search(query, strategy, top_k, user_id=user_id)
        if not docs:
            return "知识库中未找到相关文档。"
        for doc in docs:
            sources_container.append(Source(
                content=doc.page_content[:300],
                filename=doc.metadata.get("filename", "unknown"),
                score=round(doc.metadata.get("score", 0.0), 4),
            ))
        return qa_service._build_context(docs)

    def _list_docs_impl(self, user_id: int) -> str:
        """列出知识库中的所有文档及元信息。"""
        stats = vector_service.get_document_stats(user_id=user_id)
        if not stats:
            return "知识库中没有文档。"

        lines = [f"共 {len(stats)} 个文档:"]
        for s in stats:
            filename = s["filename"]
            ext = os.path.splitext(filename)[1] if "." in filename else ""
            parents = parent_store.get_by_filename(filename, user_id=user_id)
            chapter_count = len(parents)
            page_range = ""
            if parents:
                pages = [p.page_start for p in parents if p.page_start]
                if pages:
                    page_range = f", 页码范围: {min(pages)}-{max(pages)}"
            lines.append(
                f"  - {filename} ({ext}  {s.get('chunks_count', 0)} 段, "
                f"{chapter_count} 章{page_range})"
            )
        return "\n".join(lines)

    def _read_section_impl(
        self, doc_filename: str,
        heading_path: list[str] | None,
        query: str,
        user_id: int,
    ) -> str:
        """精读指定文档的某个章节。"""
        try:
            parents = parent_store.get_by_filename(doc_filename, user_id=user_id)
        except Exception as e:
            logger.warning(f"parent_store lookup failed for '{doc_filename}': {e}")
            return f"查询文档 {doc_filename} 时出错: {e}"

        if not parents:
            return f"未找到文档: {doc_filename}"

        # Mode 1: exact heading path match
        if heading_path:
            for p in parents:
                if p.heading_path == heading_path:
                    return (
                        f"`{doc_filename}` / {' > '.join(heading_path)}\n"
                        f"(页码 {p.page_start}-{p.page_end})\n\n{p.content}"
                    )

            candidates = [p for p in parents if any(h in p.heading_path for h in heading_path)]
            if candidates:
                lines = [f"未精确匹配 '{' > '.join(heading_path)}'，相近章节："]
                for c in candidates[:5]:
                    lines.append(f"  - {' > '.join(c.heading_path)} ({len(c.content)} 字)")
                return "\n".join(lines)

            return f"文档 `{doc_filename}` 中未找到与 '{' > '.join(heading_path)}' 相关的章节"

        # Mode 2: semantic search within document
        if query:
            try:
                results = vector_service.query_with_filter(
                    query=query,
                    where={"filename": doc_filename},
                    n_results=min(5, len(parents)),
                    user_id=user_id,
                )
                if results["ids"] and results["ids"][0]:
                    matched_parents: dict[str, int] = {}
                    for i, _doc_id in enumerate(results["ids"][0]):
                        meta = results["metadatas"][0][i] if results["metadatas"] else {}
                        pid = meta.get("parent_id", "")
                        if pid:
                            matched_parents[pid] = matched_parents.get(pid, 0) + 1

                    sorted_pids = sorted(matched_parents, key=matched_parents.get, reverse=True)
                    lines = [f"`{doc_filename}` 中与 '{query}' 最相关的章节:"]
                    for pid in sorted_pids[:3]:
                        p = next((p for p in parents if p.id == pid), None)
                        if p:
                            lines.append(
                                f"\n[{' > '.join(p.heading_path)}]\n"
                                f"(页码 {p.page_start}-{p.page_end})\n{p.content[:1500]}"
                            )
                    return "\n".join(lines)
            except Exception:
                logger.warning(f"Within-document search failed for '{doc_filename}'", exc_info=True)

        # Fallback: return document outline
        lines = [f"`{doc_filename}` 的章节结构:"]
        for p in parents[:20]:
            heading = " > ".join(p.heading_path)
            lines.append(f"  - {heading} (页码 {p.page_start}-{p.page_end}, {len(p.content)} 字)")
        return "\n".join(lines)

    # ---- Async timeout wrapper -----------------------------------------------

    TOOL_TIMEOUT = 30

    async def _run_with_timeout(self, func, timeout: int, error_msg: str, *args, **kwargs):
        """Run a sync function in thread executor with timeout."""
        try:
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Tool timeout ({timeout}s): {func.__name__ if hasattr(func, '__name__') else 'unknown'}")
            return error_msg

    # ---- ReAct 图工厂（接收 tools 列表，避免 contextvars）---------------------

    def _create_react_graph(self, tools):
        """Create a ReAct graph with the given tools."""
        llm_with_tools = self.llm.bind_tools(tools)

        def _agent_node(state: MultiStepState) -> dict:
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        MAX_TOOL_ROUNDS = 10

        def _should_continue(state: MultiStepState) -> str:
            last = state["messages"][-1]
            tool_calls = [
                m for m in state["messages"]
                if hasattr(m, "tool_calls") and m.tool_calls
            ]
            if len(tool_calls) >= MAX_TOOL_ROUNDS:
                return END
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        builder = StateGraph(MultiStepState)
        builder.add_node("agent", _agent_node)
        builder.add_node("tools", ToolNode(tools))
        builder.set_entry_point("agent")
        builder.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
        builder.add_edge("tools", "agent")

        return builder.compile()

    # ---- 辅助方法 -----------------------------------------------------------

    def _format_history(self, session_id: str, messages: list[BaseMessage] | None) -> str:
        if not messages:
            return ""

        from backend.services.session_service import session_service

        summary = session_service.get_summary(session_id)
        lines = []
        for msg in messages[-4:]:  # last 2 turns in full
            role = "用户" if msg.type == "human" else "助手"
            lines.append(f"{role}: {msg.content}")
        recent = "\n".join(lines)

        if summary:
            return f"[对话摘要]\n{summary}\n\n[最近对话]\n{recent}"
        return recent

    COMPRESS_PROMPT = (
        "Summarize the following conversation in 2-3 sentences. "
        "Keep key facts, decisions, entity names, and topics discussed. "
        "Write in the same language as the conversation.\n\n"
        "{input_text}\n\nSummary:"
    )

    def _maybe_compress(self, session_id: str):
        msgs = session_service.get_messages(session_id)
        if len(msgs) <= 10:
            return
        old_summary = session_service.get_summary(session_id)
        old_lines = []
        for msg in msgs[:-4]:
            role = "用户" if msg.get("role") == "user" else "助手"
            old_lines.append(f"{role}: {msg.get('content', '')}")
        old_text = "\n".join(old_lines)
        if not old_text:
            return

        input_text = old_text
        if old_summary:
            input_text = f"Previous summary: {old_summary}\n\nNew messages: {old_text}"

        prompt = self.COMPRESS_PROMPT.format(input_text=input_text)
        try:
            response = self.llm.invoke(prompt)
            new_summary = response.content.strip()
            if new_summary:
                session_service.update_summary(session_id, new_summary)
        except Exception:
            logger.warning("Agent conversation compression failed", exc_info=True)

    def _format_research_summary(self, results: list[dict]) -> str:
        if not results or all(r is None for r in results):
            return "(无检索结果)"
        lines = []
        for i, r in enumerate(results):
            if r is None:
                continue
            answer = r.get("answer", "检索失败")
            lines.append(f"子问题{i+1}：{r['sub_q']}\n回答：{answer}\n来源：{r.get('sources', [])}")
        return "\n\n".join(lines)

    async def _research_one_sub_q(
        self, sub_q: str, index: int, session_id: str, history_text: str, user_id: int,
    ):
        """并行执行单个子问题的 ReAct 研究。

        Yields 元组 ("tool"|"thinking", index, text) 供外层合并为 SSE。
        最后 yield ("__result__", index, dict) 携带结果。
        """
        sources_container: list[Source] = []

        # -- 工具闭包（捕获 user_id 和 sources_container，避免 contextvars）--

        async def _search_docs(query: str, strategy: str = "auto", top_k: int = 5) -> str:
            """搜索知识库中的文档内容。当用户询问事实性问题时使用此工具。"""
            def _impl():
                return self._search_docs_impl(query, strategy, top_k, user_id, sources_container)
            return await self._run_with_timeout(
                _impl, self.TOOL_TIMEOUT, "错误：搜索超时，请尝试缩小查询范围。"
            )

        async def _list_docs() -> str:
            """列出知识库中的所有文档及元信息。当用户询问文档列表时使用。"""
            def _impl():
                return self._list_docs_impl(user_id)
            return await self._run_with_timeout(
                _impl, self.TOOL_TIMEOUT, "错误：获取文档列表超时。"
            )

        async def _read_section(
            doc_filename: str,
            heading_path: list[str] | None = None,
            query: str = "",
        ) -> str:
            """精读指定文档的某个章节。支持章节路径匹配或语义搜索两种模式。"""
            def _impl():
                return self._read_section_impl(doc_filename, heading_path, query, user_id)
            return await self._run_with_timeout(
                _impl, self.TOOL_TIMEOUT, "错误：读取章节超时。"
            )

        search_tool = tool(_search_docs, name="search_docs")
        list_tool = tool(_list_docs, name="list_docs")
        read_tool = tool(_read_section, name="read_section")
        graph = self._create_react_graph([search_tool, list_tool, read_tool])

        try:
            react_state: MultiStepState = {
                "session_id": session_id,
                "question": sub_q,
                "chat_history": history_text,
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=f"历史对话:\n{history_text}\n\n用户问题: {sub_q}"),
                ],
                "sub_questions": [],
                "current_step": 0,
                "research_results": [],
                "final_answer": "",
                "reflection_count": 0,
                "needs_refinement": False,
            }

            sub_answer = ""
            async for event in graph.astream_events(react_state, version="v2"):
                kind = event.get("event")

                if kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    yield ("tool", index, f"调用工具: {tool_name}...")

                if kind == "on_tool_end":
                    output = event.get("data", {}).get("output", "")
                    if isinstance(output, str) and output:
                        preview = output[:200].replace("\n", " ")
                        yield ("tool", index, f"工具返回 ({len(output)} 字符): {preview}...")

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    token = chunk.content if hasattr(chunk, "content") and chunk.content else None
                    if token and not getattr(chunk, "tool_calls", None):
                        sub_answer += token
                        yield ("thinking", index, token)

            result = {
                "sub_q": sub_q,
                "answer": sub_answer or "检索未返回结果",
                "sources": [s.model_dump() for s in sources_container],
            }
            yield ("__result__", index, result)
        except Exception as e:
            logger.error(f"Sub-question research failed: {sub_q} — {e}", exc_info=True)
            yield ("__result__", index, {
                "sub_q": sub_q,
                "answer": "检索失败",
                "sources": [],
            })

    # ---- 流式入口 -----------------------------------------------------------

    async def ask_stream(
        self,
        question: str,
        session_id: str,
        chat_history_messages: list[BaseMessage] | None = None,
        user_id: int | None = None,
    ) -> AsyncIterator[str]:
        from backend.services.session_service import session_service

        actual_user_id = user_id or 0
        history_text = self._format_history(session_id, chat_history_messages)
        all_sources: list[Source] = []
        final_answer = ""

        try:
            # 1. 拆解问题
            response = self.llm.invoke([
                SystemMessage(content=DECOMPOSE_PROMPT),
                HumanMessage(content=f"用户问题：{question}"),
            ])
            raw = response.content.strip()
            json_match = re.search(r'\{[\s\S]*\}', raw)
            sub_questions = [question]
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    sub_questions = parsed.get("sub_questions", [question])
                    if not sub_questions or not isinstance(sub_questions, list):
                        sub_questions = [question]
                except json.JSONDecodeError:
                    pass

            sub_qs_text = "\n".join(f"{idx}. {q}" for idx, q in enumerate(sub_questions, 1))
            yield f"data: {json.dumps({'type': 'decompose', 'data': f'拆解为{len(sub_questions)}个子问题：\n{sub_qs_text}'}, ensure_ascii=False)}\n\n"

            # 2. 研究循环（可能伴随反思回环）
            reflection_count = 0
            max_reflections = 2

            while True:
                research_results: list[dict] = [None] * len(sub_questions)
                queue: asyncio.Queue = asyncio.Queue()
                pending = len(sub_questions)

                async def producer(sub_q: str, idx: int):
                    await queue.put(("step", idx, {"text": f"正在处理: {sub_q}", "status": "running"}))
                    try:
                        async for event in self._research_one_sub_q(sub_q, idx, session_id, history_text, actual_user_id):
                            await queue.put(event)
                    except Exception as e:
                        logger.error(f"Producer task crashed for sub_q[{idx}] '{sub_q}': {e}", exc_info=True)
                        await queue.put(("__error__", idx, str(e)))

                producers = [asyncio.create_task(producer(q, i)) for i, q in enumerate(sub_questions)]

                while pending > 0:
                    event = await queue.get()
                    event_type, idx, data = event

                    if event_type == "__result__":
                        research_results[idx] = data
                        pending -= 1
                        srcs = data.get("sources", [])
                        for s in srcs:
                            all_sources.append(Source(**s))
                        yield f"data: {json.dumps({'type': 'step', 'data': {'sub_q': idx, 'text': '完成', 'status': 'done'}}, ensure_ascii=False)}\n\n"
                    elif event_type == "__error__":
                        research_results[idx] = {"sub_q": sub_questions[idx], "answer": "检索失败", "sources": []}
                        pending -= 1
                    elif event_type == "step":
                        yield f"data: {json.dumps({'type': 'step', 'data': {'sub_q': idx, 'text': data['text'], 'status': data['status']}}, ensure_ascii=False)}\n\n"
                    elif event_type in ("tool", "thinking"):
                        yield f"data: {json.dumps({'type': event_type, 'data': {'sub_q': idx, 'text': data}}, ensure_ascii=False)}\n\n"

                await asyncio.gather(*producers)

                # 3. 合成答案
                results_text = self._format_research_summary(research_results)
                synth_prompt = SYNTHESIZE_PROMPT.format(question=question, results=results_text)
                final_answer = ""
                async for chunk in self.llm.astream([
                    SystemMessage(content=synth_prompt),
                    HumanMessage(content="请综合以上子问题分析结果，生成完整回答。"),
                ]):
                    token = chunk.content if hasattr(chunk, "content") and chunk.content else ""
                    if token:
                        final_answer += token
                        yield f"data: {json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)}\n\n"

                # 4. 自反思
                reflect_prompt = REFLECT_PROMPT.format(
                    question=question,
                    sub_questions=json.dumps(sub_questions, ensure_ascii=False),
                    answer=final_answer,
                )
                reflect_response = self.llm.invoke([
                    SystemMessage(content=reflect_prompt),
                    HumanMessage(content="请评估回答质量。"),
                ])
                raw_reflect = reflect_response.content.strip()
                json_match_r = re.search(r'\{[\s\S]*\}', raw_reflect)

                passed = True
                refinement_query = ""
                if json_match_r:
                    try:
                        reflect_parsed = json.loads(json_match_r.group(0))
                        passed = reflect_parsed.get("pass", True)
                        refinement_query = reflect_parsed.get("refinement_query", "")
                    except json.JSONDecodeError:
                        passed = True

                if passed or reflection_count >= max_reflections:
                    yield f"data: {json.dumps({'type': 'reflect', 'data': '自检通过' if passed else '已达最大自检次数，强制结束'}, ensure_ascii=False)}\n\n"
                    break
                else:
                    yield f"data: {json.dumps({'type': 'reflect', 'data': f'补充检索: {refinement_query}'}, ensure_ascii=False)}\n\n"
                    sub_questions = [refinement_query] if refinement_query else [question]
                    reflection_count += 1

        except asyncio.TimeoutError:
            logger.warning("Agent stream timeout, returning partial results")
            yield f"data: {json.dumps({'type': 'error', 'data': '处理超时，返回已收集的部分结果'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Agent stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': f'Agent 错误: {str(e)}'}, ensure_ascii=False)}\n\n"

        # 推送来源
        if all_sources:
            unique_sources = {s.filename: s for s in all_sources}.values()
            yield f"data: {json.dumps({'type': 'sources', 'data': [s.model_dump() for s in unique_sources]}, ensure_ascii=False)}\n\n"

        # 持久化会话
        try:
            session_service.add_message(session_id, "user", question)
            session_service.add_message(
                session_id, "assistant",
                final_answer or "处理时出错",
                [s.model_dump() for s in all_sources],
            )
            session = session_service.get_session(session_id)
            if session and session.get("title") == "新对话":
                title = question[:30] + ("..." if len(question) > 30 else "")
                session_service.update_title(session_id, title)
            self._maybe_compress(session_id)
        except Exception as e:
            logger.error(f"Failed to persist session: {e}")

        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"


agent_service = MultiStepAgentService()
