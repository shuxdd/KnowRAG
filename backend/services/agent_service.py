import asyncio
import contextvars
import json
import logging
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

logger = logging.getLogger(__name__)

settings = get_settings()

SYSTEM_PROMPT = """你是一个企业知识库助手。你可以使用以下工具来回答问题：

- search_docs(query, strategy, top_k): 搜索知识库中的文档内容。
  当用户询问知识库中的事实性问题时使用此工具。
  query: 搜索关键词或问题
  strategy: 检索策略。"fast"（快速关键词检索）、"precise"（混合检索）、"deep"（最全面的深度检索）、"auto"（自动选择，推荐）。
  top_k: 返回结果数量（1-20），默认5，问题范围较广时可设大一些。

- list_docs(): 列出知识库中所有文档。
  当用户问"有哪些文档"、"知识库里有什么"时使用此工具。

- get_chunks(doc_id): 查看某个文档的分段结构。
  当用户询问文档的分段方式、分块结构时使用。

规则：
- 问候、闲聊、感谢：直接回应，不调用工具。
- 知识库相关的问题：必须先调用 search_docs 检索。
- 如果没有找到相关文档，诚实告知用户。
- 回答时注明引用的文档来源（文件名）。
- 始终用中文回答。
- 不要编造检索结果中没有的信息。"""

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
            model=settings.qwen_model,
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            temperature=0.3,
        )
        self._last_search_docs_var: contextvars.ContextVar = contextvars.ContextVar(
            "last_search_docs", default=[]
        )
        self._last_search_sources_var: contextvars.ContextVar = contextvars.ContextVar(
            "last_search_sources", default=[]
        )
        self.react_graph = self._build_react_graph()
        self.orchestration_graph = self._build_orchestration_graph()

    # ---- 工具实现（Phase 1 原封不动）----------------------------------------

    def _search_docs_impl(self, query: str, strategy: str = "auto", top_k: int = 5) -> str:
        """搜索知识库中的文档内容。当用户询问事实性问题时使用此工具。"""
        docs = qa_service.search(query, strategy, top_k)
        self._last_search_docs_var.set(docs)
        if not docs:
            return "知识库中未找到相关文档。"

        self._last_search_sources_var.set([
            Source(
                content=doc.page_content[:300],
                filename=doc.metadata.get("filename", "unknown"),
                score=round(doc.metadata.get("score", 0.0), 4),
            )
            for doc in docs
        ])
        return qa_service._build_context(docs)

    def _list_docs_impl(self) -> str:
        """列出知识库中的所有文档。当用户询问文档列表时使用。"""
        stats = vector_service.get_document_stats()
        if not stats:
            return "知识库中没有文档。"
        lines = ["知识库中的文档:"]
        for s in stats:
            lines.append(f"  - {s['filename']} ({s.get('chunks_count', 0)} 个分段)")
        return "\n".join(lines)

    def _get_chunks_impl(self, doc_id: str) -> str:
        """查看指定文档的分段结构。当用户询问文档分块方式时使用。"""
        try:
            parents = parent_store.get_by_filename(doc_id)
            if not parents:
                return f"未找到文档: {doc_id}"
        except Exception as e:
            return f"查询文档 {doc_id} 时出错: {e}"

        lines = [f"`{doc_id}` 的分段预览:"]
        for p in parents:
            heading = "/".join(p.heading_path)
            lines.append(
                f"  [{p.id[:8]}...] {heading} "
                f"(字符数={len(p.content)}, 页码={p.page_start}-{p.page_end})"
            )
            try:
                leaf_results = vector_service.collection.get(where={"parent_id": p.id})
                leaf_count = len(leaf_results.get("ids", []))
                preserved = sum(
                    1 for m in (leaf_results.get("metadatas") or [])
                    if m and m.get("preserve")
                )
                lines.append(f"    {leaf_count} 个叶子块（{preserved} 个保留）")
            except Exception:
                lines.append("    (叶子信息不可用)")
        return "\n".join(lines[:80])

    def _read_section_impl(self, doc_filename: str, heading_path: list[str]) -> str:
        """精读指定文档的某个章节。"""
        try:
            parents = parent_store.get_by_filename(doc_filename)
        except Exception as e:
            return f"查询文档 {doc_filename} 时出错: {e}"

        if not parents:
            return f"未找到文档: {doc_filename}"

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

    def _resolve_target(self, target: dict) -> str:
        """解析 target 为文档内容字符串。

        target 格式：
        - {"doc_id": "parent_chunk_id"}  通过 ID 定位
        - {"doc": "filename", "heading": ["路径"]}  委托给 _read_section_impl
        - {"doc": "filename"}  返回整个文档
        """
        if "doc_id" in target:
            parents = parent_store.get_by_ids([target["doc_id"]])
            if not parents:
                return f"未找到文档ID: {target['doc_id']}"
            p = parents[0]
            return (
                f"`{p.filename}` / {' > '.join(p.heading_path)}\n"
                f"(页码 {p.page_start}-{p.page_end})\n\n{p.content}"
            )

        doc_filename = target.get("doc", "")
        heading = target.get("heading")

        if heading:
            return self._read_section_impl(doc_filename, heading)

        parents = parent_store.get_by_filename(doc_filename)
        if not parents:
            return f"未找到文档: {doc_filename}"
        parts = []
        for p in parents:
            parts.append(f"[{' > '.join(p.heading_path)}]\n{p.content}")
        return f"`{doc_filename}` 完整内容：\n\n" + "\n\n".join(parts)

    def _compare_docs_impl(self, target_a: dict, target_b: dict) -> str:
        """并排对比两个文档或章节。"""
        content_a = self._resolve_target(target_a)
        content_b = self._resolve_target(target_b)

        if content_a.startswith("未找到") or content_b.startswith("未找到"):
            return f"对比失败：\n- A: {content_a}\n- B: {content_b}"

        prompt = f"""对比以下两个文档/章节：

文档A: {content_a[:2000]}

文档B: {content_b[:2000]}

请按以下维度并排对比，用表格输出：
1. 相同点
2. 不同点
3. 关键数据对比（如有）

用中文回答，简洁清晰。"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    # ---- Phase 1 ReAct 图（原封不动）----------------------------------------

    def _build_react_graph(self):
        search_tool = tool(self._search_docs_impl)
        list_tool = tool(self._list_docs_impl)
        chunks_tool = tool(self._get_chunks_impl)
        tools = [search_tool, list_tool, chunks_tool]

        llm_with_tools = self.llm.bind_tools(tools)

        def _agent_node(state: MultiStepState) -> dict:
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        def _should_continue(state: MultiStepState) -> str:
            last = state["messages"][-1]
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

    # ---- Phase 2 编排图 -----------------------------------------------------

    def _build_orchestration_graph(self):
        builder = StateGraph(MultiStepState)

        builder.add_node("decompose", self._decompose)
        builder.add_node("research", self._research)
        builder.add_node("synthesize", self._synthesize)
        builder.add_node("reflect", self._reflect)

        builder.set_entry_point("decompose")
        builder.add_edge("decompose", "research")
        builder.add_edge("research", "synthesize")
        builder.add_conditional_edges(
            "reflect",
            self._should_continue_or_loop,
            {"research": "research", END: END},
        )
        builder.add_edge("synthesize", "reflect")

        return builder.compile()

    # ---- 节点实现 -----------------------------------------------------------

    def _decompose(self, state: MultiStepState) -> dict:
        """LLM 分析问题复杂度，输出子问题列表。解析失败时降级为单子问题。"""
        try:
            response = self.llm.invoke([
                SystemMessage(content=DECOMPOSE_PROMPT),
                HumanMessage(content=f"用户问题：{state['question']}"),
            ])
            raw = response.content.strip()
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if not json_match:
                raise ValueError("No JSON found in response")
            parsed = json.loads(json_match.group(0))
            sub_questions = parsed.get("sub_questions", [state["question"]])
            if not sub_questions or not isinstance(sub_questions, list):
                sub_questions = [state["question"]]
        except Exception as e:
            logger.warning(f"Decompose failed, falling back to single question: {e}")
            sub_questions = [state["question"]]

        return {
            "sub_questions": sub_questions,
            "current_step": 0,
            "research_results": [],
            "reflection_count": 0,
            "needs_refinement": False,
        }

    def _research(self, state: MultiStepState) -> dict:
        """遍历子问题执行 ReAct。实际流式过程在 ask_stream() 中驱动，
        此节点仅做状态校验通过。"""
        return {}

    def _synthesize(self, state: MultiStepState) -> dict:
        """LLM 综合所有子答案，生成最终回答。"""
        results_text = self._format_research_summary(state["research_results"])
        prompt = SYNTHESIZE_PROMPT.format(
            question=state["question"],
            results=results_text,
        )
        response = self.llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="请综合以上子问题分析结果，生成完整回答。"),
        ])
        return {"final_answer": response.content}

    def _reflect(self, state: MultiStepState) -> dict:
        """LLM 自检回答质量，决定是否回环补搜。"""
        prompt = REFLECT_PROMPT.format(
            question=state["question"],
            sub_questions=json.dumps(state["sub_questions"], ensure_ascii=False),
            answer=state["final_answer"],
        )
        try:
            response = self.llm.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content="请评估回答质量。"),
            ])
            raw = response.content.strip()
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if not json_match:
                raise ValueError("No JSON found in reflect response")
            parsed = json.loads(json_match.group(0))
            passed = parsed.get("pass", True)
            refinement_query = parsed.get("refinement_query", "")
            reason = parsed.get("reason", "")
        except Exception as e:
            logger.warning(f"Reflect parse failed, defaulting to pass: {e}")
            passed = True
            refinement_query = ""
            reason = ""

        if passed or state.get("reflection_count", 0) >= 2:
            return {
                "needs_refinement": False,
                "reflection_count": state.get("reflection_count", 0),
            }

        new_sub_questions = [refinement_query] if refinement_query else state.get("sub_questions", [])
        return {
            "needs_refinement": True,
            "reflection_count": state.get("reflection_count", 0) + 1,
            "sub_questions": new_sub_questions,
            "current_step": 0,
        }

    def _should_continue_or_loop(self, state: MultiStepState) -> str:
        if state.get("needs_refinement") and state.get("reflection_count", 0) < 2:
            return "research"
        return END

    # ---- 辅助方法 -----------------------------------------------------------

    def _format_history(self, messages: list[BaseMessage] | None) -> str:
        if not messages:
            return "(无历史对话)"
        lines = []
        for msg in messages[-10:]:
            role = "用户" if msg.type == "human" else "助手"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def _format_research_summary(self, results: list[dict]) -> str:
        if not results:
            return "(无检索结果)"
        lines = []
        for i, r in enumerate(results):
            answer = r.get("answer", "检索失败")
            lines.append(f"子问题{i+1}：{r['sub_q']}\n回答：{answer}\n来源：{r.get('sources', [])}")
        return "\n\n".join(lines)

    def _summarize_docs(self, docs: list) -> str:
        """文档列表摘要，用于 LLM 评估相关性。"""
        if not docs:
            return "(无文档)"
        lines = []
        for i, doc in enumerate(docs):
            filename = doc.metadata.get("filename", "unknown")
            content_preview = doc.page_content[:150].replace("\n", " ")
            lines.append(f"[{i+1}] {filename}: {content_preview}...")
        return "\n".join(lines)

    # ---- 流式入口 -----------------------------------------------------------

    async def ask_stream(
        self,
        question: str,
        session_id: str,
        chat_history_messages: list[BaseMessage] | None = None,
    ) -> AsyncIterator[str]:
        from backend.services.session_service import session_service

        self._last_search_docs_var.set([])
        self._last_search_sources_var.set([])

        history_text = self._format_history(chat_history_messages)
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

            yield f"data: {json.dumps({'type': 'decompose', 'data': f'拆解为{len(sub_questions)}个子问题'}, ensure_ascii=False)}\n\n"

            # 2. 研究循环（可能伴随反思回环）
            reflection_count = 0
            max_reflections = 2

            while True:
                research_results: list[dict] = []

                for i, sub_q in enumerate(sub_questions):
                    yield f"data: {json.dumps({'type': 'step', 'data': f'正在处理 {i+1}/{len(sub_questions)}'}, ensure_ascii=False)}\n\n"

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
                        async for event in self.react_graph.astream_events(react_state, version="v2"):
                            kind = event.get("event")

                            if kind == "on_tool_start":
                                tool_name = event.get("name", "unknown")
                                yield f"data: {json.dumps({'type': 'tool', 'data': f'调用工具: {tool_name}...'}, ensure_ascii=False)}\n\n"

                            if kind == "on_tool_end":
                                output = event.get("data", {}).get("output", "")
                                if isinstance(output, str) and output:
                                    preview = output[:200].replace("\n", " ")
                                    yield f"data: {json.dumps({'type': 'tool', 'data': f'工具返回 ({len(output)} 字符): {preview}...'}, ensure_ascii=False)}\n\n"

                            if kind == "on_chat_model_stream":
                                chunk = event["data"]["chunk"]
                                token = chunk.content if hasattr(chunk, "content") and chunk.content else None
                                if token and not getattr(chunk, "tool_calls", None):
                                    sub_answer += token
                                    yield f"data: {json.dumps({'type': 'thinking', 'data': token}, ensure_ascii=False)}\n\n"

                        sources = self._last_search_sources_var.get()
                        all_sources.extend(sources)
                        research_results.append({
                            "sub_q": sub_q,
                            "answer": sub_answer or "检索未返回结果",
                            "sources": [s.model_dump() for s in sources],
                        })
                    except Exception as e:
                        logger.error(f"Sub-question research failed: {sub_q} — {e}")
                        research_results.append({
                            "sub_q": sub_q,
                            "answer": "检索失败",
                            "sources": [],
                        })

                # 3. 合成答案
                results_text = self._format_research_summary(research_results)
                synth_prompt = SYNTHESIZE_PROMPT.format(question=question, results=results_text)
                synth_response = self.llm.invoke([
                    SystemMessage(content=synth_prompt),
                    HumanMessage(content="请综合以上子问题分析结果，生成完整回答。"),
                ])
                final_answer = synth_response.content

                yield f"data: {json.dumps({'type': 'token', 'data': final_answer}, ensure_ascii=False)}\n\n"

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
        except Exception as e:
            logger.error(f"Failed to persist session: {e}")

        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"


agent_service = MultiStepAgentService()
