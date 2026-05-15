import contextvars
import json
import logging
from typing import TypedDict, AsyncIterator

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
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


class AgentState(TypedDict):
    session_id: str
    question: str
    messages: list[BaseMessage]


class AgentService:
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
        self.graph = self._build_graph()

    # ---- Tool implementations ------------------------------------------------

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

    # ---- Graph construction --------------------------------------------------

    def _build_graph(self):
        search_tool = tool(self._search_docs_impl)
        list_tool = tool(self._list_docs_impl)
        chunks_tool = tool(self._get_chunks_impl)
        tools = [search_tool, list_tool, chunks_tool]

        llm_with_tools = self.llm.bind_tools(tools)

        def _agent_node(state: AgentState) -> dict:
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        def _should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        builder = StateGraph(AgentState)
        builder.add_node("agent", _agent_node)
        builder.add_node("tools", ToolNode(tools))
        builder.set_entry_point("agent")
        builder.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
        builder.add_edge("tools", "agent")

        return builder.compile()

    # ---- Helpers -------------------------------------------------------------

    def _format_history(self, messages: list[BaseMessage]) -> str:
        if not messages:
            return "(no history)"
        lines = []
        for msg in messages[-10:]:
            role = "User" if msg.type == "human" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    # ---- Streaming entry point -----------------------------------------------

    async def ask_stream(
        self,
        question: str,
        session_id: str,
        chat_history_messages: list[BaseMessage] | None = None,
    ) -> AsyncIterator[str]:
        from backend.services.session_service import session_service

        self._last_search_docs_var.set([])
        self._last_search_sources_var.set([])

        history_text = self._format_history(chat_history_messages or [])
        initial_state: AgentState = {
            "session_id": session_id,
            "question": question,
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Chat history:\n{history_text}\n\nUser question: {question}"
                ),
            ],
        }

        full_answer = ""

        try:
            async for event in self.graph.astream_events(initial_state, version="v2"):
                kind = event.get("event")

                if kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    yield f"data: {json.dumps({'type': 'tool', 'data': f'调用工具: {tool_name}...'}, ensure_ascii=False)}\n\n"

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    token = chunk.content if hasattr(chunk, "content") and chunk.content else None
                    if token and not getattr(chunk, "tool_calls", None):
                        full_answer += token
                        yield f"data: {json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"Agent stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': f'Agent 错误: {str(e)}'}, ensure_ascii=False)}\n\n"

        # Push sources after streaming
        sources = self._last_search_sources_var.get()
        if sources:
            yield f"data: {json.dumps({'type': 'sources', 'data': [s.model_dump() for s in sources]}, ensure_ascii=False)}\n\n"

        # Persist conversation
        try:
            session_service.add_message(session_id, "user", question)
            session_service.add_message(
                session_id, "assistant",
                full_answer or "处理时出错",
                [s.model_dump() for s in sources],
            )
            session = session_service.get_session(session_id)
            if session and session.get("title") == "新对话":
                title = question[:30] + ("..." if len(question) > 30 else "")
                session_service.update_title(session_id, title)
        except Exception as e:
            logger.error(f"Failed to persist session: {e}")

        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"


agent_service = AgentService()
