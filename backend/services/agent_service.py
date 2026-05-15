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

SYSTEM_PROMPT = """You are an enterprise knowledge base assistant. You have access to tools:

- search_docs(query, strategy, top_k): Search the knowledge base for relevant document content.
  Use this for factual questions about the knowledge base.
  strategy: "fast" (quick), "precise" (hybrid), "deep" (thorough), "auto" (automatic, recommended).
  top_k: number of results (1-20). Use 5 by default. More if the question is broad.

- list_docs(): List all documents currently in the knowledge base.
  Use this when the user asks what documents are available or what they can ask about.

- get_chunks(doc_id): View how a specific document is split into chunks.
  Use this when the user asks about document structure or chunking.

Rules:
- Greetings and casual chat: respond directly without tools.
- For factual questions, ALWAYS use search_docs first.
- If no relevant documents are found, honestly tell the user.
- Cite document sources (filenames) in your answer when using search results.
- Answer in the same language as the user's question.
- Do NOT make up information not found in the search results."""


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
        """Search the knowledge base for relevant document content."""
        docs = qa_service.search(query, strategy, top_k)
        self._last_search_docs_var.set(docs)
        if not docs:
            return "No relevant documents found in the knowledge base."

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
        """List all documents currently in the knowledge base."""
        stats = vector_service.get_document_stats()
        if not stats:
            return "No documents found in the knowledge base."
        lines = ["Documents in knowledge base:"]
        for s in stats:
            lines.append(f"  - {s['filename']} ({s.get('chunks_count', 0)} chunks)")
        return "\n".join(lines)

    def _get_chunks_impl(self, doc_id: str) -> str:
        """View how a specific document is split into chunks."""
        try:
            parents = parent_store.get_by_filename(doc_id)
            if not parents:
                return f"Document not found: {doc_id}"
        except Exception as e:
            return f"Error looking up document {doc_id}: {e}"

        lines = [f"Chunk preview for `{doc_id}`:"]
        for p in parents:
            heading = "/".join(p.heading_path)
            lines.append(
                f"  [{p.id[:8]}...] {heading} "
                f"(chars={len(p.content)}, pages={p.page_start}-{p.page_end})"
            )
            try:
                leaf_results = vector_service.collection.get(where={"parent_id": p.id})
                leaf_count = len(leaf_results.get("ids", []))
                preserved = sum(
                    1 for m in (leaf_results.get("metadatas") or [])
                    if m and m.get("preserve")
                )
                lines.append(f"    {leaf_count} leaves ({preserved} preserved)")
            except Exception:
                lines.append("    (leaf info unavailable)")
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
                    yield f"data: {json.dumps({'type': 'tool', 'data': f'Using tool: {tool_name}...'}, ensure_ascii=False)}\n\n"

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    token = chunk.content if hasattr(chunk, "content") and chunk.content else None
                    if token and not getattr(chunk, "tool_calls", None):
                        full_answer += token
                        yield f"data: {json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"Agent stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': f'Agent error: {str(e)}'}, ensure_ascii=False)}\n\n"

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
