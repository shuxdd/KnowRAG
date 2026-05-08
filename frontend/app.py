import streamlit as st
import httpx
import uuid

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="KnowRAG 知识库", page_icon="📚", layout="wide")
st.title("KnowRAG 企业知识库")

# ── Session state init ──
if "qa_session_id" not in st.session_state:
    st.session_state.qa_session_id = uuid.uuid4().hex[:12]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tab1, tab2 = st.tabs(["文档管理", "知识问答"])

# ── Tab 1: Document Management ──
with tab1:
    st.header("上传文档")

    uploaded_files = st.file_uploader(
        "选择文件（支持 PDF / Word / Markdown / TXT）",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for f in uploaded_files:
            if st.button(f"上传：{f.name}", key=f"upload_{f.name}"):
                with st.spinner(f"正在处理 {f.name}..."):
                    try:
                        resp = httpx.post(
                            f"{BACKEND_URL}/api/documents/upload",
                            files={"file": (f.name, f.getvalue())},
                            timeout=60,
                        )
                        if resp.status_code == 200:
                            st.success(f"{f.name} 上传成功")
                        else:
                            st.error(f"上传失败: {resp.json().get('detail', resp.text)}")
                    except httpx.ConnectError:
                        st.error("无法连接后端，请确认 FastAPI 服务已启动")

    st.divider()
    st.header("已入库文档")

    if st.button("刷新列表"):
        st.rerun()

    try:
        resp = httpx.get(f"{BACKEND_URL}/api/documents", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            docs = data.get("documents", [])
            if not docs:
                st.info("暂无文档，请上传文件")
            else:
                for doc in docs:
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    with col1:
                        st.text(doc["filename"])
                    with col2:
                        st.text(f"{doc['chunk_count']} 块")
                    with col3:
                        st.text(doc["file_type"])
                    with col4:
                        if st.button("删除", key=f"del_{doc['doc_id']}"):
                            del_resp = httpx.delete(
                                f"{BACKEND_URL}/api/documents/{doc['doc_id']}",
                                timeout=10,
                            )
                            if del_resp.status_code == 200:
                                st.success("已删除")
                                st.rerun()
                            else:
                                st.error("删除失败")
        else:
            st.warning(f"获取文档列表失败: {resp.status_code}")
    except httpx.ConnectError:
        st.warning("无法连接后端，请确认 FastAPI 服务已启动（http://localhost:8000）")

# ── Tab 2: Q&A ──
with tab2:
    st.header("知识问答")

    # Session controls
    col_sess, col_clear, col_new = st.columns([3, 1, 1])
    with col_sess:
        new_id = st.text_input("会话 ID", value=st.session_state.qa_session_id, key="session_id_input")
        if new_id != st.session_state.qa_session_id:
            st.session_state.qa_session_id = new_id
            st.session_state.chat_history = []
            # load history from backend
            try:
                resp = httpx.get(f"{BACKEND_URL}/api/qa/history/{new_id}", timeout=5)
                if resp.status_code == 200:
                    st.session_state.chat_history = resp.json().get("messages", [])
            except Exception:
                pass

    with col_clear:
        if st.button("清空历史"):
            try:
                httpx.delete(f"{BACKEND_URL}/api/qa/history/{st.session_state.qa_session_id}", timeout=5)
            except Exception:
                pass
            st.session_state.chat_history = []
            st.rerun()

    with col_new:
        if st.button("新会话"):
            st.session_state.qa_session_id = uuid.uuid4().hex[:12]
            st.session_state.chat_history = []
            st.rerun()

    st.divider()

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "human":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    # Top-K slider in sidebar (must precede any read of top_k_slider)
    with st.sidebar:
        st.subheader("问答设置")
        top_k = st.slider(
            "检索结果数",
            min_value=1,
            max_value=10,
            value=4,
            key="top_k_slider",
        )

    # Question input
    question = st.chat_input("请输入您的问题", key="qa_input")

    if question:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "human", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    resp = httpx.post(
                        f"{BACKEND_URL}/api/qa/ask",
                        json={
                            "question": question,
                            "top_k": top_k,
                            "session_id": st.session_state.qa_session_id,
                        },
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.markdown(data["answer"])

                        st.session_state.chat_history.append(
                            {"role": "ai", "content": data["answer"]}
                        )

                        with st.expander(f"参考来源（{len(data['sources'])} 个）"):
                            for i, src in enumerate(data["sources"], 1):
                                st.markdown(f"**{i}. {src['filename']}**")
                                st.text(src["content_snippet"])
                    else:
                        st.error(f"查询失败: {resp.json().get('detail', resp.text)}")
                except httpx.ConnectError:
                    st.error("无法连接后端，请确认 FastAPI 服务已启动")
