import pytest
from unittest.mock import patch, MagicMock, ANY
import json


class TestMultiStepAgentServiceInit:
    @patch("backend.services.agent_service.ChatOpenAI")
    def test_create_react_graph_returns_compiled_graph(self, mock_llm):
        """_create_react_graph 接受工具列表并返回编译后的 LangGraph。"""
        mock_llm.return_value = MagicMock()
        from backend.services.agent_service import MultiStepAgentService

        svc = MultiStepAgentService()

        async def fake_tool():
            pass

        tools = [MagicMock(), MagicMock()]
        graph = svc._create_react_graph(tools)
        nodes = graph.get_graph().nodes
        assert "agent" in nodes
        assert "tools" in nodes

    @patch("backend.services.agent_service.ChatOpenAI")
    def test_multistep_state_fields(self, mock_llm):
        """MultiStepState 包含所有必需字段。"""
        mock_llm.return_value = MagicMock()
        from backend.services.agent_service import MultiStepState

        annotations = MultiStepState.__annotations__
        assert "session_id" in annotations
        assert "question" in annotations
        assert "chat_history" in annotations
        assert "messages" in annotations
        assert "sub_questions" in annotations
        assert "current_step" in annotations
        assert "research_results" in annotations
        assert "final_answer" in annotations
        assert "reflection_count" in annotations
        assert "needs_refinement" in annotations


class TestModuleSingleton:
    @patch("backend.services.agent_service.ChatOpenAI")
    def test_agent_service_instance_exists(self, mock_llm):
        """模块级 agent_service 是 MultiStepAgentService 实例。"""
        mock_llm.return_value = MagicMock()
        from backend.services.agent_service import agent_service, MultiStepAgentService

        assert isinstance(agent_service, MultiStepAgentService)


class TestReadSection:
    @patch("backend.services.agent_service.parent_store")
    @patch("backend.services.agent_service.ChatOpenAI")
    def test_read_section_found(self, mock_llm, mock_parent_store):
        """精确匹配章节路径时返回完整内容和页码。"""
        mock_llm.return_value = MagicMock()
        from backend.services.agent_service import MultiStepAgentService
        from backend.models.chunk_types import ParentChunk

        mock_parent_store.get_by_filename.return_value = [
            ParentChunk(
                id="p1", content="年假每年10天，工龄满5年后每年增加2天。",
                filename="员工手册.pdf", heading_path=["休假政策", "年假"],
                page_start=12, page_end=15,
            ),
            ParentChunk(
                id="p2", content="病假每年5天，需提供医院证明。",
                filename="员工手册.pdf", heading_path=["休假政策", "病假"],
                page_start=16, page_end=17,
            ),
        ]

        svc = MultiStepAgentService()
        result = svc._read_section_impl("员工手册.pdf", ["休假政策", "年假"], "", 0)
        assert "员工手册.pdf" in result
        assert "休假政策 > 年假" in result
        assert "页码 12-15" in result
        assert "年假每年10天" in result

    @patch("backend.services.agent_service.parent_store")
    @patch("backend.services.agent_service.ChatOpenAI")
    def test_read_section_not_found(self, mock_llm, mock_parent_store):
        """文档不存在时返回错误信息。"""
        mock_llm.return_value = MagicMock()
        from backend.services.agent_service import MultiStepAgentService

        mock_parent_store.get_by_filename.return_value = []

        svc = MultiStepAgentService()
        result = svc._read_section_impl("不存在.pdf", ["某章节"], "", 0)
        assert "未找到文档" in result
        assert "不存在.pdf" in result

    @patch("backend.services.agent_service.parent_store")
    @patch("backend.services.agent_service.ChatOpenAI")
    def test_read_section_partial_match(self, mock_llm, mock_parent_store):
        """无精确匹配时列出包含关键词的相近章节。"""
        mock_llm.return_value = MagicMock()
        from backend.services.agent_service import MultiStepAgentService
        from backend.models.chunk_types import ParentChunk

        mock_parent_store.get_by_filename.return_value = [
            ParentChunk(
                id="p1", content="年假内容...", filename="手册.pdf",
                heading_path=["休假政策", "年假"], page_start=1, page_end=2,
            ),
            ParentChunk(
                id="p2", content="病假内容...", filename="手册.pdf",
                heading_path=["休假政策", "病假"], page_start=3, page_end=4,
            ),
        ]

        svc = MultiStepAgentService()
        result = svc._read_section_impl("手册.pdf", ["休假政策", "事假"], "", 0)
        assert "未精确匹配" in result
        assert "相近章节" in result
        assert "年假" in result
        assert "病假" in result

    @patch("backend.services.agent_service.parent_store")
    @patch("backend.services.agent_service.ChatOpenAI")
    def test_read_section_no_match(self, mock_llm, mock_parent_store):
        """父块存在但无任何 heading 匹配时返回未找到提示。"""
        mock_llm.return_value = MagicMock()
        from backend.services.agent_service import MultiStepAgentService
        from backend.models.chunk_types import ParentChunk

        mock_parent_store.get_by_filename.return_value = [
            ParentChunk(
                id="p1", content="内容...", filename="手册.pdf",
                heading_path=["其他章节"], page_start=1, page_end=2,
            ),
        ]

        svc = MultiStepAgentService()
        result = svc._read_section_impl("手册.pdf", ["不存在的章节"], "", 0)
        assert "未找到" in result
        assert "相关的章节" in result


class TestSynthesizeStreaming:
    """测试 ask_stream() synthesize 阶段的流式输出行为。"""

    @patch("backend.services.agent_service.ChatOpenAI")
    def test_synthesize_streams_tokens(self, mock_llm_class):
        """synthesize 阶段应通过 astream 逐 token 推送 SSE 事件。"""
        import asyncio

        mock_llm_instance = MagicMock()

        # decompose 响应：简单问题，单子问题
        dec_response = MagicMock()
        dec_response.content = '{"complexity": "simple", "sub_questions": ["测试"]}'
        # reflect 响应：通过
        ref_response = MagicMock()
        ref_response.content = '{"pass": true, "reason": "ok"}'
        mock_llm_instance.invoke.side_effect = [dec_response, ref_response]

        # astream 返回逐 token
        async def mock_astream(messages):
            for t in ["这", "是", "合成", "答案"]:
                chunk = MagicMock()
                chunk.content = t
                yield chunk

        mock_llm_instance.astream.side_effect = mock_astream
        mock_llm_class.return_value = mock_llm_instance

        from backend.services.agent_service import MultiStepAgentService
        svc = MultiStepAgentService()

        # mock graph: 无子问题研究事件（空流）
        mock_graph = MagicMock()
        async def empty_stream(state, version=None):
            if False:
                yield
        mock_graph.astream_events = empty_stream
        svc._create_react_graph = MagicMock(return_value=mock_graph)

        async def collect():
            events = []
            async for e in svc.ask_stream("测试问题", "s1"):
                events.append(e)
            return events

        events = asyncio.run(collect())

        # 验证 astream 被调用（替代了 invoke 用于 synthesize）
        mock_llm_instance.astream.assert_called_once()

        # 验证 invoke 仅被调用 2 次（decompose + reflect，不含 synthesize）
        assert mock_llm_instance.invoke.call_count == 2

        # 提取 token 事件
        token_events = [e for e in events if '"type": "token"' in e]
        assert len(token_events) >= 1
        all_tokens = ""
        for te in token_events:
            data = json.loads(te.split("data: ")[1])
            all_tokens += data["data"]
        assert "这是合成答案" == all_tokens

        # 确认 done 事件存在
        assert any('"type": "done"' in e for e in events)

    @patch("backend.services.session_service.session_service")
    @patch("backend.services.agent_service.ChatOpenAI")
    def test_synthesize_final_answer_used_in_reflect(self, mock_llm_class, mock_session):
        """streaming 后 final_answer 被正确用于 reflect 和持久化。"""
        import asyncio

        mock_llm_instance = MagicMock()

        dec_response = MagicMock()
        dec_response.content = '{"complexity": "simple", "sub_questions": ["测试"]}'
        ref_response = MagicMock()
        ref_response.content = '{"pass": true, "reason": "ok"}'
        mock_llm_instance.invoke.side_effect = [dec_response, ref_response]

        async def mock_astream(messages):
            for t in ["合成", "结果"]:
                chunk = MagicMock()
                chunk.content = t
                yield chunk

        mock_llm_instance.astream.side_effect = mock_astream
        mock_llm_class.return_value = mock_llm_instance

        from backend.services.agent_service import MultiStepAgentService
        svc = MultiStepAgentService()

        mock_graph = MagicMock()
        async def empty_stream(state, version=None):
            if False:
                yield
        mock_graph.astream_events = empty_stream
        svc._create_react_graph = MagicMock(return_value=mock_graph)

        async def collect():
            events = []
            async for e in svc.ask_stream("测试问题", "s1"):
                events.append(e)
            return events

        events = asyncio.run(collect())

        token_events = [e for e in events if '"type": "token"' in e]
        all_tokens = ""
        for te in token_events:
            data = json.loads(te.split("data: ")[1])
            all_tokens += data["data"]
        assert all_tokens == "合成结果"

        # 验证 final_answer 出现在某个 invoke 调用中（reflect 使用了它）
        any_has_answer = any(
            "合成结果" in str(msg.content)
            for call in mock_llm_instance.invoke.call_args_list
            for msg in call[0][0]
            if hasattr(msg, "content")
        )
        assert any_has_answer, "final_answer should appear in reflect prompt"

        # 验证 session_service.add_message 被调用，包含 final_answer
        mock_session.add_message.assert_any_call(
            "s1", "assistant", "合成结果", ANY
        )

        assert any('"type": "done"' in e for e in events)

    @patch("backend.services.agent_service.ChatOpenAI")
    def test_synthesize_empty_response(self, mock_llm_class):
        """astream 无 token 输出时流程不崩溃，正常结束。"""
        import asyncio

        mock_llm_instance = MagicMock()

        dec_response = MagicMock()
        dec_response.content = '{"complexity": "simple", "sub_questions": ["测试"]}'
        ref_response = MagicMock()
        ref_response.content = '{"pass": true, "reason": "ok"}'
        mock_llm_instance.invoke.side_effect = [dec_response, ref_response]

        async def empty_astream(messages):
            if False:
                yield
        mock_llm_instance.astream.side_effect = empty_astream
        mock_llm_class.return_value = mock_llm_instance

        from backend.services.agent_service import MultiStepAgentService
        svc = MultiStepAgentService()

        mock_graph = MagicMock()
        async def empty_stream(state, version=None):
            if False:
                yield
        mock_graph.astream_events = empty_stream
        svc._create_react_graph = MagicMock(return_value=mock_graph)

        async def collect():
            events = []
            async for e in svc.ask_stream("测试问题", "s1"):
                events.append(e)
            return events

        events = asyncio.run(collect())

        assert any('"type": "done"' in e for e in events)
        assert any('"type": "reflect"' in e for e in events)

    @patch("backend.services.agent_service.ChatOpenAI")
    def test_synthesize_chunk_without_content(self, mock_llm_class):
        """astream chunk 无 content 属性时不追加 token。"""
        import asyncio

        mock_llm_instance = MagicMock()

        dec_response = MagicMock()
        dec_response.content = '{"complexity": "simple", "sub_questions": ["测试"]}'
        ref_response = MagicMock()
        ref_response.content = '{"pass": true, "reason": "ok"}'
        mock_llm_instance.invoke.side_effect = [dec_response, ref_response]

        async def mock_astream(messages):
            c1 = MagicMock(spec=[])  # no 'content' attr
            yield c1
            c2 = MagicMock()
            c2.content = ""
            yield c2
            c3 = MagicMock()
            c3.content = "有效"
            yield c3

        mock_llm_instance.astream.side_effect = mock_astream
        mock_llm_class.return_value = mock_llm_instance

        from backend.services.agent_service import MultiStepAgentService
        svc = MultiStepAgentService()

        mock_graph = MagicMock()
        async def empty_stream(state, version=None):
            if False:
                yield
        mock_graph.astream_events = empty_stream
        svc._create_react_graph = MagicMock(return_value=mock_graph)

        async def collect():
            events = []
            async for e in svc.ask_stream("测试问题", "s1"):
                events.append(e)
            return events

        events = asyncio.run(collect())

        token_events = [e for e in events if '"type": "token"' in e]
        assert len(token_events) == 1
        assert '"data": "有效"' in token_events[0]
        assert any('"type": "done"' in e for e in events)


class TestParallelResearch:
    """测试 _research_one_sub_q 和 ask_stream() 并行研究。"""

    @patch("backend.services.agent_service.ChatOpenAI")
    def test_research_one_sub_q_yields_events(self, mock_llm_class):
        """_research_one_sub_q 应 yield tool 和 thinking 事件，最后是 __result__。"""
        import asyncio

        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        from backend.services.agent_service import MultiStepAgentService
        svc = MultiStepAgentService()

        mock_graph = MagicMock()
        async def mock_events(state, version=None):
            yield {"event": "on_tool_start", "name": "search_docs"}
            tool_end = {"event": "on_tool_end", "data": {"output": "检索结果内容"}}
            yield tool_end
            chunk_data = {"event": "on_chat_model_stream", "data": {"chunk": MagicMock()}}
            chunk_data["data"]["chunk"].content = "思考token"
            chunk_data["data"]["chunk"].tool_calls = None
            yield chunk_data
        mock_graph.astream_events = mock_events
        svc._create_react_graph = MagicMock(return_value=mock_graph)

        async def collect():
            events = []
            async for e in svc._research_one_sub_q("测试子问题", 0, "s1", "", 0):
                events.append(e)
            return events

        events = asyncio.run(collect())

        types = [e[0] for e in events]
        assert "tool" in types
        assert "thinking" in types
        assert "__result__" in types
        assert types[-1] == "__result__"
        result = events[-1]
        assert result[1] == 0
        assert result[2]["sub_q"] == "测试子问题"
        assert "answer" in result[2]

    @patch("backend.services.agent_service.ChatOpenAI")
    def test_research_one_sub_q_handles_exception(self, mock_llm_class):
        """_research_one_sub_q 内部异常时应返回检索失败标记。"""
        import asyncio

        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        from backend.services.agent_service import MultiStepAgentService
        svc = MultiStepAgentService()

        mock_graph = MagicMock()
        async def crashing_events(state, version=None):
            raise RuntimeError("模拟错误")
            yield
        mock_graph.astream_events = crashing_events
        svc._create_react_graph = MagicMock(return_value=mock_graph)

        async def collect():
            events = []
            async for e in svc._research_one_sub_q("失败子问题", 0, "s1", "", 0):
                events.append(e)
            return events

        events = asyncio.run(collect())

        assert len(events) == 1
        assert events[0][0] == "__result__"
        assert events[0][2]["answer"] == "检索失败"

    @patch("backend.services.agent_service.ChatOpenAI")
    def test_research_one_sub_q_event_index_correct(self, mock_llm_class):
        """_research_one_sub_q yield 的事件应携带正确的 index。"""
        import asyncio

        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        from backend.services.agent_service import MultiStepAgentService
        svc = MultiStepAgentService()

        mock_graph = MagicMock()
        async def mock_events(state, version=None):
            yield {"event": "on_tool_start", "name": "search_docs"}
        mock_graph.astream_events = mock_events
        svc._create_react_graph = MagicMock(return_value=mock_graph)

        async def collect():
            events = []
            async for e in svc._research_one_sub_q("子问题2", 1, "s1", "", 0):
                events.append(e)
            return events

        events = asyncio.run(collect())

        tool_event = events[0]
        assert tool_event[1] == 1
        result_event = events[-1]
        assert result_event[1] == 1

    @patch("backend.services.agent_service.ChatOpenAI")
    def test_parallel_research_all_results_collected(self, mock_llm_class):
        """并行研究完成后 research_results 应包含所有子问题结果。"""
        import asyncio

        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        from backend.services.agent_service import MultiStepAgentService
        svc = MultiStepAgentService()

        async def quick_result(sub_q, index, session_id, history_text, user_id):
            yield ("__result__", index, {"sub_q": sub_q, "answer": f"答案{index}", "sources": []})

        svc._research_one_sub_q = quick_result

        queue: asyncio.Queue = asyncio.Queue()
        pending = 3
        sub_questions = ["q0", "q1", "q2"]
        results = [None] * 3

        async def producer(sub_q, idx):
            try:
                async for event in svc._research_one_sub_q(sub_q, idx, "s1", "", 0):
                    await queue.put(event)
            except Exception as e:
                await queue.put(("__error__", idx, str(e)))

        async def run():
            producers = [asyncio.create_task(producer(q, i)) for i, q in enumerate(sub_questions)]

            collect_pending = pending
            while collect_pending > 0:
                event = await queue.get()
                event_type, idx, data = event
                if event_type == "__result__":
                    results[idx] = data
                    collect_pending -= 1
                elif event_type == "__error__":
                    results[idx] = {"sub_q": sub_questions[idx], "answer": "检索失败", "sources": []}
                    collect_pending -= 1

            await asyncio.gather(*producers)
            return results

        results = asyncio.run(run())

        assert None not in results
        assert results[0]["sub_q"] == "q0"
        assert results[1]["sub_q"] == "q1"
        assert results[2]["sub_q"] == "q2"
        assert results[0]["answer"] == "答案0"
        assert results[1]["answer"] == "答案1"
        assert results[2]["answer"] == "答案2"
