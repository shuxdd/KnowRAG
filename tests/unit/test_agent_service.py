import pytest
from unittest.mock import patch, MagicMock


class TestMultiStepAgentServiceInit:
    @patch("backend.services.agent_service.ChatOpenAI")
    def test_both_graphs_compiled(self, mock_llm):
        """MultiStepAgentService 编译 react_graph 和 orchestration_graph。"""
        mock_llm.return_value = MagicMock()
        from backend.services.agent_service import MultiStepAgentService

        svc = MultiStepAgentService()
        assert svc.react_graph is not None
        assert svc.orchestration_graph is not None

        react_nodes = svc.react_graph.get_graph().nodes
        assert "agent" in react_nodes
        assert "tools" in react_nodes

        orch_nodes = svc.orchestration_graph.get_graph().nodes
        assert "decompose" in orch_nodes
        assert "research" in orch_nodes
        assert "synthesize" in orch_nodes
        assert "reflect" in orch_nodes

    @patch("backend.services.agent_service.ChatOpenAI")
    def test_multistep_state_fields(self, mock_llm):
        """MultiStepState 包含所有 Phase 2 字段。"""
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


class TestDecompose:
    @patch("backend.services.agent_service.ChatOpenAI")
    def test_simple_question_returns_single_sub(self, mock_llm):
        """简单问候被拆解为单个子问题。"""
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"complexity": "simple", "sub_questions": ["你好"]}'
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        from backend.services.agent_service import MultiStepAgentService

        svc = MultiStepAgentService()
        state = {
            "session_id": "s1",
            "question": "你好",
            "chat_history": "",
            "messages": [],
            "sub_questions": [],
            "current_step": 0,
            "research_results": [],
            "final_answer": "",
            "reflection_count": 0,
            "needs_refinement": False,
        }
        result = svc._decompose(state)
        assert result["sub_questions"] == ["你好"]
        assert result["current_step"] == 0

    @patch("backend.services.agent_service.ChatOpenAI")
    def test_complex_question_splits_multiple(self, mock_llm):
        """复杂对比问题被拆解为多个子问题。"""
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"complexity": "complex", "sub_questions": ["子问题A", "子问题B", "子问题C"]}'
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        from backend.services.agent_service import MultiStepAgentService

        svc = MultiStepAgentService()
        state = {
            "session_id": "s1",
            "question": "对比产品A和产品B的性能差异",
            "chat_history": "",
            "messages": [],
            "sub_questions": [],
            "current_step": 0,
            "research_results": [],
            "final_answer": "",
            "reflection_count": 0,
            "needs_refinement": False,
        }
        result = svc._decompose(state)
        assert len(result["sub_questions"]) == 3

    @patch("backend.services.agent_service.ChatOpenAI")
    def test_malformed_json_falls_back(self, mock_llm):
        """JSON 解析失败时降级为单个子问题。"""
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "这不是JSON格式"
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        from backend.services.agent_service import MultiStepAgentService

        svc = MultiStepAgentService()
        state = {
            "session_id": "s1",
            "question": "复杂问题",
            "chat_history": "",
            "messages": [],
            "sub_questions": [],
            "current_step": 0,
            "research_results": [],
            "final_answer": "",
            "reflection_count": 0,
            "needs_refinement": False,
        }
        result = svc._decompose(state)
        assert result["sub_questions"] == ["复杂问题"]


class TestReflect:
    @patch("backend.services.agent_service.ChatOpenAI")
    def test_reflect_pass_ends_loop(self, mock_llm):
        """LLM 通过质量检查时 needs_refinement=False。"""
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"pass": true, "reason": "回答完整准确"}'
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        from backend.services.agent_service import MultiStepAgentService

        svc = MultiStepAgentService()
        state = {
            "session_id": "s1",
            "question": "测试",
            "chat_history": "",
            "messages": [],
            "sub_questions": ["测试"],
            "current_step": 0,
            "research_results": [],
            "final_answer": "完整答案",
            "reflection_count": 0,
            "needs_refinement": False,
        }
        result = svc._reflect(state)
        assert result["needs_refinement"] is False

    @patch("backend.services.agent_service.ChatOpenAI")
    def test_reflect_max_count_forces_end(self, mock_llm):
        """达到最大反思次数时即使 LLM 判定不通过也强制结束。"""
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"pass": false, "refinement_query": "再查一次", "reason": "缺少信息"}'
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        from backend.services.agent_service import MultiStepAgentService

        svc = MultiStepAgentService()
        state = {
            "session_id": "s1",
            "question": "测试",
            "chat_history": "",
            "messages": [],
            "sub_questions": ["测试"],
            "current_step": 0,
            "research_results": [],
            "final_answer": "不完整答案",
            "reflection_count": 2,
            "needs_refinement": False,
        }
        result = svc._reflect(state)
        assert result["needs_refinement"] is False


class TestModuleSingleton:
    @patch("backend.services.agent_service.ChatOpenAI")
    def test_agent_service_instance_exists(self, mock_llm):
        """模块级 agent_service 是 MultiStepAgentService 实例。"""
        mock_llm.return_value = MagicMock()
        from backend.services.agent_service import agent_service, MultiStepAgentService

        assert isinstance(agent_service, MultiStepAgentService)


class TestSummarizeDocs:
    @patch("backend.services.agent_service.ChatOpenAI")
    def test_summarize_docs_normal(self, mock_llm):
        """多个文档时返回带编号的摘要列表。"""
        mock_llm.return_value = MagicMock()
        from backend.services.agent_service import MultiStepAgentService

        svc = MultiStepAgentService()
        from langchain_core.documents import Document
        docs = [
            Document(page_content="这是第一段测试内容包含足够多的文字来展示摘要功能", metadata={"filename": "a.pdf", "score": 0.9}),
            Document(page_content="第二段内容不同用于验证摘要截断", metadata={"filename": "b.pdf", "score": 0.7}),
        ]
        result = svc._summarize_docs(docs)
        assert "[1]" in result
        assert "a.pdf" in result
        assert "[2]" in result
        assert "b.pdf" in result

    @patch("backend.services.agent_service.ChatOpenAI")
    def test_summarize_docs_empty(self, mock_llm):
        """空列表返回无文档提示。"""
        mock_llm.return_value = MagicMock()
        from backend.services.agent_service import MultiStepAgentService

        svc = MultiStepAgentService()
        result = svc._summarize_docs([])
        assert "无文档" in result


class TestResolveTarget:
    @patch("backend.services.agent_service.parent_store")
    @patch("backend.services.agent_service.ChatOpenAI")
    def test_resolve_by_doc_id(self, mock_llm, mock_parent_store):
        """通过 doc_id 解析目标内容。"""
        mock_llm.return_value = MagicMock()
        from backend.services.agent_service import MultiStepAgentService
        from backend.models.chunk_types import ParentChunk

        mock_parent_store.get_by_ids.return_value = [
            ParentChunk(
                id="abc-123", content="测试内容。", filename="test.pdf",
                heading_path=["章一"], page_start=1, page_end=2,
            ),
        ]

        svc = MultiStepAgentService()
        result = svc._resolve_target({"doc_id": "abc-123"})
        assert "test.pdf" in result
        assert "测试内容" in result

    @patch("backend.services.agent_service.parent_store")
    @patch("backend.services.agent_service.ChatOpenAI")
    def test_resolve_by_doc_and_heading(self, mock_llm, mock_parent_store):
        """通过 doc + heading 解析，委托给 _read_section_impl。"""
        mock_llm.return_value = MagicMock()
        from backend.services.agent_service import MultiStepAgentService
        from backend.models.chunk_types import ParentChunk

        mock_parent_store.get_by_filename.return_value = [
            ParentChunk(
                id="p1", content="章节内容。", filename="doc.pdf",
                heading_path=["A", "B"], page_start=1, page_end=2,
            ),
        ]

        svc = MultiStepAgentService()
        result = svc._resolve_target({"doc": "doc.pdf", "heading": ["A", "B"]})
        assert "doc.pdf" in result
        assert "章节内容" in result

    @patch("backend.services.agent_service.parent_store")
    @patch("backend.services.agent_service.ChatOpenAI")
    def test_resolve_by_doc_only(self, mock_llm, mock_parent_store):
        """仅 doc 无 heading 时返回整个文档所有父块。"""
        mock_llm.return_value = MagicMock()
        from backend.services.agent_service import MultiStepAgentService
        from backend.models.chunk_types import ParentChunk

        mock_parent_store.get_by_filename.return_value = [
            ParentChunk(
                id="p1", content="第一节。", filename="doc.pdf",
                heading_path=["一"], page_start=1, page_end=1,
            ),
            ParentChunk(
                id="p2", content="第二节。", filename="doc.pdf",
                heading_path=["二"], page_start=2, page_end=2,
            ),
        ]

        svc = MultiStepAgentService()
        result = svc._resolve_target({"doc": "doc.pdf"})
        assert "doc.pdf" in result
        assert "完整内容" in result
        assert "第一节" in result
        assert "第二节" in result


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
        result = svc._read_section_impl("员工手册.pdf", ["休假政策", "年假"])
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
        result = svc._read_section_impl("不存在.pdf", ["某章节"])
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
        result = svc._read_section_impl("手册.pdf", ["休假政策", "事假"])
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
        result = svc._read_section_impl("手册.pdf", ["不存在的章节"])
        assert "未找到" in result
        assert "相关的章节" in result


class TestCompareDocs:
    @patch("backend.services.agent_service.parent_store")
    @patch("backend.services.agent_service.ChatOpenAI")
    def test_compare_docs_both_found(self, mock_llm, mock_parent_store):
        """两个目标都解析成功时 LLM 返回对比结果。"""
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "| 维度 | A | B |\n| 相同点 | x | x |\n| 不同点 | y | z |"
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        from backend.services.agent_service import MultiStepAgentService
        from backend.models.chunk_types import ParentChunk

        mock_parent_store.get_by_ids.side_effect = [
            [ParentChunk(
                id="a1", content="A公司年假10天。", filename="A.pdf",
                heading_path=["政策"], page_start=1, page_end=1,
            )],
            [ParentChunk(
                id="b1", content="B公司年假15天。", filename="B.pdf",
                heading_path=["政策"], page_start=1, page_end=1,
            )],
        ]

        svc = MultiStepAgentService()
        result = svc._compare_docs_impl(
            {"doc_id": "a1"},
            {"doc_id": "b1"},
        )
        assert "相同点" in result
        assert "不同点" in result

    @patch("backend.services.agent_service.parent_store")
    @patch("backend.services.agent_service.ChatOpenAI")
    def test_compare_docs_one_missing(self, mock_llm, mock_parent_store):
        """一个目标解析失败时返回错误信息。"""
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        from backend.services.agent_service import MultiStepAgentService
        from backend.models.chunk_types import ParentChunk

        mock_parent_store.get_by_ids.side_effect = [
            [ParentChunk(
                id="a1", content="内容A", filename="A.pdf",
                heading_path=["政策"], page_start=1, page_end=1,
            )],
            [],  # B not found
        ]

        svc = MultiStepAgentService()
        result = svc._compare_docs_impl(
            {"doc_id": "a1"},
            {"doc_id": "missing"},
        )
        assert "对比失败" in result
        # B 解析失败，should not call LLM
        mock_llm_instance.invoke.assert_not_called()
