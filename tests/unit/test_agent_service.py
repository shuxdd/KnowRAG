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
