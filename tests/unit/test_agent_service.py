import pytest
from unittest.mock import patch, MagicMock


class TestAgentServiceInit:
    @patch("backend.services.agent_service.ChatOpenAI")
    def test_graph_is_compiled(self, mock_llm):
        """AgentService compiles a graph with agent + tools nodes."""
        mock_llm.return_value = MagicMock()
        from backend.services.agent_service import AgentService

        svc = AgentService()
        assert svc.graph is not None
        nodes = svc.graph.get_graph().nodes
        assert "agent" in nodes
        assert "tools" in nodes

    @patch("backend.services.agent_service.ChatOpenAI")
    def test_agent_state_fields(self, mock_llm):
        """AgentState includes messages, session_id, final_answer."""
        mock_llm.return_value = MagicMock()
        from backend.services.agent_service import AgentService, AgentState

        svc = AgentService()
        schema = svc.graph.get_input_schema()
        fields = schema.model_fields
        # LangGraph wraps TypedDict state under a 'root' field in v2
        if "root" in fields:
            state_annotation = fields["root"].annotation
            assert state_annotation == AgentState
        assert "messages" in AgentState.__annotations__
        assert "session_id" in AgentState.__annotations__
        assert "final_answer" in AgentState.__annotations__
