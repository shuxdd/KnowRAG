import pytest


class TestAgentRequest:
    def test_agent_request_schema_valid(self):
        from backend.models.schemas import AgentRequest
        req = AgentRequest(question="test question", session_id="s1")
        assert req.question == "test question"
        assert req.session_id == "s1"

    def test_agent_request_defaults(self):
        from backend.models.schemas import AgentRequest
        req = AgentRequest(question="test")
        assert req.session_id is None


class TestAgentRoute:
    def test_agent_endpoint_exists(self):
        # Only import the router, not the full app (avoids model loading)
        from backend.routers.qa import router
        paths = [r.path for r in router.routes]
        assert "/api/qa/agent" in paths
