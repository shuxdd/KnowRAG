"""Integration test: full auth flow + protected route access."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from backend.main import app
    return TestClient(app)


def test_full_auth_flow(client):
    """端到端认证流程：注册 → 登录 → 访问受保护路由 → 错误 token 被拒。"""

    # 1. Register
    resp = client.post(
        "/api/auth/register",
        json={"username": "test_integration_user", "password": "flowpass"},
    )
    assert resp.status_code == 200
    assert resp.json()["username"] == "test_integration_user"

    # 2. Login
    resp = client.post(
        "/api/auth/login",
        json={"username": "test_integration_user", "password": "flowpass"},
    )
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    assert len(token) > 0

    # 3. Access protected route with token
    resp = client.get(
        "/api/qa/sessions",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200  # Not 401

    # 4. Access protected route without token
    resp = client.get("/api/qa/sessions")
    assert resp.status_code == 401

    # 5. Access with wrong token
    resp = client.get(
        "/api/qa/sessions",
        headers={"Authorization": "Bearer fake-token-that-is-invalid"},
    )
    assert resp.status_code == 401

    # 6. Me endpoint
    resp = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["username"] == "test_integration_user"
