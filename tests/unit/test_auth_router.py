"""Tests for auth router — register, login, me endpoints."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from backend.main import app
    return TestClient(app)


class TestRegister:
    def test_register_new_user(self, client):
        resp = client.post(
            "/api/auth/register",
            json={"username": "test_alice", "password": "secret123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "test_alice"
        assert "id" in data

    def test_register_duplicate_user(self, client):
        client.post(
            "/api/auth/register",
            json={"username": "test_bob", "password": "secret123"},
        )
        resp = client.post(
            "/api/auth/register",
            json={"username": "test_bob", "password": "secret456"},
        )
        assert resp.status_code == 409

    def test_register_invalid_username(self, client):
        resp = client.post(
            "/api/auth/register",
            json={"username": "ab", "password": "secret123"},
        )
        assert resp.status_code == 422


class TestLogin:
    def test_login_success(self, client):
        client.post(
            "/api/auth/register",
            json={"username": "test_charlie", "password": "mypassword"},
        )
        resp = client.post(
            "/api/auth/login",
            json={"username": "test_charlie", "password": "mypassword"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client):
        client.post(
            "/api/auth/register",
            json={"username": "test_dave", "password": "correct"},
        )
        resp = client.post(
            "/api/auth/login",
            json={"username": "test_dave", "password": "wrong"},
        )
        assert resp.status_code == 401

    def test_login_nonexistent_user(self, client):
        resp = client.post(
            "/api/auth/login",
            json={"username": "no_such_user", "password": "irrelevant"},
        )
        assert resp.status_code == 401


class TestMe:
    def test_me_with_valid_token(self, client):
        client.post(
            "/api/auth/register",
            json={"username": "test_eve", "password": "secret123"},
        )
        login_resp = client.post(
            "/api/auth/login",
            json={"username": "test_eve", "password": "secret123"},
        )
        token = login_resp.json()["access_token"]

        resp = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["username"] == "test_eve"

    def test_me_without_token(self, client):
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401
