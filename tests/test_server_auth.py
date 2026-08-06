"""E2E tests for REST API server authentication.

Tests the actual server/main.py app through FastAPI's TestClient (full ASGI
round-trip) covering the token-bound security contract:
  - Unauthenticated requests are rejected on every protected endpoint.
  - The legacy ADMIN_API_KEY maps to the default server user (or the
    bootstrap admin for admin-only routes on a fresh deploy).
  - AUTH_DISABLED mode stays open for local development but still refuses
    user-scoped memory operations when no user exists.
  - Startup logging and the JWT_SECRET requirement.
"""

import importlib
import logging
import os
import uuid
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi import HTTPException
from fastapi.testclient import TestClient

USER_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")

PROTECTED_ENDPOINTS = [
    ("POST", "/configure"),
    ("POST", "/memories"),
    ("GET", "/memories"),
    ("GET", "/memories/test-id"),
    ("POST", "/search"),
    ("PUT", "/memories/test-id"),
    ("GET", "/memories/test-id/history"),
    ("DELETE", "/memories/test-id"),
    ("DELETE", "/memories"),
    ("POST", "/reset"),
]


def _mock_memory_instance():
    mock_instance = MagicMock()
    mock_instance.get.return_value = {
        "id": "mem-1",
        "memory": "test memory",
        "user_id": str(USER_ID),
    }
    mock_instance.get_all.return_value = {"results": []}
    mock_instance.add.return_value = {"results": [{"id": "mem-1", "event": "ADD", "memory": "test"}]}
    mock_instance.search.return_value = {"results": []}
    mock_instance.update.return_value = {"message": "Memory updated"}
    mock_instance.history.return_value = [{"id": "mem-1", "old_memory": "a", "new_memory": "b"}]
    mock_instance.delete.return_value = None
    mock_instance.delete_all.return_value = None
    mock_instance.reset.return_value = None
    return mock_instance


def _load_app(env_overrides, memory, monkeypatch, default_user="unset"):
    """Reload auth and server.main with the given environment and return the module."""
    with patch.dict(os.environ, {"MEM0_TELEMETRY": "false", **env_overrides}, clear=True):
        with patch("mem0.Memory.from_config", return_value=memory):
            import auth as server_auth
            import server_state

            importlib.reload(server_auth)
            # Keep the reload offline: config overrides normally come from PostgreSQL.
            monkeypatch.setattr(server_state, "_load_overrides", lambda: {})
            monkeypatch.setattr(server_state, "_save_overrides", lambda overrides: None)
            if default_user != "unset":
                monkeypatch.setattr(server_auth, "_get_default_user", lambda db: default_user)
            import server.main as server_main

            importlib.reload(server_main)
            monkeypatch.setattr(server_main, "_should_log_request", lambda request: False)
            return server_main


def _default_user(role="user"):
    from models import User

    return User(
        id=USER_ID,
        name="CSI",
        email="csi@example.test",
        password_hash="unused",
        role=role,
    )


# ---------------------------------------------------------------------------
# Auth enabled: unauthenticated requests are rejected
# ---------------------------------------------------------------------------

class TestUnauthenticatedRejection:
    API_KEY = "test-secret-key-12345"

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        self.mock = _mock_memory_instance()
        module = _load_app(
            {"JWT_SECRET": "test-jwt-secret", "ADMIN_API_KEY": self.API_KEY},
            self.mock,
            monkeypatch,
        )
        self.client = TestClient(module.app)

    def test_root_redirects_to_docs(self):
        resp = self.client.get("/", follow_redirects=False)
        assert resp.status_code == 307
        assert "/docs" in resp.headers["location"]

    @pytest.mark.parametrize("method,path", PROTECTED_ENDPOINTS)
    def test_all_endpoints_reject_without_credentials(self, method, path):
        resp = self.client.request(method, path)
        assert resp.status_code == 401, f"{method} {path} should require auth"

    def test_missing_credentials_detail_mentions_header(self):
        resp = self.client.get("/memories/mem-1")
        assert "X-API-Key" in resp.json()["detail"]

    def test_401_includes_www_authenticate_header(self):
        resp = self.client.get("/memories/mem-1")
        assert resp.headers.get("www-authenticate") == "Bearer"

    def test_handlers_not_reached_without_credentials(self):
        for method, path in PROTECTED_ENDPOINTS:
            self.client.request(method, path)
        self.mock.add.assert_not_called()
        self.mock.get.assert_not_called()
        self.mock.search.assert_not_called()
        self.mock.update.assert_not_called()
        self.mock.history.assert_not_called()
        self.mock.delete.assert_not_called()
        self.mock.reset.assert_not_called()

    def test_openapi_schema_accessible_without_key(self):
        resp = self.client.get("/openapi.json")
        assert resp.status_code == 200
        assert "paths" in resp.json()

    def test_openapi_schema_documents_auth(self):
        schema = self.client.get("/openapi.json").json()
        assert "Authentication" in schema.get("info", {}).get("description", "")

    def test_user_provisioning_routes_are_registered(self):
        """Auth enforcement for /users is covered in test_users_router; here we
        only prove server.main registers the router."""
        paths = self.client.get("/openapi.json").json()["paths"]
        assert "get" in paths["/users"]
        assert "post" in paths["/users"]


# ---------------------------------------------------------------------------
# Legacy ADMIN_API_KEY resolves to the default server user
# ---------------------------------------------------------------------------

class TestAdminApiKey:
    API_KEY = "test-secret-key-12345"

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        self.mock = _mock_memory_instance()
        self.module = _load_app(
            {"JWT_SECRET": "test-jwt-secret", "ADMIN_API_KEY": self.API_KEY},
            self.mock,
            monkeypatch,
            default_user=_default_user(role="admin"),
        )
        self.monkeypatch = monkeypatch
        self.client = TestClient(self.module.app)

    def _authed(self, method, path, **kwargs):
        headers = kwargs.pop("headers", {})
        headers["X-API-Key"] = self.API_KEY
        return self.client.request(method, path, headers=headers, **kwargs)

    def test_create_memory_with_key_binds_default_user(self):
        resp = self._authed("POST", "/memories", json={
            "messages": [{"role": "user", "content": "I like pizza"}],
            "scope": "project",
            "project_id": "test-project",
        })
        assert resp.status_code == 200
        _, kwargs = self.mock.add.call_args
        assert kwargs["user_id"] == str(USER_ID)

    def test_search_with_key(self):
        resp = self._authed("POST", "/search", json={
            "query": "pizza",
            "scope": "project",
            "project_id": "test-project",
        })
        assert resp.status_code == 200

    def test_get_memory_with_key(self):
        resp = self._authed("GET", "/memories/mem-1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "mem-1"

    def test_get_all_memories_with_key(self):
        resp = self._authed("GET", "/memories", params={"scope": "global"})
        assert resp.status_code == 200

    def test_update_memory_with_key(self):
        resp = self._authed("PUT", "/memories/mem-1", json={"text": "updated"})
        assert resp.status_code == 200

    def test_history_with_key(self):
        resp = self._authed("GET", "/memories/mem-1/history")
        assert resp.status_code == 200

    def test_delete_memory_with_key(self):
        resp = self._authed("DELETE", "/memories/mem-1")
        assert resp.status_code == 200

    def test_reset_with_key(self):
        resp = self._authed("POST", "/reset")
        assert resp.status_code == 200

    def test_configure_with_key(self):
        with patch("mem0.Memory.from_config", return_value=self.mock):
            resp = self._authed("POST", "/configure", json={"version": "v1.1"})
        assert resp.status_code == 200

    def test_wrong_key_rejected_before_handler(self):
        import auth as server_auth

        def _reject(key, db):
            raise HTTPException(status_code=401, detail="Invalid API key.")

        self.monkeypatch.setattr(server_auth, "_resolve_user_from_api_key", _reject)
        resp = self.client.get("/memories/mem-1", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401
        assert "Invalid" in resp.json()["detail"]
        self.mock.get.assert_not_called()


class TestAdminApiKeyFreshDeploy:
    """Fresh deploy: admin key exists but no user rows yet."""

    API_KEY = "test-secret-key-12345"

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        self.mock = _mock_memory_instance()
        module = _load_app(
            {"JWT_SECRET": "test-jwt-secret", "ADMIN_API_KEY": self.API_KEY},
            self.mock,
            monkeypatch,
            default_user=None,
        )
        self.client = TestClient(module.app)

    def _authed(self, method, path, **kwargs):
        headers = kwargs.pop("headers", {})
        headers["X-API-Key"] = self.API_KEY
        return self.client.request(method, path, headers=headers, **kwargs)

    def test_user_scoped_memory_operations_rejected_without_user(self):
        resp = self._authed("POST", "/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            "scope": "global",
        })
        assert resp.status_code == 401
        self.mock.add.assert_not_called()

    def test_admin_routes_bootstrap_without_user(self):
        resp = self._authed("POST", "/reset")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# AUTH_DISABLED mode (local development only)
# ---------------------------------------------------------------------------

class TestAuthDisabled:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        self.mock = _mock_memory_instance()
        module = _load_app(
            {"AUTH_DISABLED": "true"},
            self.mock,
            monkeypatch,
            default_user=_default_user(role="admin"),
        )
        self.client = TestClient(module.app)

    def test_create_memory_without_key_uses_default_user(self):
        resp = self.client.post("/memories", json={
            "messages": [{"role": "user", "content": "I like pizza"}],
            "scope": "project",
            "project_id": "test-project",
        })
        assert resp.status_code == 200
        _, kwargs = self.mock.add.call_args
        assert kwargs["user_id"] == str(USER_ID)

    def test_search_without_key(self):
        resp = self.client.post("/search", json={
            "query": "pizza",
            "scope": "global",
        })
        assert resp.status_code == 200

    def test_supplying_key_matching_nothing_is_rejected(self):
        """Sending an unknown X-API-Key must not silently fall back to open mode."""
        import auth as server_auth

        with patch.object(
            server_auth,
            "_resolve_user_from_api_key",
            side_effect=HTTPException(status_code=401, detail="Invalid API key."),
        ):
            resp = self.client.get("/memories/mem-1", headers={"X-API-Key": "some-random-key"})
        assert resp.status_code == 401

    @pytest.mark.parametrize("method,path", PROTECTED_ENDPOINTS)
    def test_no_endpoint_returns_401_when_auth_disabled(self, method, path):
        resp = self.client.request(method, path)
        assert resp.status_code != 401, f"{method} {path} should not require auth"


class TestAuthDisabledWithoutUsers:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        self.mock = _mock_memory_instance()
        module = _load_app(
            {"AUTH_DISABLED": "true"},
            self.mock,
            monkeypatch,
            default_user=None,
        )
        self.client = TestClient(module.app)

    def test_user_scoped_operations_need_a_user(self):
        resp = self.client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            "scope": "global",
        })
        assert resp.status_code == 401
        self.mock.add.assert_not_called()


# ---------------------------------------------------------------------------
# Startup contract
# ---------------------------------------------------------------------------

class TestStartup:
    def test_warning_when_auth_disabled(self, caplog, monkeypatch):
        with caplog.at_level(logging.WARNING):
            _load_app({"AUTH_DISABLED": "true"}, _mock_memory_instance(), monkeypatch)
        assert any("AUTH_DISABLED" in r.message for r in caplog.records)

    def test_warning_when_admin_key_too_short(self, caplog, monkeypatch):
        with caplog.at_level(logging.WARNING):
            _load_app(
                {"JWT_SECRET": "test-jwt-secret", "ADMIN_API_KEY": "short"},
                _mock_memory_instance(),
                monkeypatch,
            )
        assert any("shorter than" in r.message for r in caplog.records)

    def test_missing_jwt_secret_refuses_to_start(self, monkeypatch):
        with pytest.raises(RuntimeError, match="JWT_SECRET"):
            _load_app({"ADMIN_API_KEY": "test-secret-key-12345"}, _mock_memory_instance(), monkeypatch)
