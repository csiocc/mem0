import importlib
import os
import uuid
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("mcp", reason="mcp sdk not installed")

from fastapi import HTTPException
from fastapi.testclient import TestClient

MCP_ACCEPT = "application/json, text/event-stream"

USER_A_ID = "11111111-1111-1111-1111-111111111111"
USER_B_ID = "22222222-2222-2222-2222-222222222222"


def _rpc_body(method, params=None, id=1):
    body = {"jsonrpc": "2.0", "id": id, "method": method}
    if params is not None:
        body["params"] = params
    return body


def rpc(client, method, params=None, api_key="key-a", id=1):
    headers = {"Accept": MCP_ACCEPT, "Content-Type": "application/json"}
    if api_key is not None:
        headers["X-API-Key"] = api_key
    return client.post("/mcp", json=_rpc_body(method, params, id), headers=headers)


INIT_PARAMS = {
    "protocolVersion": "2026-07-28",
    "capabilities": {},
    "clientInfo": {"name": "pytest", "version": "0"},
}


@pytest.fixture
def mcp_client():
    memory = MagicMock()
    memory.add.return_value = {"results": [{"id": "mem-1", "event": "ADD", "memory": "stored"}]}
    memory.get.return_value = {
        "id": "mem-1",
        "memory": "stored",
        "user_id": USER_A_ID,
        "metadata": {"scope_key": "project:occ"},
    }
    memory.get_all.return_value = {"results": []}
    memory.search.return_value = {"results": []}

    with patch.dict(
        os.environ,
        {
            "AUTH_DISABLED": "false",
            "JWT_SECRET": "test-jwt-secret",
            "POSTGRES_PASSWORD": "test-password",
            "MEM0_TELEMETRY": "false",
        },
        clear=True,
    ):
        with patch("mem0.Memory.from_config", return_value=memory):
            import auth as server_auth
            import server_state

            importlib.reload(server_auth)
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setattr(server_state, "_load_overrides", lambda: {})
            import mcp_endpoint

            importlib.reload(mcp_endpoint)
            import server.main as server_main

            importlib.reload(server_main)
            monkeypatch.setattr(server_main, "_should_log_request", lambda request: False)

            user_a = server_main.User(
                id=uuid.UUID(USER_A_ID), name="A", email="a@example.test",
                password_hash="unused", role="user",
            )
            user_b = server_main.User(
                id=uuid.UUID(USER_B_ID), name="B", email="b@example.test",
                password_hash="unused", role="user",
            )

            def fake_resolver(api_key):
                users = {"key-a": user_a, "key-b": user_b}
                if api_key not in users:
                    raise HTTPException(status_code=401, detail="Invalid API key.")
                return users[api_key]

            monkeypatch.setattr(mcp_endpoint, "resolve_mcp_user", fake_resolver)
            with TestClient(server_main.app) as client:
                yield client, memory, mcp_endpoint
            monkeypatch.undo()


def test_mcp_requires_api_key(mcp_client):
    client, _, _ = mcp_client
    response = rpc(client, "initialize", INIT_PARAMS, api_key=None)
    assert response.status_code == 401


def test_mcp_rejects_invalid_api_key(mcp_client):
    client, _, _ = mcp_client
    response = rpc(client, "initialize", INIT_PARAMS, api_key="wrong")
    assert response.status_code == 401
    assert "wrong" not in response.text


def test_mcp_initialize_with_valid_key(mcp_client):
    client, _, _ = mcp_client
    response = rpc(client, "initialize", INIT_PARAMS)
    assert response.status_code == 200
    assert response.json()["result"]["serverInfo"]["name"] == "mem0"
