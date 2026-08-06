import importlib
import os
import uuid
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient


@pytest.fixture
def scoped_client():
    memory = MagicMock()
    memory.add.return_value = {
        "results": [{"id": "mem-1", "event": "ADD", "memory": "stored"}]
    }
    memory.get.return_value = {
        "id": "mem-1",
        "memory": "stored",
        "user_id": "11111111-1111-1111-1111-111111111111",
        "metadata": {"scope_key": "project:LernendeLab"},
    }
    memory.get_all.return_value = {"results": []}
    memory.search.return_value = {"results": []}

    with patch.dict(
        os.environ,
        {
            "AUTH_DISABLED": "true",
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
            # Keep the reload offline: config overrides normally come from PostgreSQL.
            monkeypatch.setattr(server_state, "_load_overrides", lambda: {})
            import server.main as server_main

            importlib.reload(server_main)
            monkeypatch.setattr(
                server_main,
                "_should_log_request",
                lambda request: False,
            )
            user = server_main.User(
                id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                name="CSI",
                email="csi@example.test",
                password_hash="unused",
                role="user",
            )
            server_main.app.dependency_overrides[server_main.require_auth] = lambda: user
            yield TestClient(server_main.app), memory, server_main
            server_main.app.dependency_overrides.clear()
            monkeypatch.undo()


def test_create_uses_authenticated_user(scoped_client):
    client, memory, _ = scoped_client

    response = client.post(
        "/memories",
        json={
            "messages": [{"role": "user", "content": "Use local embeddings."}],
            "scope": "project",
            "project_id": "LernendeLab",
            "infer": False,
        },
    )

    assert response.status_code == 200
    _, kwargs = memory.add.call_args
    assert kwargs["user_id"] == "11111111-1111-1111-1111-111111111111"
    assert kwargs["metadata"]["scope_key"] == "project:LernendeLab"


def test_create_rejects_client_user_id(scoped_client):
    client, memory, _ = scoped_client

    response = client.post(
        "/memories",
        json={
            "messages": [{"role": "user", "content": "malicious"}],
            "user_id": "another-user",
            "scope": "global",
        },
    )

    assert response.status_code == 400
    assert "user_id" in response.json()["detail"]
    memory.add.assert_not_called()


def test_project_search_includes_only_own_global_and_project_scope(scoped_client):
    client, memory, _ = scoped_client

    response = client.post(
        "/search",
        json={
            "query": "embeddings",
            "scope": "project",
            "project_id": "LernendeLab",
        },
    )

    assert response.status_code == 200
    _, kwargs = memory.search.call_args
    assert kwargs["filters"] == {
        "user_id": "11111111-1111-1111-1111-111111111111",
        "OR": [
            {"scope_key": "global"},
            {"scope_key": "project:LernendeLab"},
        ],
    }


def test_foreign_memory_is_hidden(scoped_client):
    client, memory, _ = scoped_client
    memory.get.return_value = {
        "id": "foreign",
        "memory": "secret",
        "user_id": "22222222-2222-2222-2222-222222222222",
    }

    response = client.get("/memories/foreign")

    assert response.status_code == 404


@pytest.mark.parametrize(
    "method,path,json_body",
    [
        ("GET", "/memories/foreign", None),
        ("PUT", "/memories/foreign", {"text": "changed"}),
        ("GET", "/memories/foreign/history", None),
        ("DELETE", "/memories/foreign", None),
    ],
)
def test_foreign_memory_cannot_be_read_or_mutated(
    scoped_client,
    method,
    path,
    json_body,
):
    client, memory, _ = scoped_client
    memory.get.return_value = {
        "id": "foreign",
        "memory": "secret",
        "user_id": "22222222-2222-2222-2222-222222222222",
    }

    response = client.request(method, path, json=json_body)

    assert response.status_code == 404
    memory.update.assert_not_called()
    memory.history.assert_not_called()
    memory.delete.assert_not_called()
