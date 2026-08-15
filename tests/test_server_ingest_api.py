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


def test_async_flag_returns_202_and_enqueues(scoped_client, monkeypatch):
    http, memory, server_main = scoped_client
    ingest_id = uuid.uuid4()
    enqueue = MagicMock(return_value=ingest_id)
    monkeypatch.setattr(server_main.ingest_worker, "enqueue_ingest", enqueue)

    response = http.post(
        "/memories",
        json={
            "messages": [{"role": "user", "content": "remember this"}],
            "scope": "project",
            "project_id": "occ",
            "infer": True,
            "async": True,
        },
    )

    assert response.status_code == 202
    assert response.json() == {"ingest_id": str(ingest_id), "status": "pending"}
    memory.add.assert_not_called()
    kwargs = enqueue.call_args.kwargs
    assert kwargs["payload"]["messages"] == [{"role": "user", "content": "remember this"}]
    assert kwargs["payload"]["params"] == {"infer": True}
    assert kwargs["scope"] == "project"
    assert kwargs["project_id"] == "occ"
    assert "async" not in kwargs["payload"]["params"]
    assert "async_processing" not in kwargs["payload"]["params"]


def test_without_flag_stays_synchronous(scoped_client, monkeypatch):
    http, memory, server_main = scoped_client
    enqueue = MagicMock()
    monkeypatch.setattr(server_main.ingest_worker, "enqueue_ingest", enqueue)

    response = http.post(
        "/memories",
        json={
            "messages": [{"role": "user", "content": "remember this"}],
            "scope": "project",
            "project_id": "occ",
        },
    )

    assert response.status_code == 200
    memory.add.assert_called_once()
    enqueue.assert_not_called()


def test_async_flag_keeps_validation_synchronous(scoped_client, monkeypatch):
    http, memory, server_main = scoped_client
    enqueue = MagicMock()
    monkeypatch.setattr(server_main.ingest_worker, "enqueue_ingest", enqueue)

    response = http.post(
        "/memories",
        json={
            "messages": [{"role": "user", "content": "x"}],
            "user_id": "someone-else",
            "scope": "project",
            "project_id": "occ",
            "async": True,
        },
    )

    assert response.status_code == 400
    enqueue.assert_not_called()
