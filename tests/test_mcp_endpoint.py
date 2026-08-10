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


def test_unknown_path_is_404_not_401(mcp_client):
    client, _, _ = mcp_client
    get_response = client.get("/definitely-not-a-route")
    assert get_response.status_code == 404

    post_response = client.post("/definitely-not-a-route", json={})
    assert post_response.status_code == 404

    # /mcp itself is still gated by the API key.
    still_gated = rpc(client, "initialize", INIT_PARAMS, api_key=None)
    assert still_gated.status_code == 401


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


def tool_call(client, name, arguments, api_key="key-a"):
    response = rpc(
        client,
        "tools/call",
        {"name": name, "arguments": arguments},
        api_key=api_key,
        id=2,
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    return result


def tool_payload(result):
    import json

    if result.get("structuredContent") is not None:
        return result["structuredContent"]
    return json.loads(result["content"][0]["text"])


def test_search_scopes_to_authenticated_user_and_wraps_untrusted(mcp_client):
    client, memory, _ = mcp_client
    memory.search.return_value = {"results": [{"id": "mem-1", "memory": "note"}]}

    result = tool_call(
        client, "search_memories", {"query": "note", "scope": "project", "project_id": "occ"}
    )

    payload = tool_payload(result)
    assert payload["untrusted_reference_data"] is True
    assert payload["results"][0]["memory"] == "note"
    _, kwargs = memory.search.call_args
    assert kwargs["filters"] == {
        "user_id": USER_A_ID,
        "OR": [{"scope_key": "global"}, {"scope_key": "project:occ"}],
    }
    assert kwargs["top_k"] == 100


def test_search_rejects_bad_slug_regardless_of_scope_casing(mcp_client):
    client, memory, _ = mcp_client

    result = tool_call(
        client,
        "search_memories",
        {"query": "note", "scope": " Project", "project_id": "Bad Slug"},
    )

    assert result.get("isError") is True
    assert "project_id must be a lowercase slug" in result["content"][0]["text"]
    memory.search.assert_not_called()


def test_search_requires_project_id_for_project_scope(mcp_client):
    client, memory, _ = mcp_client

    result = tool_call(client, "search_memories", {"query": "note"})

    assert result.get("isError") is True
    assert "project_id is required" in result["content"][0]["text"]
    memory.search.assert_not_called()


def test_search_backend_error_is_sanitized(mcp_client):
    client, memory, _ = mcp_client
    memory.search.side_effect = RuntimeError("postgresql://user:pass@host/db exploded")

    result = tool_call(client, "search_memories", {"query": "q", "scope": "global"})

    assert result.get("isError") is True
    text = result["content"][0]["text"]
    assert "Memory backend unavailable." in text
    assert "postgresql" not in text


def test_search_budget_omits_oversized_entries(mcp_client):
    client, memory, _ = mcp_client
    memory.search.return_value = {
        "results": [{"id": "m1", "memory": "x" * 200}, {"id": "m2", "memory": "y" * 5000}]
    }

    result = tool_call(
        client,
        "search_memories",
        {"query": "q", "scope": "global", "context_budget_chars": 1000},
    )

    payload = tool_payload(result)
    assert [entry["id"] for entry in payload["results"]] == ["m1"]
    assert payload["omitted_results"] == 1


def test_get_memories_lists_scope(mcp_client):
    client, memory, _ = mcp_client
    memory.get_all.return_value = {"results": [{"id": "mem-1", "memory": "note"}]}

    result = tool_call(client, "get_memories", {"scope": "project", "project_id": "occ"})

    payload = tool_payload(result)
    assert payload["untrusted_reference_data"] is True
    _, kwargs = memory.get_all.call_args
    assert kwargs["top_k"] == 20


def test_get_memory_returns_owned(mcp_client):
    client, memory, _ = mcp_client

    result = tool_call(client, "get_memory", {"memory_id": "mem-1"})

    assert tool_payload(result)["id"] == "mem-1"


def test_get_memory_foreign_is_not_found(mcp_client):
    client, memory, _ = mcp_client
    memory.get.return_value = {"id": "mem-9", "memory": "foreign", "user_id": USER_B_ID}

    result = tool_call(client, "get_memory", {"memory_id": "mem-9"})

    assert result.get("isError") is True
    assert "not found" in result["content"][0]["text"].lower()


def test_add_memory_binds_scope_and_disables_inference(mcp_client):
    client, memory, _ = mcp_client

    result = tool_call(
        client, "add_memory", {"text": "Use uv for tooling", "scope": "project", "project_id": "occ"}
    )

    assert tool_payload(result)["results"][0]["event"] == "ADD"
    _, kwargs = memory.add.call_args
    assert kwargs["user_id"] == USER_A_ID
    assert kwargs["infer"] is False
    assert kwargs["metadata"]["scope_key"] == "project:occ"
    assert kwargs["messages"] == [{"role": "user", "content": "Use uv for tooling"}]


def test_add_memory_rejects_reserved_metadata(mcp_client):
    client, memory, _ = mcp_client

    result = tool_call(
        client,
        "add_memory",
        {"text": "x", "scope": "global", "metadata": {"user_id": "evil"}},
    )

    assert result.get("isError") is True
    assert "reserved metadata" in result["content"][0]["text"]
    memory.add.assert_not_called()


def test_add_memory_rejects_bad_project_slug(mcp_client):
    client, memory, _ = mcp_client

    result = tool_call(
        client, "add_memory", {"text": "x", "scope": "project", "project_id": "Bad Slug"}
    )

    assert result.get("isError") is True
    memory.add.assert_not_called()


def test_update_memory_checks_ownership_and_reserved_keys(mcp_client):
    client, memory, _ = mcp_client
    memory.update.return_value = {"message": "Memory updated successfully"}

    result = tool_call(client, "update_memory", {"memory_id": "mem-1", "text": "new text"})

    assert tool_payload(result)["message"] == "Memory updated successfully"
    _, kwargs = memory.update.call_args
    assert kwargs == {"memory_id": "mem-1", "text": "new text"}

    bad = tool_call(
        client,
        "update_memory",
        {"memory_id": "mem-1", "text": "t", "metadata": {"scope": "global"}},
    )
    assert bad.get("isError") is True


def test_update_memory_foreign_is_not_found(mcp_client):
    client, memory, _ = mcp_client
    memory.get.return_value = {"id": "mem-9", "memory": "foreign", "user_id": USER_B_ID}

    result = tool_call(client, "update_memory", {"memory_id": "mem-9", "text": "hijack"})

    assert result.get("isError") is True
    memory.update.assert_not_called()


def test_delete_memory_owned_only(mcp_client):
    client, memory, _ = mcp_client

    result = tool_call(client, "delete_memory", {"memory_id": "mem-1"})

    assert tool_payload(result)["message"] == "Memory deleted successfully"
    memory.delete.assert_called_once_with(memory_id="mem-1")

    memory.get.return_value = {"id": "mem-9", "memory": "foreign", "user_id": USER_B_ID}
    foreign = tool_call(client, "delete_memory", {"memory_id": "mem-9"})
    assert foreign.get("isError") is True


def test_begin_memory_import(mcp_client, monkeypatch):
    client, _, mcp_endpoint = mcp_client
    source = MagicMock(
        id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        stored_count=0, duplicate_count=0, failed_count=0,
    )
    begin = MagicMock(state="started", source=source)
    called = {}

    def fake_begin_import(**kwargs):
        called.update(kwargs)
        return begin

    monkeypatch.setattr(mcp_endpoint, "begin_import", fake_begin_import)
    monkeypatch.setattr(mcp_endpoint, "SessionLocal", MagicMock())

    result = tool_call(
        client,
        "begin_memory_import",
        {
            "source_ref": "docs/memory.md",
            "source_sha256": "a" * 64,
            "scope": "project",
            "project_id": "occ",
        },
    )

    payload = tool_payload(result)
    assert payload["state"] == "started"
    assert payload["import_id"] == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    assert called["user_id"] == uuid.UUID(USER_A_ID)


def test_begin_memory_import_rejects_bad_sha(mcp_client, monkeypatch):
    client, _, mcp_endpoint = mcp_client
    guard = MagicMock()
    monkeypatch.setattr(mcp_endpoint, "begin_import", guard)

    result = tool_call(
        client,
        "begin_memory_import",
        {"source_ref": "x.md", "source_sha256": "nothex", "scope": "global"},
    )

    assert result.get("isError") is True
    guard.assert_not_called()


def test_append_memory_import(mcp_client, monkeypatch):
    client, _, mcp_endpoint = mcp_client
    source = MagicMock(
        id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        stored_count=2, duplicate_count=1, failed_count=0,
    )
    append = MagicMock(source=source, failed=[], errors=[])
    monkeypatch.setattr(mcp_endpoint, "append_memories", MagicMock(return_value=append))
    monkeypatch.setattr(mcp_endpoint, "SessionLocal", MagicMock())

    result = tool_call(
        client,
        "append_memory_import",
        {
            "import_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "memories": [
                {"text": "First distilled memory", "source_section": "Setup"},
                {"text": "Second distilled memory", "source_section": None},
            ],
        },
    )

    payload = tool_payload(result)
    assert payload["state"] == "in_progress"
    assert payload["stored_count"] == 2


def test_append_rejects_foreign_import(mcp_client, monkeypatch):
    client, _, mcp_endpoint = mcp_client
    from memory_imports import ImportConflict

    monkeypatch.setattr(
        mcp_endpoint,
        "append_memories",
        MagicMock(side_effect=ImportConflict("Import not found.")),
    )
    monkeypatch.setattr(mcp_endpoint, "SessionLocal", MagicMock())

    result = tool_call(
        client,
        "append_memory_import",
        {
            "import_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "memories": [{"text": "x", "source_section": None}],
        },
    )

    assert result.get("isError") is True
    assert "not found" in result["content"][0]["text"].lower()


def test_finish_memory_import(mcp_client, monkeypatch):
    client, _, mcp_endpoint = mcp_client
    source = MagicMock(
        id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        stored_count=2, duplicate_count=1, failed_count=0,
    )
    monkeypatch.setattr(mcp_endpoint, "finish_import", MagicMock(return_value=source))
    monkeypatch.setattr(mcp_endpoint, "SessionLocal", MagicMock())

    result = tool_call(
        client, "finish_memory_import", {"import_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"}
    )

    assert tool_payload(result)["state"] == "completed"


def test_user_b_cannot_read_update_or_delete_user_a_memory(mcp_client):
    client, memory, _ = mcp_client
    # memory.get keeps returning USER_A's mem-1.

    for name, args in [
        ("get_memory", {"memory_id": "mem-1"}),
        ("update_memory", {"memory_id": "mem-1", "text": "hijack"}),
        ("delete_memory", {"memory_id": "mem-1"}),
    ]:
        result = tool_call(client, name, args, api_key="key-b")
        assert result.get("isError") is True, name
        assert "not found" in result["content"][0]["text"].lower()
    memory.update.assert_not_called()
    memory.delete.assert_not_called()


def test_search_filters_are_bound_to_the_calling_user(mcp_client):
    client, memory, _ = mcp_client
    memory.search.return_value = {"results": []}

    tool_call(client, "search_memories", {"query": "q", "scope": "global"}, api_key="key-b")

    _, kwargs = memory.search.call_args
    assert kwargs["filters"] == {"user_id": USER_B_ID, "scope_key": "global"}


def test_session_manager_is_stateless(mcp_client):
    _, _, mcp_endpoint = mcp_client
    assert mcp_endpoint.mcp.session_manager.stateless is True


def test_tools_list_exposes_exactly_nine_tools_without_user_id(mcp_client):
    client, _, _ = mcp_client
    response = rpc(client, "tools/list", {}, id=3)
    tools = response.json()["result"]["tools"]
    names = sorted(tool["name"] for tool in tools)
    assert names == [
        "add_memory",
        "append_memory_import",
        "begin_memory_import",
        "delete_memory",
        "finish_memory_import",
        "get_memories",
        "get_memory",
        "search_memories",
        "update_memory",
    ]
    for tool in tools:
        assert "user_id" not in tool["inputSchema"].get("properties", {})
