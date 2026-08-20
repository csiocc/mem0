"""Tests for REST API parameter forwarding.

Verifies that the Pydantic request models in server/main.py correctly accept
and forward all parameters supported by the underlying Memory class methods
(top_k, threshold, explain, infer, memory_type, prompt) while the identity
and scope filters are derived from the authenticated user.
"""

import importlib
import os
import uuid
from unittest.mock import MagicMock, patch

import pytest

from mem0.exceptions import ValidationError as Mem0ValidationError

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient

USER_ID = "11111111-1111-1111-1111-111111111111"
SCOPE = {"scope": "project", "project_id": "test-project"}
PROJECT_FILTERS = {
    "user_id": USER_ID,
    "OR": [
        {"scope_key": "global"},
        {"scope_key": "project:test-project"},
    ],
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_memory():
    """Patch Memory.from_config so the server imports without a real backend."""
    mock_instance = MagicMock()
    mock_instance.add.return_value = {"results": [{"id": "mem-1", "event": "ADD", "memory": "test"}]}
    mock_instance.search.return_value = {"results": []}
    mock_instance.get.return_value = {"id": "mem-1", "memory": "test memory", "user_id": USER_ID}
    mock_instance.get_all.return_value = {"results": []}
    mock_instance.update.return_value = {"message": "Memory updated"}
    mock_instance.history.return_value = [{"id": "mem-1", "old_memory": "a", "new_memory": "b"}]
    mock_instance.delete.return_value = None
    return mock_instance


@pytest.fixture
def client(mock_memory, monkeypatch):
    """Return a TestClient wired to the server app with mocked Memory and a stable test user."""
    with patch.dict(
        os.environ,
        {
            "AUTH_DISABLED": "true",
            "POSTGRES_PASSWORD": "test-password",
            "MEM0_TELEMETRY": "false",
        },
        clear=True,
    ):
        with patch("mem0.Memory.from_config", return_value=mock_memory):
            import auth as server_auth
            import server_state

            importlib.reload(server_auth)
            # Keep the reload offline: config overrides normally come from PostgreSQL.
            monkeypatch.setattr(server_state, "_load_overrides", lambda: {})
            import server.main as server_main

            importlib.reload(server_main)
            monkeypatch.setattr(server_main, "_should_log_request", lambda request: False)
            user = server_main.User(
                id=uuid.UUID(USER_ID),
                name="CSI",
                email="csi@example.test",
                password_hash="unused",
                role="user",
            )
            server_main.app.dependency_overrides[server_main.require_auth] = lambda: user
            yield TestClient(server_main.app)
            server_main.app.dependency_overrides.clear()


# ===========================================================================
# SearchRequest: top_k parameter
# ===========================================================================

class TestSearchLimit:
    """Verify that the top_k parameter is accepted and forwarded to Memory.search()."""

    def test_limit_forwarded(self, client, mock_memory):
        resp = client.post("/search", json={"query": "food", **SCOPE, "top_k": 5})
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert kwargs["top_k"] == 5

    def test_limit_one(self, client, mock_memory):
        resp = client.post("/search", json={"query": "food", **SCOPE, "top_k": 1})
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert kwargs["top_k"] == 1

    def test_limit_omitted_uses_memory_default(self, client, mock_memory):
        """When top_k is not sent, it should not appear in the kwargs,
        allowing Memory.search() to use its own default."""
        resp = client.post("/search", json={"query": "food", **SCOPE})
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert "top_k" not in kwargs


# ===========================================================================
# SearchRequest: threshold parameter
# ===========================================================================

class TestSearchThreshold:
    """Verify that the threshold parameter is accepted and forwarded."""

    def test_threshold_forwarded(self, client, mock_memory):
        resp = client.post("/search", json={"query": "food", **SCOPE, "threshold": 0.8})
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert kwargs["threshold"] == 0.8

    def test_threshold_zero(self, client, mock_memory):
        """threshold=0.0 is a valid falsy value that must not be filtered out."""
        resp = client.post("/search", json={"query": "food", **SCOPE, "threshold": 0.0})
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert kwargs["threshold"] == 0.0

    def test_threshold_omitted_uses_memory_default(self, client, mock_memory):
        resp = client.post("/search", json={"query": "food", **SCOPE})
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert "threshold" not in kwargs


# ===========================================================================
# SearchRequest: explain parameter
# ===========================================================================

class TestSearchExplain:
    """Verify that the explain parameter is accepted and forwarded."""

    def test_explain_true_forwarded(self, client, mock_memory):
        resp = client.post("/search", json={"query": "food", **SCOPE, "explain": True})
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert kwargs["explain"] is True

    def test_explain_false_forwarded(self, client, mock_memory):
        resp = client.post("/search", json={"query": "food", **SCOPE, "explain": False})
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert kwargs["explain"] is False

    def test_explain_omitted_uses_memory_default(self, client, mock_memory):
        resp = client.post("/search", json={"query": "food", **SCOPE})
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert "explain" not in kwargs


class TestSearchRerank:
    """Reranking is opt-in per request: vector order is cheap, the LLM
    judgement is not, so an unset flag must never trigger it."""

    def test_rerank_true_forwarded(self, client, mock_memory):
        resp = client.post("/search", json={"query": "food", **SCOPE, "rerank": True})
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert kwargs["rerank"] is True

    def test_rerank_omitted_uses_memory_default(self, client, mock_memory):
        resp = client.post("/search", json={"query": "food", **SCOPE})
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert "rerank" not in kwargs


# ===========================================================================
# SearchRequest: top_k + threshold together
# ===========================================================================

class TestSearchLimitAndThreshold:

    def test_both_forwarded(self, client, mock_memory):
        resp = client.post("/search", json={
            "query": "food", **SCOPE, "top_k": 10, "threshold": 0.5
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert kwargs["top_k"] == 10
        assert kwargs["threshold"] == 0.5


# ===========================================================================
# MemoryCreate: infer parameter
# ===========================================================================

class TestAddInfer:
    """Verify that the infer parameter is accepted and forwarded to Memory.add()."""

    def test_infer_false_forwarded(self, client, mock_memory):
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "Store this exactly"}],
            **SCOPE,
            "infer": False,
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.add.call_args
        assert kwargs["infer"] is False

    def test_infer_true_forwarded(self, client, mock_memory):
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "I like pizza"}],
            **SCOPE,
            "infer": True,
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.add.call_args
        assert kwargs["infer"] is True

    def test_infer_omitted_uses_memory_default(self, client, mock_memory):
        """When infer is not sent, it should not appear in kwargs,
        allowing Memory.add() to use its own default (True)."""
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "hello"}],
            **SCOPE,
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.add.call_args
        assert "infer" not in kwargs


# ===========================================================================
# MemoryCreate: memory_type parameter
# ===========================================================================

class TestAddMemoryType:
    """Verify that the memory_type parameter is accepted and forwarded."""

    def test_memory_type_forwarded(self, client, mock_memory):
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "I like pizza"}],
            **SCOPE,
            "memory_type": "core",
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.add.call_args
        assert kwargs["memory_type"] == "core"

    def test_memory_type_omitted(self, client, mock_memory):
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "hello"}],
            **SCOPE,
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.add.call_args
        assert "memory_type" not in kwargs


# ===========================================================================
# MemoryCreate: prompt parameter
# ===========================================================================

class TestAddPrompt:
    """Verify that the prompt parameter is accepted and forwarded."""

    def test_prompt_forwarded(self, client, mock_memory):
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "I like pizza"}],
            **SCOPE,
            "prompt": "Extract food preferences only.",
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.add.call_args
        assert kwargs["prompt"] == "Extract food preferences only."

    def test_prompt_omitted(self, client, mock_memory):
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "hello"}],
            **SCOPE,
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.add.call_args
        assert "prompt" not in kwargs


# ===========================================================================
# MemoryCreate: all params together
# ===========================================================================

class TestAddAllNewParams:

    def test_infer_memory_type_and_prompt_together(self, client, mock_memory):
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "I like pizza"}],
            **SCOPE,
            "infer": False,
            "memory_type": "core",
            "prompt": "Custom extraction prompt.",
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.add.call_args
        assert kwargs["infer"] is False
        assert kwargs["memory_type"] == "core"
        assert kwargs["prompt"] == "Custom extraction prompt."


# ===========================================================================
# Identity and scope binding
# ===========================================================================

class TestScopeBinding:
    """Identity comes from authentication, scope from the request body."""

    def test_add_binds_authenticated_user_and_scope(self, client, mock_memory):
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            **SCOPE,
            "metadata": {"source": "test"},
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.add.call_args
        assert kwargs["user_id"] == USER_ID
        assert kwargs["metadata"]["source"] == "test"
        assert kwargs["metadata"]["scope_key"] == "project:test-project"

    def test_add_rejects_client_user_id(self, client, mock_memory):
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            "user_id": "someone-else",
            **SCOPE,
        })
        assert resp.status_code == 400
        mock_memory.add.assert_not_called()

    def test_add_rejects_reserved_metadata(self, client, mock_memory):
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            **SCOPE,
            "metadata": {"scope_key": "global"},
        })
        assert resp.status_code == 400
        mock_memory.add.assert_not_called()

    def test_search_rejects_client_user_id(self, client, mock_memory):
        resp = client.post("/search", json={
            "query": "food",
            "user_id": "someone-else",
            **SCOPE,
        })
        assert resp.status_code == 400
        mock_memory.search.assert_not_called()

    def test_search_builds_project_filters(self, client, mock_memory):
        resp = client.post("/search", json={"query": "food", **SCOPE})
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert kwargs["filters"] == PROJECT_FILTERS

    def test_search_global_scope_excludes_projects(self, client, mock_memory):
        resp = client.post("/search", json={"query": "food", "scope": "global"})
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert kwargs["filters"] == {"user_id": USER_ID, "scope_key": "global"}

    def test_search_project_scope_requires_project_id(self, client, mock_memory):
        resp = client.post("/search", json={"query": "food", "scope": "project"})
        assert resp.status_code == 400
        mock_memory.search.assert_not_called()


# ===========================================================================
# Edge cases: falsy-but-valid values must not be filtered out
# ===========================================================================

class TestFalsyValues:
    """The handler filters with `v is not None`. Falsy values like False, 0,
    0.0, and empty string must still be forwarded."""

    def test_infer_false_not_filtered(self, client, mock_memory):
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            **SCOPE,
            "infer": False,
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.add.call_args
        assert kwargs["infer"] is False

    def test_threshold_zero_not_filtered(self, client, mock_memory):
        resp = client.post("/search", json={
            "query": "food", **SCOPE, "threshold": 0.0,
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert kwargs["threshold"] == 0.0


# ===========================================================================
# Extra/unknown fields are still silently ignored (existing Pydantic behavior)
# ===========================================================================

class TestUnknownFieldsIgnored:

    def test_unknown_search_field_ignored(self, client, mock_memory):
        resp = client.post("/search", json={
            "query": "food", **SCOPE, "bogus_field": "xyz",
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert "bogus_field" not in kwargs

    def test_unknown_add_field_ignored(self, client, mock_memory):
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            **SCOPE,
            "unknown_param": 42,
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.add.call_args
        assert "unknown_param" not in kwargs


# ===========================================================================
# OpenAPI schema: fields are documented
# ===========================================================================

class TestOpenAPISchema:
    """Verify the request fields appear in the auto-generated OpenAPI schema."""

    def test_search_schema_includes_limit(self, client):
        schema = client.get("/openapi.json").json()
        search_props = schema["components"]["schemas"]["SearchRequest"]["properties"]
        assert "top_k" in search_props
        assert search_props["top_k"]["description"] == "Maximum number of results to return."

    def test_search_schema_includes_threshold(self, client):
        schema = client.get("/openapi.json").json()
        search_props = schema["components"]["schemas"]["SearchRequest"]["properties"]
        assert "threshold" in search_props

    def test_search_schema_includes_scope(self, client):
        schema = client.get("/openapi.json").json()
        search_props = schema["components"]["schemas"]["SearchRequest"]["properties"]
        assert "scope" in search_props
        assert "project_id" in search_props

    def test_add_schema_includes_infer(self, client):
        schema = client.get("/openapi.json").json()
        add_props = schema["components"]["schemas"]["MemoryCreate"]["properties"]
        assert "infer" in add_props

    def test_add_schema_includes_memory_type(self, client):
        schema = client.get("/openapi.json").json()
        add_props = schema["components"]["schemas"]["MemoryCreate"]["properties"]
        assert "memory_type" in add_props

    def test_add_schema_includes_prompt(self, client):
        schema = client.get("/openapi.json").json()
        add_props = schema["components"]["schemas"]["MemoryCreate"]["properties"]
        assert "prompt" in add_props


# ===========================================================================
# Pydantic type validation: invalid types return 422
# ===========================================================================

class TestTypeValidation:
    """Verify FastAPI/Pydantic rejects invalid types with 422."""

    def test_limit_string_rejected(self, client):
        resp = client.post("/search", json={
            "query": "food", **SCOPE, "top_k": "not_a_number",
        })
        assert resp.status_code == 422

    def test_threshold_string_rejected(self, client):
        resp = client.post("/search", json={
            "query": "food", **SCOPE, "threshold": "high",
        })
        assert resp.status_code == 422

    def test_invalid_scope_rejected(self, client):
        resp = client.post("/search", json={
            "query": "food", "scope": "team",
        })
        assert resp.status_code == 422

    def test_infer_string_coerced_by_pydantic(self, client, mock_memory):
        """Pydantic v2 coerces truthy strings like 'yes' to True for bool fields."""
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            **SCOPE,
            "infer": "yes",
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.add.call_args
        assert kwargs["infer"] is True

    def test_infer_invalid_value_rejected(self, client):
        """A value that cannot be coerced to bool should be rejected."""
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            **SCOPE,
            "infer": [1, 2, 3],
        })
        assert resp.status_code == 422

    def test_limit_float_rejected(self, client):
        resp = client.post("/search", json={
            "query": "food", **SCOPE, "top_k": 5.7,
        })
        assert resp.status_code == 422

    def test_memory_type_int_rejected(self, client):
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            **SCOPE,
            "memory_type": 123,
        })
        assert resp.status_code == 422


# ===========================================================================
# Explicit null values: treated as omitted (filtered by `is not None`)
# ===========================================================================

class TestExplicitNull:
    """When a client sends null for an optional field, it should be treated
    as omitted — the Memory class default should be used."""

    def test_limit_null_uses_memory_default(self, client, mock_memory):
        resp = client.post("/search", json={
            "query": "food", **SCOPE, "top_k": None,
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert "top_k" not in kwargs

    def test_infer_null_uses_memory_default(self, client, mock_memory):
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            **SCOPE,
            "infer": None,
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.add.call_args
        assert "infer" not in kwargs

    def test_prompt_null_uses_memory_default(self, client, mock_memory):
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "test"}],
            **SCOPE,
            "prompt": None,
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.add.call_args
        assert "prompt" not in kwargs


# ===========================================================================
# Verify exact call signatures match Memory method params
# ===========================================================================

class TestCallSignatureMatch:
    """Ensure forwarded params exactly match Memory.add() and Memory.search()
    keyword argument names — a typo here would cause a TypeError at runtime."""

    def test_search_kwargs_are_valid(self, client, mock_memory):
        """All kwargs forwarded to Memory.search() must be in its signature."""
        resp = client.post("/search", json={
            "query": "food", **SCOPE, "top_k": 10, "threshold": 0.5,
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        valid_params = {"query", "top_k", "filters", "threshold", "explain", "show_expired", "rerank"}
        for key in kwargs:
            assert key in valid_params, f"Unexpected kwarg '{key}' forwarded to Memory.search()"
        assert kwargs["filters"] == PROJECT_FILTERS

    def test_add_kwargs_are_valid(self, client, mock_memory):
        """All kwargs forwarded to Memory.add() must be in its signature."""
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "hi"}],
            **SCOPE,
            "metadata": {"k": "v"},
            "infer": False, "memory_type": "core", "prompt": "custom",
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.add.call_args
        valid_params = {"messages", "user_id", "metadata", "infer", "memory_type", "prompt", "expiration_date"}
        for key in kwargs:
            assert key in valid_params, f"Unexpected kwarg '{key}' forwarded to Memory.add()"

    def test_messages_excluded_from_params_dict(self, client, mock_memory):
        """messages is passed separately via messages= kwarg, not duplicated from model_dump."""
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "hi"}],
            **SCOPE,
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.add.call_args
        assert "messages" in kwargs
        assert isinstance(kwargs["messages"], list)
        assert kwargs["messages"][0] == {"role": "user", "content": "hi"}

    def test_query_passed_explicitly(self, client, mock_memory):
        """query is passed as an explicit keyword arg to Memory.search()."""
        resp = client.post("/search", json={"query": "food", **SCOPE})
        assert resp.status_code == 200
        _, kwargs = mock_memory.search.call_args
        assert kwargs["query"] == "food"


# ===========================================================================
# MemoryUpdate: text and metadata forwarding
# ===========================================================================

class TestUpdateMemory:
    """Verify that PUT /memories/{id} extracts text and metadata from the
    request body and forwards them correctly to Memory.update()."""

    def test_text_forwarded_as_data(self, client, mock_memory):
        resp = client.put("/memories/mem-1", json={"text": "Likes tennis"})
        assert resp.status_code == 200
        _, kwargs = mock_memory.update.call_args
        assert kwargs["data"] == "Likes tennis"

    def test_metadata_forwarded(self, client, mock_memory):
        resp = client.put("/memories/mem-1", json={
            "text": "Likes tennis",
            "metadata": {"category": "sports"},
        })
        assert resp.status_code == 200
        _, kwargs = mock_memory.update.call_args
        assert kwargs["metadata"] == {"category": "sports"}

    def test_metadata_omitted_passes_none(self, client, mock_memory):
        resp = client.put("/memories/mem-1", json={"text": "Likes tennis"})
        assert resp.status_code == 200
        _, kwargs = mock_memory.update.call_args
        assert "metadata" not in kwargs

    def test_reserved_metadata_rejected(self, client, mock_memory):
        resp = client.put("/memories/mem-1", json={
            "text": "Likes tennis",
            "metadata": {"user_id": "someone-else"},
        })
        assert resp.status_code == 400
        mock_memory.update.assert_not_called()

    def test_expiration_date_forwarded_without_text(self, client, mock_memory):
        resp = client.put("/memories/mem-1", json={"expiration_date": "2999-01-01"})
        assert resp.status_code == 200
        _, kwargs = mock_memory.update.call_args
        assert kwargs["expiration_date"] == "2999-01-01"
        assert "data" not in kwargs

    def test_null_expiration_date_forwarded_for_clear(self, client, mock_memory):
        resp = client.put("/memories/mem-1", json={"expiration_date": None})
        assert resp.status_code == 200
        _, kwargs = mock_memory.update.call_args
        assert kwargs["expiration_date"] is None

    def test_dict_not_passed_as_data(self, client, mock_memory):
        """The entire dict must NOT be passed as data."""
        resp = client.put("/memories/mem-1", json={"text": "updated content"})
        assert resp.status_code == 200
        _, kwargs = mock_memory.update.call_args
        assert isinstance(kwargs["data"], str)


class TestUpdateOpenAPISchema:
    """Verify the MemoryUpdate schema appears in the OpenAPI docs."""

    def test_update_schema_includes_text(self, client):
        schema = client.get("/openapi.json").json()
        update_props = schema["components"]["schemas"]["MemoryUpdate"]["properties"]
        assert "text" in update_props

    def test_update_schema_includes_metadata(self, client):
        schema = client.get("/openapi.json").json()
        update_props = schema["components"]["schemas"]["MemoryUpdate"]["properties"]
        assert "metadata" in update_props


# ===========================================================================
# GET /memories: scope-based listing
# ===========================================================================

class TestGetMemories:
    """Verify that GET /memories builds the scoped filters for the authenticated user."""

    def test_project_scope_filters(self, client, mock_memory):
        response = client.get("/memories", params={"scope": "project", "project_id": "test-project"})

        assert response.status_code == 200
        _, kwargs = mock_memory.get_all.call_args
        assert kwargs["filters"] == PROJECT_FILTERS
        assert kwargs["top_k"] == 20

    def test_global_scope_filters(self, client, mock_memory):
        response = client.get("/memories", params={"scope": "global"})

        assert response.status_code == 200
        _, kwargs = mock_memory.get_all.call_args
        assert kwargs["filters"] == {"user_id": USER_ID, "scope_key": "global"}

    def test_top_k_forwarded(self, client, mock_memory):
        response = client.get(
            "/memories",
            params={"scope": "project", "project_id": "test-project", "top_k": 1000},
        )

        assert response.status_code == 200
        _, kwargs = mock_memory.get_all.call_args
        assert kwargs["top_k"] == 1000

    def test_rejects_top_k_above_limit(self, client, mock_memory):
        response = client.get(
            "/memories",
            params={"scope": "project", "project_id": "test-project", "top_k": 1001},
        )

        assert response.status_code == 422
        mock_memory.get_all.assert_not_called()

    def test_project_scope_without_project_id_rejected(self, client, mock_memory):
        response = client.get("/memories", params={"scope": "project"})

        assert response.status_code == 400
        mock_memory.get_all.assert_not_called()


# ===========================================================================
# Search validation errors from the core map to 400
# ===========================================================================

class TestSearchValidationErrors:
    """Verify that ValueError from Memory.search() returns 400, not 502."""

    def test_core_value_error_returns_400(self, client, mock_memory):
        mock_memory.search.side_effect = ValueError("query must not be empty")
        resp = client.post("/search", json={"query": "food", **SCOPE})
        assert resp.status_code == 400
        assert "query must not be empty" in resp.json()["detail"]


# ===========================================================================
# add / update / delete: map core errors to 4xx instead of 502
# ===========================================================================

class TestWriteHandlerErrorMapping:
    """ValueError("... not found") -> 404, other ValueError / Mem0ValidationError
    -> 400. A real outage still surfaces as 502 via upstream_error()."""

    def test_update_not_found_returns_404(self, client, mock_memory):
        mock_memory.update.side_effect = ValueError("Memory with id mem-1 not found")
        resp = client.put("/memories/mem-1", json={"text": "new"})
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_delete_not_found_returns_404(self, client, mock_memory):
        mock_memory.delete.side_effect = ValueError("Memory with id mem-1 not found")
        resp = client.delete("/memories/mem-1")
        assert resp.status_code == 404

    def test_update_other_value_error_returns_400(self, client, mock_memory):
        mock_memory.update.side_effect = ValueError("data must be a non-empty string")
        resp = client.put("/memories/mem-1", json={"text": "new"})
        assert resp.status_code == 400

    def test_add_validation_error_returns_400(self, client, mock_memory):
        mock_memory.add.side_effect = Mem0ValidationError(
            message="messages must be str, dict, or list[dict]", error_code="VALIDATION_003"
        )
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "hi"}], **SCOPE,
        })
        assert resp.status_code == 400

    def test_add_real_outage_still_returns_502(self, client, mock_memory):
        mock_memory.add.side_effect = RuntimeError("vector store unreachable")
        resp = client.post("/memories", json={
            "messages": [{"role": "user", "content": "hi"}], **SCOPE,
        })
        assert resp.status_code == 502
