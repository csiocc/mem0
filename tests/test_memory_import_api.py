import uuid
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import require_auth
from db import get_db
from models import User
from routers import memory_imports


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(memory_imports.router)
    user = User(
        id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        name="CSI",
        email="csi@example.test",
        password_hash="unused",
        role="user",
    )
    db = MagicMock()
    app.dependency_overrides[require_auth] = lambda: user
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app), db, user


def test_begin_uses_authenticated_user(client, monkeypatch):
    http, db, user = client
    source = MagicMock(
        id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        stored_count=0,
        duplicate_count=0,
        failed_count=0,
    )
    begin = MagicMock(state="started", source=source)
    called = {}

    def fake_begin_import(**kwargs):
        called.update(kwargs)
        return begin

    monkeypatch.setattr(memory_imports, "begin_import", fake_begin_import)

    response = http.post(
        "/memory-imports",
        json={
            "source_ref": "docs/memory.md",
            "source_sha256": "a" * 64,
            "scope": "project",
            "project_id": "LernendeLab",
        },
    )

    assert response.status_code == 200
    assert called["user_id"] == user.id
    assert "user_id" not in response.request.content.decode()


def test_append_rejects_foreign_import(client, monkeypatch):
    http, _, _ = client
    monkeypatch.setattr(memory_imports, "get_memory_instance", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        memory_imports,
        "append_memories",
        MagicMock(side_effect=memory_imports.ImportConflict("Memory import not found.")),
    )

    response = http.post(
        "/memory-imports/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/memories",
        json={"memories": [{"text": "secret", "source_section": "A"}]},
    )

    assert response.status_code == 404


def test_finish_returns_final_counts(client, monkeypatch):
    http, _, _ = client
    source = MagicMock(
        id=uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        status="completed",
        stored_count=8,
        duplicate_count=2,
        failed_count=0,
    )
    monkeypatch.setattr(memory_imports, "finish_import", MagicMock(return_value=source))

    response = http.post(
        "/memory-imports/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/complete"
    )

    assert response.status_code == 200
    assert response.json()["stored_count"] == 8
    assert response.json()["duplicate_count"] == 2
