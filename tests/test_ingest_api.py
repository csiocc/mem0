import uuid

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from auth import require_auth
from db import Base, get_db
from models import MemoryIngest, User
from routers import ingests

OWNER_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
OTHER_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


@pytest.fixture
def client():
    # TestClient serves the app from a worker thread; share one connection.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = factory()
    for user_id, email in ((OWNER_ID, "csi@example.test"), (OTHER_ID, "other@example.test")):
        session.add(User(id=user_id, name="u", email=email, password_hash="unused", role="user"))
    session.commit()

    app = FastAPI()
    app.include_router(ingests.router)
    owner = session.get(User, OWNER_ID)
    app.dependency_overrides[require_auth] = lambda: owner
    app.dependency_overrides[get_db] = lambda: session
    yield TestClient(app), session
    session.close()


def _row(session, *, user_id=OWNER_ID, status="pending", last_error=None):
    row = MemoryIngest(
        user_id=user_id,
        payload={"messages": [{"role": "user", "content": "secret excerpt"}], "params": {}},
        bound_metadata={"scope_key": "project:occ"},
        scope="project",
        project_id="occ",
        scope_key="project:occ",
        status=status,
        last_error=last_error,
        request_id="req-1",
    )
    session.add(row)
    session.commit()
    return row


def test_list_is_owner_scoped_and_content_free(client):
    http, session = client
    mine = _row(session)
    _row(session, user_id=OTHER_ID)

    response = http.get("/memory-ingests")

    assert response.status_code == 200
    items = response.json()
    assert [item["id"] for item in items] == [str(mine.id)]
    assert "payload" not in items[0] and "bound_metadata" not in items[0]
    assert "secret excerpt" not in response.text


def test_list_filters_by_status(client):
    http, session = client
    _row(session, status="pending")
    failed = _row(session, status="failed", last_error="ConnectionError")

    response = http.get("/memory-ingests", params={"status": "failed"})

    assert [item["id"] for item in response.json()] == [str(failed.id)]
    assert response.json()[0]["last_error"] == "ConnectionError"


def test_get_foreign_row_is_404(client):
    http, session = client
    foreign = _row(session, user_id=OTHER_ID)

    assert http.get(f"/memory-ingests/{foreign.id}").status_code == 404


def test_retry_resets_a_failed_row(client):
    http, session = client
    failed = _row(session, status="failed", last_error="ConnectionError")

    response = http.post(f"/memory-ingests/{failed.id}/retry")

    assert response.status_code == 200
    session.refresh(failed)
    assert failed.status == "pending"
    assert failed.attempts == 0
    assert failed.last_error is None


def test_retry_rejects_non_failed_rows(client):
    http, session = client
    pending = _row(session, status="pending")

    assert http.post(f"/memory-ingests/{pending.id}/retry").status_code == 409
