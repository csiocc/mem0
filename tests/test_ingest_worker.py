import uuid

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db import Base
from models import MemoryIngest, User


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with factory() as session:
        session.add(
            User(
                id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                name="CSI",
                email="csi@example.test",
                password_hash="unused",
                role="user",
            )
        )
        session.commit()
    return factory


def _enqueue(factory, **overrides):
    import ingest_worker

    values = dict(
        user_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        payload={"messages": [{"role": "user", "content": "probe"}], "params": {"infer": True}},
        bound_metadata={"scope_key": "project:occ"},
        scope="project",
        project_id="occ",
        scope_key="project:occ",
        request_id="req-1",
    )
    values.update(overrides)
    return ingest_worker.enqueue_ingest(factory, **values)


def test_ingest_row_defaults(session_factory):
    ingest_id = _enqueue(session_factory)

    with session_factory() as session:
        row = session.get(MemoryIngest, ingest_id)
        assert row.status == "pending"
        assert row.attempts == 0
        assert row.claimed_at is None
        assert row.last_error is None
        assert row.payload["params"] == {"infer": True}
