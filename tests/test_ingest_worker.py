import uuid
from datetime import timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import ingest_worker
from db import Base
from ingest_worker import _utcnow
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


def _run_with_memory(factory, memory):
    with patch.object(ingest_worker, "get_memory_instance", return_value=memory):
        return ingest_worker.run_pending(factory)


def test_successful_ingest_deletes_the_row(session_factory):
    ingest_id = _enqueue(session_factory)
    memory = MagicMock()

    processed = _run_with_memory(session_factory, memory)

    assert processed == 1
    memory.add.assert_called_once_with(
        messages=[{"role": "user", "content": "probe"}],
        user_id="11111111-1111-1111-1111-111111111111",
        run_id="project:occ",
        metadata={"scope_key": "project:occ"},
        infer=True,
    )
    with session_factory() as session:
        assert session.get(MemoryIngest, ingest_id) is None


def test_inferred_ingest_stays_inside_its_scope(session_factory):
    """Deferred writes are the inferred ones, and inference is where scopes leak.

    mem0 builds both its recent-message context and its deduplication search
    from user/agent/run identifiers only, dropping scope_key. Without a
    run_id an inferred write sees the user's messages and memories from every
    other project and can extract their content into this scope.
    """
    _enqueue(session_factory, bound_metadata={"scope_key": "global"}, scope="global",
             project_id=None, scope_key="global")
    memory = MagicMock()

    _run_with_memory(session_factory, memory)

    assert memory.add.call_args.kwargs["run_id"] == "global"


def test_transient_failure_backs_off_and_stays_pending(session_factory):
    ingest_id = _enqueue(session_factory)
    memory = MagicMock()
    memory.add.side_effect = ConnectionError("provider down")

    _run_with_memory(session_factory, memory)

    with session_factory() as session:
        row = session.get(MemoryIngest, ingest_id)
        assert row.status == "pending"
        assert row.attempts == 1
        assert row.last_error == "ConnectionError"
        # SQLite returns naive datetimes; Postgres (production) keeps the offset.
        assert row.next_attempt_at.replace(tzinfo=timezone.utc) > _utcnow()
        assert row.payload["messages"]  # payload survives for the retry


def test_permanent_failure_fails_immediately(session_factory):
    from mem0.exceptions import ValidationError as Mem0ValidationError

    ingest_id = _enqueue(session_factory)
    memory = MagicMock()
    memory.add.side_effect = Mem0ValidationError("bad input", "VAL_001")

    _run_with_memory(session_factory, memory)

    with session_factory() as session:
        row = session.get(MemoryIngest, ingest_id)
        assert row.status == "failed"
        assert row.attempts == 1


def test_attempt_cap_marks_failed(session_factory):
    ingest_id = _enqueue(session_factory)
    memory = MagicMock()
    memory.add.side_effect = ConnectionError("still down")

    for _ in range(ingest_worker.MAX_ATTEMPTS):
        with session_factory() as session:
            session.execute(
                ingest_worker.update(MemoryIngest)
                .where(MemoryIngest.id == ingest_id)
                .values(next_attempt_at=_utcnow() - timedelta(seconds=1))
            )
            session.commit()
        _run_with_memory(session_factory, memory)

    with session_factory() as session:
        row = session.get(MemoryIngest, ingest_id)
        assert row.status == "failed"
        assert row.attempts == ingest_worker.MAX_ATTEMPTS


def test_claimed_row_is_not_processed_twice(session_factory):
    ingest_id = _enqueue(session_factory)
    with session_factory() as session:
        session.execute(
            ingest_worker.update(MemoryIngest)
            .where(MemoryIngest.id == ingest_id)
            .values(status="processing", claimed_at=_utcnow())
        )
        session.commit()
    memory = MagicMock()

    processed = _run_with_memory(session_factory, memory)

    assert processed == 0
    memory.add.assert_not_called()


def test_maintenance_reclaims_stale_and_purges_old_failed(session_factory):
    stale_id = _enqueue(session_factory)
    old_failed_id = _enqueue(session_factory)
    with session_factory() as session:
        session.execute(
            ingest_worker.update(MemoryIngest)
            .where(MemoryIngest.id == stale_id)
            .values(
                status="processing",
                claimed_at=_utcnow() - timedelta(seconds=ingest_worker.RECLAIM_AFTER_SECONDS + 60),
            )
        )
        session.execute(
            ingest_worker.update(MemoryIngest)
            .where(MemoryIngest.id == old_failed_id)
            .values(
                status="failed",
                created_at=_utcnow() - timedelta(seconds=ingest_worker.PURGE_FAILED_AFTER_SECONDS + 60),
            )
        )
        session.commit()

    ingest_worker.run_maintenance(session_factory)

    with session_factory() as session:
        assert session.get(MemoryIngest, stale_id).status == "pending"
        assert session.get(MemoryIngest, old_failed_id) is None
