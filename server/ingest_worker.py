"""Persistent background processing for deferred memory writes.

POST /memories with {"async": true} only validates and persists the payload;
this module's worker thread performs the actual Memory.add with retry and
backoff. Success deletes the row so raw excerpt content never outlives its
processing; failures stay visible (and retryable) for 30 days. Log lines
carry ingest IDs and exception class names, never payload content.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from memory_scope import scope_run_id
from models import MemoryIngest
from server_state import get_memory_instance
from sqlalchemy import delete, select, update

from mem0.exceptions import ValidationError as Mem0ValidationError

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 5
BACKOFF_SECONDS = (60, 300, 900, 3600)
CLAIM_BATCH = 10
RECLAIM_AFTER_SECONDS = 15 * 60
PURGE_FAILED_AFTER_SECONDS = 30 * 24 * 3600
IDLE_WAIT_SECONDS = 5.0
DB_ERROR_WAIT_SECONDS = 30.0
PERMANENT_ERRORS = (Mem0ValidationError, ValueError)

wake_event = threading.Event()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def enqueue_ingest(
    session_factory,
    *,
    user_id: uuid.UUID,
    payload: dict[str, Any],
    bound_metadata: dict[str, Any],
    scope: str,
    project_id: str | None,
    scope_key: str,
    request_id: str | None,
) -> uuid.UUID:
    row = MemoryIngest(
        user_id=user_id,
        payload=payload,
        bound_metadata=bound_metadata,
        scope=scope,
        project_id=project_id,
        scope_key=scope_key,
        request_id=request_id,
    )
    with session_factory() as session:
        session.add(row)
        session.commit()
        ingest_id = row.id
    wake_event.set()
    return ingest_id


def _claim(session, ingest_id: uuid.UUID) -> bool:
    """Optimistic claim: portable to SQLite in tests, no long transactions."""
    result = session.execute(
        update(MemoryIngest)
        .where(MemoryIngest.id == ingest_id, MemoryIngest.status == "pending")
        .values(status="processing", claimed_at=_utcnow())
    )
    session.commit()
    return result.rowcount == 1


def _record_failure(
    session_factory, ingest_id: uuid.UUID, previous_attempts: int, exc: Exception
) -> None:
    attempts = previous_attempts + 1
    permanent = isinstance(exc, PERMANENT_ERRORS) or attempts >= MAX_ATTEMPTS
    backoff = BACKOFF_SECONDS[min(attempts - 1, len(BACKOFF_SECONDS) - 1)]
    status = "failed" if permanent else "pending"
    with session_factory() as session:
        session.execute(
            update(MemoryIngest)
            .where(MemoryIngest.id == ingest_id)
            .values(
                status=status,
                attempts=attempts,
                last_error=type(exc).__name__[:128],
                next_attempt_at=_utcnow() + timedelta(seconds=backoff),
                claimed_at=None,
            )
        )
        session.commit()
    logger.warning(
        "memory ingest %s %s (%s, attempt %d)", ingest_id, status, type(exc).__name__, attempts
    )


def _process_claimed(session_factory, ingest_id: uuid.UUID) -> None:
    # Read what the add needs, then close the session: Memory.add can take
    # minutes and must not hold a database transaction open meanwhile.
    with session_factory() as session:
        row = session.get(MemoryIngest, ingest_id)
        if row is None:
            return
        messages = list((row.payload or {}).get("messages") or [])
        params = dict((row.payload or {}).get("params") or {})
        metadata = dict(row.bound_metadata or {})
        mem0_user_id = str(row.user_id)
        previous_attempts = row.attempts
    try:
        get_memory_instance().add(
            messages=messages,
            user_id=mem0_user_id,
            run_id=scope_run_id(metadata),
            metadata=metadata,
            **params,
        )
    except Exception as exc:  # noqa: BLE001 - every outcome is recorded per row
        _record_failure(session_factory, ingest_id, previous_attempts, exc)
        return
    with session_factory() as session:
        session.execute(delete(MemoryIngest).where(MemoryIngest.id == ingest_id))
        session.commit()
    logger.info("memory ingest %s done", ingest_id)


def run_pending(session_factory) -> int:
    """Process every due pending row once. Returns the processed count."""
    processed = 0
    while True:
        with session_factory() as session:
            candidates = (
                session.execute(
                    select(MemoryIngest.id)
                    .where(
                        MemoryIngest.status == "pending",
                        MemoryIngest.next_attempt_at <= _utcnow(),
                    )
                    .order_by(MemoryIngest.created_at)
                    .limit(CLAIM_BATCH)
                )
                .scalars()
                .all()
            )
        if not candidates:
            return processed
        for ingest_id in candidates:
            with session_factory() as session:
                claimed = _claim(session, ingest_id)
            if claimed:
                _process_claimed(session_factory, ingest_id)
                processed += 1


def run_maintenance(session_factory) -> None:
    """Reclaim rows orphaned by a crashed worker; purge old failed rows."""
    now = _utcnow()
    with session_factory() as session:
        session.execute(
            update(MemoryIngest)
            .where(
                MemoryIngest.status == "processing",
                MemoryIngest.claimed_at < now - timedelta(seconds=RECLAIM_AFTER_SECONDS),
            )
            .values(status="pending", claimed_at=None)
        )
        session.execute(
            delete(MemoryIngest).where(
                MemoryIngest.status == "failed",
                MemoryIngest.created_at < now - timedelta(seconds=PURGE_FAILED_AFTER_SECONDS),
            )
        )
        session.commit()


def start_worker(session_factory) -> tuple[threading.Thread, threading.Event]:
    stop = threading.Event()

    def _loop() -> None:
        while not stop.is_set():
            try:
                run_maintenance(session_factory)
                run_pending(session_factory)
                wait = IDLE_WAIT_SECONDS
            except Exception:  # noqa: BLE001 - the worker must survive DB/provider outages
                logger.exception("memory ingest worker iteration failed")
                wait = DB_ERROR_WAIT_SECONDS
            wake_event.wait(wait)
            wake_event.clear()

    thread = threading.Thread(target=_loop, name="memory-ingest-worker", daemon=True)
    thread.start()
    return thread, stop
