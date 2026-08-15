"""Persistent background processing for deferred memory writes."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from models import MemoryIngest

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
