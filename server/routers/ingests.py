"""Owner-scoped visibility into the async memory ingest queue.

Responses never include payload or bound metadata: raw excerpt content
stays inside the queue table until the worker deletes it.
"""

import uuid
from datetime import datetime
from typing import Optional

from auth import require_auth
from db import get_db
from fastapi import APIRouter, Depends, HTTPException, Query
from ingest_worker import _utcnow, wake_event
from models import MemoryIngest, User
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

router = APIRouter(prefix="/memory-ingests", tags=["memory-ingests"])

LIST_LIMIT = 100


class MemoryIngestItem(BaseModel):
    id: uuid.UUID
    status: str
    attempts: int
    scope: str
    project_id: Optional[str]
    last_error: Optional[str]
    request_id: Optional[str]
    created_at: datetime
    next_attempt_at: datetime


def _item(row: MemoryIngest) -> MemoryIngestItem:
    return MemoryIngestItem(
        id=row.id,
        status=row.status,
        attempts=row.attempts,
        scope=row.scope,
        project_id=row.project_id,
        last_error=row.last_error,
        request_id=row.request_id,
        created_at=row.created_at,
        next_attempt_at=row.next_attempt_at,
    )


def _owned_or_404(db: Session, ingest_id: uuid.UUID, user: User) -> MemoryIngest:
    row = db.get(MemoryIngest, ingest_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(status_code=404, detail="Ingest not found.")
    return row


@router.get("", response_model=list[MemoryIngestItem])
def list_ingests(
    status: Optional[str] = Query(None, pattern="^(pending|processing|failed)$"),
    db: Session = Depends(get_db),
    user: User = Depends(require_auth),
):
    query = (
        select(MemoryIngest)
        .where(MemoryIngest.user_id == user.id)
        .order_by(MemoryIngest.created_at.desc())
        .limit(LIST_LIMIT)
    )
    if status:
        query = query.where(MemoryIngest.status == status)
    rows = db.execute(query).scalars().all()
    return [_item(row) for row in rows]


@router.get("/{ingest_id}", response_model=MemoryIngestItem)
def get_ingest(
    ingest_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(require_auth),
):
    return _item(_owned_or_404(db, ingest_id, user))


@router.post("/{ingest_id}/retry", response_model=MemoryIngestItem)
def retry_ingest(
    ingest_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(require_auth),
):
    row = _owned_or_404(db, ingest_id, user)
    if row.status != "failed":
        raise HTTPException(status_code=409, detail="Only failed ingests can be retried.")
    row.status = "pending"
    row.attempts = 0
    row.last_error = None
    row.claimed_at = None
    row.next_attempt_at = _utcnow()
    db.commit()
    wake_event.set()
    return _item(row)
