import uuid

from auth import require_auth
from db import get_db
from fastapi import APIRouter, Depends, HTTPException
from memory_imports import (
    ImportConflict,
    append_memories,
    begin_import,
    finish_import,
)
from memory_scope import ScopeError, resolve_scope
from models import User
from schemas import (
    MemoryImportAppendRequest,
    MemoryImportBeginRequest,
    MemoryImportStatusResponse,
)
from server_state import get_memory_instance
from sqlalchemy.orm import Session

router = APIRouter(prefix="/memory-imports", tags=["memory-imports"])


def _response(source, *, state: str, errors: list[dict] | None = None):
    return MemoryImportStatusResponse(
        state=state,
        import_id=source.id,
        stored_count=source.stored_count,
        duplicate_count=source.duplicate_count,
        failed_count=source.failed_count,
        errors=errors or [],
    )


@router.post("", response_model=MemoryImportStatusResponse)
def begin(
    request: MemoryImportBeginRequest,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    try:
        target = resolve_scope(request.scope, request.project_id)
        result = begin_import(
            db=db,
            user_id=user.id,
            target=target,
            source_ref=request.source_ref,
            source_sha256=request.source_sha256,
        )
        return _response(result.source, state=result.state)
    except (ScopeError, ImportConflict) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/{import_id}/memories", response_model=MemoryImportStatusResponse)
def append(
    import_id: uuid.UUID,
    request: MemoryImportAppendRequest,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    try:
        result = append_memories(
            db=db,
            memory=get_memory_instance(),
            user_id=user.id,
            import_id=import_id,
            items=[item.model_dump() for item in request.memories],
        )
        state = "failed" if result.failed else "in_progress"
        return _response(result.source, state=state, errors=result.errors)
    except ImportConflict as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/{import_id}/complete", response_model=MemoryImportStatusResponse)
def complete(
    import_id: uuid.UUID,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    try:
        source = finish_import(db=db, user_id=user.id, import_id=import_id)
        return _response(source, state="completed")
    except ImportConflict as exc:
        status = 404 if "not found" in str(exc).lower() else 409
        raise HTTPException(status_code=status, detail=str(exc))
