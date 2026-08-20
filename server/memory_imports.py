from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from memory_scope import MemoryScope
from models import MemoryImportSource
from sqlalchemy import select
from sqlalchemy.orm import Session
from write_guard import InjectedInstructionError, reject_override_instructions


class ImportConflict(ValueError):
    pass


@dataclass(frozen=True)
class BeginResult:
    state: str
    source: MemoryImportSource


@dataclass(frozen=True)
class AppendResult:
    stored: int
    duplicates: int
    failed: int
    errors: list[dict[str, Any]]
    source: MemoryImportSource


def canonicalize_memory_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", normalized).strip()


def content_sha256(text: str) -> str:
    return hashlib.sha256(canonicalize_memory_text(text).encode("utf-8")).hexdigest()


def _owned_source(
    db: Session,
    *,
    user_id: uuid.UUID,
    import_id: uuid.UUID,
) -> MemoryImportSource:
    source = db.get(MemoryImportSource, import_id)
    if source is None or source.user_id != user_id:
        raise ImportConflict("Memory import not found.")
    return source


def begin_import(
    *,
    db: Session,
    user_id: uuid.UUID,
    target: MemoryScope,
    source_ref: str,
    source_sha256: str,
) -> BeginResult:
    safe_ref = source_ref.strip()
    if (
        not safe_ref
        or safe_ref.startswith("/")
        or ".." in safe_ref.split("/")
        or "\\" in safe_ref
        or ":" in safe_ref.split("/", 1)[0]
    ):
        raise ImportConflict("source_ref must be a safe relative path.")
    if not safe_ref.lower().endswith((".md", ".txt")):
        raise ImportConflict("source_ref must identify a .md or .txt file.")
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ImportConflict("source_sha256 must be a lowercase SHA-256 value.")

    source = db.scalar(
        select(MemoryImportSource).where(
            MemoryImportSource.user_id == user_id,
            MemoryImportSource.scope_key == target.scope_key,
            MemoryImportSource.source_ref == safe_ref,
            MemoryImportSource.source_sha256 == source_sha256,
        )
    )
    if source is not None:
        if source.status == "completed":
            return BeginResult(state="unchanged", source=source)
        source.status = "in_progress"
        source.failed_count = 0
        db.commit()
        return BeginResult(state="resumed", source=source)

    source = MemoryImportSource(
        user_id=user_id,
        scope=target.scope,
        project_id=target.project_id,
        scope_key=target.scope_key,
        source_ref=safe_ref,
        source_sha256=source_sha256,
        status="in_progress",
    )
    db.add(source)
    db.commit()
    db.refresh(source)
    return BeginResult(state="started", source=source)


def _results(value: Any) -> list[Any]:
    if isinstance(value, dict):
        result = value.get("results", [])
        return result if isinstance(result, list) else []
    return value if isinstance(value, list) else []


def append_memories(
    *,
    db: Session,
    memory: Any,
    user_id: uuid.UUID,
    import_id: uuid.UUID,
    items: list[dict[str, Any]],
) -> AppendResult:
    source = _owned_source(db, user_id=user_id, import_id=import_id)
    if source.status == "completed":
        raise ImportConflict("Completed imports cannot accept more memories.")

    stored = 0
    duplicates = 0
    failed = 0
    errors: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        text = canonicalize_memory_text(str(item.get("text", "")))
        if not text:
            failed += 1
            errors.append({"index": index, "code": "empty_memory"})
            continue

        # Imported files are untrusted input, so a line written to redirect a
        # later reader is refused here rather than stored verbatim.
        try:
            reject_override_instructions(text)
        except InjectedInstructionError:
            failed += 1
            errors.append({"index": index, "code": "injected_instructions"})
            continue

        digest = content_sha256(text)
        filters = {
            "user_id": str(user_id),
            "scope_key": source.scope_key,
            "import_content_sha256": digest,
        }
        if _results(memory.get_all(filters=filters, top_k=1)):
            duplicates += 1
            continue

        metadata = {
            "scope": source.scope,
            "project_id": source.project_id,
            "scope_key": source.scope_key,
            "import_id": str(source.id),
            "source_ref": source.source_ref,
            "source_sha256": source.source_sha256,
            "source_section": item.get("source_section"),
            "import_content_sha256": digest,
            "importer": "claude-code-haiku",
        }
        try:
            memory.add(
                messages=[{"role": "user", "content": text}],
                user_id=str(user_id),
                metadata=metadata,
                infer=False,
            )
            stored += 1
        except Exception:
            # Storage failures are counted without logging memory text.
            failed += 1
            errors.append({"index": index, "code": "storage_failed"})

    source.stored_count += stored
    source.duplicate_count += duplicates
    # failed_count tracks unresolved failures from the latest append attempt,
    # so a successful correction clears it.
    source.failed_count = failed
    source.status = "failed" if failed else "in_progress"
    db.commit()

    return AppendResult(
        stored=stored,
        duplicates=duplicates,
        failed=failed,
        errors=errors,
        source=source,
    )


def finish_import(
    *,
    db: Session,
    user_id: uuid.UUID,
    import_id: uuid.UUID,
) -> MemoryImportSource:
    source = _owned_source(db, user_id=user_id, import_id=import_id)
    if source.failed_count:
        raise ImportConflict("Memory import has unresolved failures.")

    source.status = "completed"
    source.completed_at = datetime.now(timezone.utc)
    db.commit()
    return source
