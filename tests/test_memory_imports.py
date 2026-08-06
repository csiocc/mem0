import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db import Base
from memory_imports import (
    ImportConflict,
    append_memories,
    begin_import,
    finish_import,
)
from memory_scope import resolve_scope
from models import MemoryImportSource, User


class FakeMemory:
    def __init__(self):
        self.existing_hashes = set()
        self.add_calls = []

    def get_all(self, *, filters, top_k):
        content_hash = filters["import_content_sha256"]
        results = [{"id": "existing"}] if content_hash in self.existing_hashes else []
        return {"results": results[:top_k]}

    def add(self, *, messages, user_id, metadata, infer):
        self.add_calls.append(
            {
                "messages": messages,
                "user_id": user_id,
                "metadata": metadata,
                "infer": infer,
            }
        )
        self.existing_hashes.add(metadata["import_content_sha256"])
        return {"results": [{"id": f"mem-{len(self.add_calls)}"}]}


@pytest.fixture
def db():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
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
    try:
        yield session
    finally:
        session.close()


def test_completed_unchanged_source_is_skipped(db):
    target = resolve_scope("project", "LernendeLab")
    first = begin_import(
        db=db,
        user_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        target=target,
        source_ref="docs/memory.md",
        source_sha256="a" * 64,
    )
    first.source.status = "completed"
    db.commit()

    second = begin_import(
        db=db,
        user_id=first.source.user_id,
        target=target,
        source_ref="docs/memory.md",
        source_sha256="a" * 64,
    )

    assert second.state == "unchanged"
    assert second.source.id == first.source.id


def test_incomplete_source_resumes(db):
    target = resolve_scope("global", None)
    first = begin_import(
        db=db,
        user_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        target=target,
        source_ref="MEMORY.md",
        source_sha256="b" * 64,
    )
    first.source.failed_count = 2
    db.commit()

    second = begin_import(
        db=db,
        user_id=first.source.user_id,
        target=target,
        source_ref="MEMORY.md",
        source_sha256="b" * 64,
    )

    assert second.state == "resumed"
    assert second.source.id == first.source.id
    assert second.source.failed_count == 0


def test_begin_rejects_non_text_source(db):
    with pytest.raises(ImportConflict, match=r"\.md or \.txt"):
        begin_import(
            db=db,
            user_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
            target=resolve_scope("project", "LernendeLab"),
            source_ref="docs/memory.pdf",
            source_sha256="9" * 64,
        )


def test_append_forces_infer_false_and_stores_provenance(db):
    memory = FakeMemory()
    started = begin_import(
        db=db,
        user_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        target=resolve_scope("project", "LernendeLab"),
        source_ref="docs/memory.md",
        source_sha256="c" * 64,
    )

    result = append_memories(
        db=db,
        memory=memory,
        user_id=started.source.user_id,
        import_id=started.source.id,
        items=[{"text": "The project uses local Ollama embeddings.", "source_section": "Architecture"}],
    )

    assert result.stored == 1
    assert result.duplicates == 0
    call = memory.add_calls[0]
    assert call["infer"] is False
    assert call["user_id"] == str(started.source.user_id)
    assert call["metadata"]["scope_key"] == "project:LernendeLab"
    assert call["metadata"]["source_ref"] == "docs/memory.md"
    assert call["metadata"]["importer"] == "claude-code-haiku"


def test_duplicate_content_is_not_added_twice(db):
    memory = FakeMemory()
    started = begin_import(
        db=db,
        user_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        target=resolve_scope("global", None),
        source_ref="MEMORY.md",
        source_sha256="d" * 64,
    )
    item = {"text": "Use concise German guidance.", "source_section": "Preferences"}

    first = append_memories(
        db=db,
        memory=memory,
        user_id=started.source.user_id,
        import_id=started.source.id,
        items=[item],
    )
    second = append_memories(
        db=db,
        memory=memory,
        user_id=started.source.user_id,
        import_id=started.source.id,
        items=[item],
    )

    assert first.stored == 1
    assert second.duplicates == 1
    assert len(memory.add_calls) == 1


def test_foreign_user_cannot_append(db):
    started = begin_import(
        db=db,
        user_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        target=resolve_scope("global", None),
        source_ref="MEMORY.md",
        source_sha256="e" * 64,
    )

    with pytest.raises(ImportConflict, match="not found"):
        append_memories(
            db=db,
            memory=FakeMemory(),
            user_id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
            import_id=started.source.id,
            items=[{"text": "foreign", "source_section": None}],
        )


def test_finish_requires_zero_failures(db):
    started = begin_import(
        db=db,
        user_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        target=resolve_scope("global", None),
        source_ref="MEMORY.md",
        source_sha256="f" * 64,
    )
    started.source.failed_count = 1
    db.commit()

    with pytest.raises(ImportConflict, match="unresolved failures"):
        finish_import(
            db=db,
            user_id=started.source.user_id,
            import_id=started.source.id,
        )


def test_successful_retry_clears_current_unresolved_failures(db):
    memory = FakeMemory()
    started = begin_import(
        db=db,
        user_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        target=resolve_scope("global", None),
        source_ref="MEMORY.md",
        source_sha256="8" * 64,
    )

    failed = append_memories(
        db=db,
        memory=memory,
        user_id=started.source.user_id,
        import_id=started.source.id,
        items=[{"text": "", "source_section": "Preferences"}],
    )
    corrected = append_memories(
        db=db,
        memory=memory,
        user_id=started.source.user_id,
        import_id=started.source.id,
        items=[{"text": "Use concise guidance.", "source_section": "Preferences"}],
    )

    assert failed.failed == 1
    assert corrected.failed == 0
    assert corrected.source.failed_count == 0
    assert finish_import(
        db=db,
        user_id=started.source.user_id,
        import_id=started.source.id,
    ).status == "completed"


def test_changed_source_creates_new_append_only_import(db):
    target = resolve_scope("project", "LernendeLab")
    first = begin_import(
        db=db,
        user_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        target=target,
        source_ref="docs/memory.md",
        source_sha256="1" * 64,
    )
    second = begin_import(
        db=db,
        user_id=first.source.user_id,
        target=target,
        source_ref="docs/memory.md",
        source_sha256="2" * 64,
    )

    assert second.state == "started"
    assert second.source.id != first.source.id
    assert db.query(MemoryImportSource).count() == 2


def test_many_append_calls_have_no_total_limit(db):
    memory = FakeMemory()
    started = begin_import(
        db=db,
        user_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        target=resolve_scope("global", None),
        source_ref="large-memory.md",
        source_sha256="3" * 64,
    )

    for index in range(250):
        result = append_memories(
            db=db,
            memory=memory,
            user_id=started.source.user_id,
            import_id=started.source.id,
            items=[{"text": f"Durable memory number {index}.", "source_section": str(index)}],
        )
        assert result.stored == 1

    completed = finish_import(
        db=db,
        user_id=started.source.user_id,
        import_id=started.source.id,
    )
    assert completed.stored_count == 250
    assert completed.status == "completed"
