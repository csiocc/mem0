from sqlalchemy import CheckConstraint, UniqueConstraint

from models import MemoryImportSource


def test_import_source_has_required_columns():
    columns = MemoryImportSource.__table__.columns

    assert set(columns.keys()) == {
        "id",
        "user_id",
        "scope",
        "project_id",
        "scope_key",
        "source_ref",
        "source_sha256",
        "status",
        "stored_count",
        "duplicate_count",
        "failed_count",
        "created_at",
        "completed_at",
    }
    assert not columns.user_id.nullable
    assert not columns.scope_key.nullable
    assert not columns.source_sha256.nullable


def test_import_source_has_source_idempotency_constraint():
    constraints = MemoryImportSource.__table__.constraints
    unique_columns = {
        tuple(column.name for column in constraint.columns)
        for constraint in constraints
        if isinstance(constraint, UniqueConstraint)
    }

    assert (
        "user_id",
        "scope_key",
        "source_ref",
        "source_sha256",
    ) in unique_columns


def test_import_source_checks_scope_and_status():
    sql = " ".join(
        str(constraint.sqltext)
        for constraint in MemoryImportSource.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    )

    assert "global" in sql
    assert "project" in sql
    assert "in_progress" in sql
    assert "completed" in sql
    assert "failed" in sql
