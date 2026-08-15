"""Create memory import source ledger.

Revision ID: 007
Revises: 006
Create Date: 2026-08-06
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# Fork-prefixed revision ID (renamed from "007"): upstream migration numbering
# stands at 006, so a bare "007" would collide with the next upstream
# migration's revision ID and stop Alembic from booting at all. Existing
# deployments need a one-time fixup before upgrading:
#   UPDATE alembic_version SET version_num = 'occ007' WHERE version_num = '007';
revision: str = "occ007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_import_sources",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("scope", sa.String(length=20), nullable=False),
        sa.Column("project_id", sa.String(length=255), nullable=True),
        sa.Column("scope_key", sa.String(length=300), nullable=False),
        sa.Column("source_ref", sa.String(length=1024), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default="in_progress",
        ),
        sa.Column("stored_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("duplicate_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "scope IN ('global', 'project')",
            name="ck_memory_import_sources_scope",
        ),
        sa.CheckConstraint(
            "status IN ('in_progress', 'completed', 'failed')",
            name="ck_memory_import_sources_status",
        ),
        sa.UniqueConstraint(
            "user_id",
            "scope_key",
            "source_ref",
            "source_sha256",
            name="uq_memory_import_sources_identity",
        ),
    )
    op.create_index(
        "ix_memory_import_sources_user_id",
        "memory_import_sources",
        ["user_id"],
    )
    op.create_index(
        "ix_memory_import_sources_user_status",
        "memory_import_sources",
        ["user_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_memory_import_sources_user_status",
        table_name="memory_import_sources",
    )
    op.drop_index(
        "ix_memory_import_sources_user_id",
        table_name="memory_import_sources",
    )
    op.drop_table("memory_import_sources")
