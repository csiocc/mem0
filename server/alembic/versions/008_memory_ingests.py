"""Create the async memory ingest queue.

Revision ID: occ008
Revises: occ007
Create Date: 2026-08-15
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# Fork-prefixed revision ID: upstream numbers its migrations 001, 002, ...
# and already stands at 006 while the fork occupies 007. A prefixed ID can
# never collide with a future upstream "008" — worst case is two Alembic
# heads at sync time, resolved with one `alembic merge` revision.
revision: str = "occ008"
down_revision: Union[str, None] = "occ007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_ingests",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("bound_metadata", sa.JSON(), nullable=False),
        sa.Column("scope", sa.String(length=20), nullable=False),
        sa.Column("project_id", sa.String(length=255), nullable=True),
        sa.Column("scope_key", sa.String(length=300), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.String(length=128), nullable=True),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'failed')",
            name="ck_memory_ingests_status",
        ),
    )
    op.create_index("ix_memory_ingests_user_id", "memory_ingests", ["user_id"])
    op.create_index("ix_memory_ingests_status", "memory_ingests", ["status"])
    op.create_index("ix_memory_ingests_next_attempt_at", "memory_ingests", ["next_attempt_at"])


def downgrade() -> None:
    op.drop_table("memory_ingests")
