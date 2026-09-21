"""add people_note, provider_results, compare_providers

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-21

The three columns this repo's model gained after the baseline in 0001.
Guarded per-column (not just per-migration) so this is safe to re-run
against a database that already has some of them — e.g. one that was
manually patched with an ALTER TABLE before this migration existed, the way
photo-analysis's own production database was the first time this bug hit.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_NEW_COLUMNS = [
    sa.Column("people_note", sa.String(), nullable=True),
    sa.Column("provider_results", sa.JSON(), nullable=False, server_default="[]"),
    sa.Column("compare_providers", sa.Boolean(), nullable=False, server_default=sa.false()),
]


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {c["name"] for c in inspector.get_columns("analysis_jobs")}
    for column in _NEW_COLUMNS:
        if column.name not in existing:
            op.add_column("analysis_jobs", column.copy())


def downgrade() -> None:
    for column in _NEW_COLUMNS:
        op.drop_column("analysis_jobs", column.name)
