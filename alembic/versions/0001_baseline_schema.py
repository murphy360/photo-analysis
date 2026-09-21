"""baseline schema

Revision ID: 0001
Revises:
Create Date: 2026-09-21

Guarded (checks whether the table already exists) rather than a plain
op.create_table, specifically so this applies cleanly to *both* a brand new
database *and* an existing production one that was created by init_db()'s
create_all before this repo had any real migrations — the latter already has
this exact table, and a blind create_table would fail with "table already
exists" against it. Represents the schema as it stood before people_note /
provider_results / compare_providers existed; migration 0002 adds those.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "analysis_jobs" in inspector.get_table_names():
        return

    op.create_table(
        "analysis_jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column(
            "media_type", sa.Enum("IMAGE", "VIDEO", name="mediatype"), nullable=False
        ),
        sa.Column("media_path", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column(
            "status",
            sa.Enum("PENDING", "RUNNING", "COMPLETED", "FAILED", name="jobstatus"),
            nullable=False,
        ),
        sa.Column(
            "tier_requested",
            sa.Enum("SKIP", "CHEAP", "STANDARD", "THOROUGH", name="analysistier"),
            nullable=True,
        ),
        sa.Column(
            "tier_used",
            sa.Enum("SKIP", "CHEAP", "STANDARD", "THOROUGH", name="analysistier"),
            nullable=True,
        ),
        sa.Column("callback_url", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("triage_objects", sa.JSON(), nullable=True),
        sa.Column("people", sa.JSON(), nullable=True),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("description_providers", sa.JSON(), nullable=True),
        sa.Column("budget_note", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("error", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_analysis_jobs_created_at"), "analysis_jobs", ["created_at"], unique=False
    )
    op.create_index(op.f("ix_analysis_jobs_source"), "analysis_jobs", ["source"], unique=False)
    op.create_index(op.f("ix_analysis_jobs_status"), "analysis_jobs", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_analysis_jobs_status"), table_name="analysis_jobs")
    op.drop_index(op.f("ix_analysis_jobs_source"), table_name="analysis_jobs")
    op.drop_index(op.f("ix_analysis_jobs_created_at"), table_name="analysis_jobs")
    op.drop_table("analysis_jobs")
