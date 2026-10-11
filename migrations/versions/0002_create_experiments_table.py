"""Create M3 research schema: experiments

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-08 00:00:00

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "experiments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("experiment_id", sa.String(length=64), nullable=False),
        sa.Column("experiment_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column("strategy_name", sa.String(length=128), nullable=False),
        sa.Column("instrument_symbol", sa.String(length=32), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("config_hash", sa.String(length=64), nullable=False),
        sa.Column("experiment_name", sa.String(length=255), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "parent_experiment_id",
            sa.String(length=64),
            sa.ForeignKey("experiments.experiment_id"),
            nullable=True,
        ),
        sa.Column("result_summary", sa.JSON(), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_experiments_experiment_id", "experiments", ["experiment_id"], unique=True)
    op.create_index("ix_experiments_strategy_name", "experiments", ["strategy_name"], unique=False)
    op.create_index("ix_experiments_instrument_symbol", "experiments", ["instrument_symbol"], unique=False)
    op.create_index("ix_experiments_config_hash", "experiments", ["config_hash"], unique=False)
    op.create_index("ix_experiments_parent_experiment_id", "experiments", ["parent_experiment_id"], unique=False)
    op.create_index(
        "ix_experiments_strategy_instrument",
        "experiments",
        ["strategy_name", "instrument_symbol"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_experiments_strategy_instrument", table_name="experiments")
    op.drop_index("ix_experiments_parent_experiment_id", table_name="experiments")
    op.drop_index("ix_experiments_config_hash", table_name="experiments")
    op.drop_index("ix_experiments_instrument_symbol", table_name="experiments")
    op.drop_index("ix_experiments_strategy_name", table_name="experiments")
    op.drop_index("ix_experiments_experiment_id", table_name="experiments")
    op.drop_table("experiments")
