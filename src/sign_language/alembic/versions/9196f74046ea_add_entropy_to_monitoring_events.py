"""add entropy to monitoring events

Revision ID: 9196f74046ea
Revises: 90aaa54c87cc
Create Date: 2026-06-03 13:45:20.759269

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9196f74046ea"
down_revision: Union[str, None] = "90aaa54c87cc"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "monitoring_events",
        sa.Column("entropy", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("monitoring_events", "entropy")
