"""merge user-data and monitoring migration chains

Revision ID: 05f69a2e3bc6
Revises: 5c3180f8767a, 9196f74046ea
Create Date: 2026-06-08 16:40:16.893799

"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "05f69a2e3bc6"
down_revision: Union[str, Sequence[str], None] = ("5c3180f8767a", "9196f74046ea")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
