"""Add word limits to rubrics

Adds nullable min_words/max_words to rubrics. Null means no limit, which is
what every rubric created before this keeps -- nothing existing is changed and
no submission already recorded is affected.

The three CHECK constraints were added by hand: Alembic's autogenerate does not
detect table-level CheckConstraints, so without them the model and the database
would disagree. They stop a range that no submission could ever satisfy
(negative, zero-length, or min above max).

Revision ID: 68d10bab5603
Revises: 37d50225c138
Create Date: 2026-10-05 13:40:23.219645

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '68d10bab5603'
down_revision = '37d50225c138'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('rubrics', schema=None) as batch_op:
        batch_op.add_column(sa.Column('min_words', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('max_words', sa.Integer(), nullable=True))
        batch_op.create_check_constraint('ck_rubrics_min_words', 'min_words is null or min_words >= 0')
        batch_op.create_check_constraint('ck_rubrics_max_words', 'max_words is null or max_words >= 1')
        batch_op.create_check_constraint(
            'ck_rubrics_word_range',
            'min_words is null or max_words is null or min_words <= max_words',
        )


def downgrade():
    with op.batch_alter_table('rubrics', schema=None) as batch_op:
        batch_op.drop_constraint('ck_rubrics_word_range', type_='check')
        batch_op.drop_constraint('ck_rubrics_max_words', type_='check')
        batch_op.drop_constraint('ck_rubrics_min_words', type_='check')
        batch_op.drop_column('max_words')
        batch_op.drop_column('min_words')
