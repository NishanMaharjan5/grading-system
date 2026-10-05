"""Add grade_revisions audit table

Records every change to an already-released grade: the score and feedback
before and after, which teacher made the change, and when. Approval used to be
terminal, so a mistake stood for good; this is what makes correcting one safe
to allow.

Append-only -- nothing in the app updates or deletes a row. Revisions follow
their submission on delete, since a history with no submission means nothing.

Revision ID: 23b9e64f3752
Revises: 68d10bab5603
Create Date: 2026-10-05 13:56:31.742061

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '23b9e64f3752'
down_revision = '68d10bab5603'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('grade_revisions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('submission_id', sa.Integer(), nullable=False),
    sa.Column('criterion_id', sa.Integer(), nullable=False),
    sa.Column('old_final_score', sa.Numeric(precision=6, scale=2), nullable=True),
    sa.Column('old_final_feedback', sa.Text(), nullable=True),
    sa.Column('new_final_score', sa.Numeric(precision=6, scale=2), nullable=False),
    sa.Column('new_final_feedback', sa.Text(), nullable=True),
    sa.Column('revised_by', sa.Integer(), nullable=False),
    sa.Column('revised_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['criterion_id'], ['rubric_criteria.id'], ),
    sa.ForeignKeyConstraint(['revised_by'], ['users.id'], ),
    sa.ForeignKeyConstraint(['submission_id'], ['submissions.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    # The history is always read for one submission at a time.
    op.create_index('ix_grade_revisions_submission_id', 'grade_revisions', ['submission_id'])


def downgrade():
    op.drop_index('ix_grade_revisions_submission_id', table_name='grade_revisions')
    op.drop_table('grade_revisions')
