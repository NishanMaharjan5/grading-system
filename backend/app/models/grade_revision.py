from datetime import datetime, timezone

from app.extensions import db


class GradeRevision(db.Model):
    """One criterion's score changing after the grade was already released.

    Approval used to be terminal: once a teacher released a grade there was no
    way to correct it, so a mistake stood for good. Revision fixes that, and
    this table is what makes it safe to allow -- every change is recorded with
    what it was, what it became, who made it and when.

    The audit row is written *before* the live Grade is updated, so a failure
    part-way leaves a record that something was attempted rather than a quietly
    changed score with no trace.

    Rows are append-only. Nothing in the app edits or deletes one; the history
    is the point.
    """

    __tablename__ = "grade_revisions"

    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.Integer, db.ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False)
    criterion_id = db.Column(db.Integer, db.ForeignKey("rubric_criteria.id"), nullable=False)

    # The values as they stood before this revision. Nullable because a
    # criterion could have been released with no feedback written.
    old_final_score = db.Column(db.Numeric(6, 2), nullable=True)
    old_final_feedback = db.Column(db.Text, nullable=True)
    new_final_score = db.Column(db.Numeric(6, 2), nullable=False)
    new_final_feedback = db.Column(db.Text, nullable=True)

    revised_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    revised_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    submission = db.relationship("Submission", back_populates="revisions")
    criterion = db.relationship("RubricCriterion")
    reviser = db.relationship("User", foreign_keys=[revised_by])

    @property
    def changed_score(self):
        """False when only the feedback wording was rewritten."""
        return self.old_final_score is None or float(self.old_final_score) != float(self.new_final_score)
