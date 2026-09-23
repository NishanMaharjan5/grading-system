from datetime import datetime, timezone

from app.extensions import db


class Grade(db.Model):
    """One row per (submission, criterion). Holds both the AI engine's suggestion
    and the teacher's final, approved value side by side -- never overwrite
    ai_score/ai_feedback with the teacher's edits, so what the AI actually
    suggested stays auditable."""

    __tablename__ = "grades"

    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.Integer, db.ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False)
    criterion_id = db.Column(db.Integer, db.ForeignKey("rubric_criteria.id"), nullable=False)

    ai_score = db.Column(db.Numeric(6, 2), nullable=True)
    ai_feedback = db.Column(db.Text, nullable=True)

    final_score = db.Column(db.Numeric(6, 2), nullable=True)
    final_feedback = db.Column(db.Text, nullable=True)

    approved_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    approved_at = db.Column(db.DateTime(timezone=True), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        db.CheckConstraint("ai_score is null or ai_score >= 0", name="ck_grades_ai_score_nonneg"),
        db.CheckConstraint("final_score is null or final_score >= 0", name="ck_grades_final_score_nonneg"),
        db.UniqueConstraint("submission_id", "criterion_id", name="uq_grade_per_criterion_per_submission"),
    )

    submission = db.relationship("Submission", back_populates="grades")
    criterion = db.relationship("RubricCriterion", back_populates="grades")
    approver = db.relationship("User", foreign_keys=[approved_by])

    def __repr__(self):
        return f"<Grade submission={self.submission_id} criterion={self.criterion_id} final={self.final_score}>"
