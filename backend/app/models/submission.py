from datetime import datetime, timezone

from app.extensions import db

# submitted -> ai_graded -> approved; grading_failed if the engine errors, so a
# teacher can still grade by hand.
STATUSES = ("submitted", "ai_graded", "grading_failed", "approved")


class Submission(db.Model):
    __tablename__ = "submissions"

    id = db.Column(db.Integer, primary_key=True)
    rubric_id = db.Column(db.Integer, db.ForeignKey("rubrics.id"), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    content = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), nullable=False, default="submitted")

    # Whole-submission commentary. Split the same way as ai_score/final_score on
    # Grade: ai_summary is the engine's draft and is teacher-only, final_summary
    # is what the teacher released and is the only one a student ever sees.
    ai_summary = db.Column(db.Text, nullable=True)
    final_summary = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        db.CheckConstraint("status in ('submitted', 'ai_graded', 'grading_failed', 'approved')",
                            name="ck_submissions_status"),
        # One row per student per rubric. This is the resubmission model, not a
        # bar on it: resubmitting overwrites this row's content and re-grades
        # it, so only the latest attempt is kept. Keeping every attempt would
        # mean dropping this and adding an attempt number.
        db.UniqueConstraint("rubric_id", "student_id", name="uq_submission_per_student_per_rubric"),
    )

    rubric = db.relationship("Rubric", back_populates="submissions")
    student = db.relationship("User", back_populates="submissions", foreign_keys=[student_id])
    grades = db.relationship("Grade", back_populates="submission", cascade="all, delete-orphan")

    @property
    def ai_total(self):
        scores = [g.ai_score for g in self.grades if g.ai_score is not None]
        return sum(scores) if scores else None

    @property
    def final_total(self):
        """None until every criterion has been approved -- mirrors the old rule that
        students never see a partial/unapproved grade."""
        if not self.grades or any(g.final_score is None for g in self.grades):
            return None
        return sum(g.final_score for g in self.grades)

    def __repr__(self):
        return f"<Submission {self.id} rubric={self.rubric_id} student={self.student_id} {self.status}>"
