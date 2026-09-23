from datetime import datetime, timezone

from app.extensions import db


class Rubric(db.Model):
    """The gradable unit a student submits against (what earlier drafts called
    'Assignment'). `type` records whether it's graded by criteria (text, via the
    SBERT+classifier engine) or by test cases (code) — the code-grading path adds
    its own test-case table later and isn't modeled here yet."""

    __tablename__ = "rubrics"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False, default="")
    type = db.Column(db.String(10), nullable=False, default="text")
    due_date = db.Column(db.DateTime(timezone=True), nullable=True)
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        db.CheckConstraint("type in ('text', 'code')", name="ck_rubrics_type"),
    )

    creator = db.relationship("User", back_populates="rubrics_created", foreign_keys=[created_by])
    criteria = db.relationship(
        "RubricCriterion", back_populates="rubric",
        cascade="all, delete-orphan", order_by="RubricCriterion.position",
    )
    submissions = db.relationship("Submission", back_populates="rubric")

    @property
    def total_points(self):
        return sum(c.max_points for c in self.criteria)

    def __repr__(self):
        return f"<Rubric {self.id} {self.title!r}>"


class RubricCriterion(db.Model):
    __tablename__ = "rubric_criteria"

    id = db.Column(db.Integer, primary_key=True)
    rubric_id = db.Column(db.Integer, db.ForeignKey("rubrics.id", ondelete="CASCADE"), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False, default="")
    max_points = db.Column(db.Numeric(6, 2), nullable=False)
    position = db.Column(db.Integer, nullable=False, default=0)  # display order within the rubric

    __table_args__ = (
        db.CheckConstraint("max_points > 0", name="ck_criteria_positive_points"),
        db.UniqueConstraint("rubric_id", "name", name="uq_criteria_rubric_name"),
    )

    rubric = db.relationship("Rubric", back_populates="criteria")
    grades = db.relationship("Grade", back_populates="criterion")

    def __repr__(self):
        return f"<RubricCriterion {self.id} {self.name!r} /{self.max_points}>"
