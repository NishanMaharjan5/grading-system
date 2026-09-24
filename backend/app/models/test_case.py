from datetime import datetime, timezone

from app.extensions import db


class TestCase(db.Model):
    """One stdin/stdout case for a criterion on a code rubric.

    Hung off the criterion rather than the rubric so scoring stays per
    criterion exactly as the text path is: a criterion is worth max_points,
    and a submission earns the fraction of that criterion's cases it passes.
    """

    __tablename__ = "test_cases"
    __test__ = False  # stops pytest collecting this as a test class

    id = db.Column(db.Integer, primary_key=True)
    criterion_id = db.Column(db.Integer, db.ForeignKey("rubric_criteria.id", ondelete="CASCADE"), nullable=False)

    stdin = db.Column(db.Text, nullable=False, default="")
    expected_output = db.Column(db.Text, nullable=False)
    position = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    criterion = db.relationship("RubricCriterion", back_populates="test_cases")

    def __repr__(self):
        return f"<TestCase {self.id} criterion={self.criterion_id}>"
