from datetime import datetime, timezone

from app.extensions import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(255), nullable=False, unique=True, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="student")
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        db.CheckConstraint("role in ('student', 'teacher')", name="ck_users_role"),
    )

    # Rubrics this user authored (only meaningful for teachers)
    rubrics_created = db.relationship("Rubric", back_populates="creator", foreign_keys="Rubric.created_by")
    # Submissions this user made (only meaningful for students)
    submissions = db.relationship("Submission", back_populates="student", foreign_keys="Submission.student_id")

    def __repr__(self):
        return f"<User {self.id} {self.email} {self.role}>"
