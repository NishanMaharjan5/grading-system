"""Creates a demo teacher and a demo rubric so there is something for
training_data/sample_answers.json to reference. Idempotent -- safe to run
again. Not needed once you have real rubrics from the app itself.

    ./venv/bin/python scripts/seed_demo_rubric.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extensions import db
from app.models import Rubric, RubricCriterion, User
from app.security import hash_password

DEMO_TEACHER_EMAIL = "demo_teacher@example.com"
DEMO_TEACHER_PASSWORD = "password1"
DEMO_RUBRIC_TITLE = "Essay 1"


def seed(quiet=False):
    """Ensures the demo teacher and rubric exist. Returns (teacher, rubric).
    Must be called inside an app context."""
    def say(message):
        if not quiet:
            print(message)

    teacher = db.session.query(User).filter_by(email=DEMO_TEACHER_EMAIL).first()
    if not teacher:
        teacher = User(
            name="Demo Teacher",
            email=DEMO_TEACHER_EMAIL,
            password_hash=hash_password(DEMO_TEACHER_PASSWORD),
            role="teacher",
        )
        db.session.add(teacher)
        db.session.commit()
        say(f"created {DEMO_TEACHER_EMAIL} (password: {DEMO_TEACHER_PASSWORD})")

    rubric = db.session.query(Rubric).filter_by(title=DEMO_RUBRIC_TITLE).first()
    if not rubric:
        rubric = Rubric(title=DEMO_RUBRIC_TITLE, type="text", created_by=teacher.id)
        rubric.criteria = [
            RubricCriterion(name="Thesis", max_points=5, position=0,
                            description="A clear, arguable claim."),
            RubricCriterion(name="Evidence", max_points=10, position=1,
                            description="Specific, verifiable support for the claim."),
        ]
        db.session.add(rubric)
        db.session.commit()
        say(f"created rubric {DEMO_RUBRIC_TITLE!r} (id={rubric.id}) with criteria "
            f"{[(c.id, c.name) for c in rubric.criteria]}")
    else:
        say(f"rubric {DEMO_RUBRIC_TITLE!r} already exists (id={rubric.id})")

    return teacher, rubric


def main():
    app = create_app()
    with app.app_context():
        seed()


if __name__ == "__main__":
    main()
