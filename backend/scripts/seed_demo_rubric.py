"""One-time convenience: creates a demo teacher + a demo rubric so there's
something for training_data/sample_answers.json to reference. Idempotent --
safe to run again. Not needed once you have real rubrics from the app itself."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extensions import db
from app.models import Rubric, RubricCriterion, User
from app.security import hash_password

DEMO_TEACHER_EMAIL = "demo_teacher@example.com"
DEMO_RUBRIC_TITLE = "Essay 1"


def main():
    app = create_app()
    with app.app_context():
        teacher = db.session.query(User).filter_by(email=DEMO_TEACHER_EMAIL).first()
        if not teacher:
            teacher = User(
                name="Demo Teacher", email=DEMO_TEACHER_EMAIL,
                password_hash=hash_password("password1"), role="teacher",
            )
            db.session.add(teacher)
            db.session.commit()
            print(f"created {DEMO_TEACHER_EMAIL} (password: password1)")

        rubric = db.session.query(Rubric).filter_by(title=DEMO_RUBRIC_TITLE).first()
        if not rubric:
            rubric = Rubric(title=DEMO_RUBRIC_TITLE, type="text", created_by=teacher.id)
            rubric.criteria = [
                RubricCriterion(name="Thesis", max_points=5, position=0),
                RubricCriterion(name="Evidence", max_points=10, position=1),
            ]
            db.session.add(rubric)
            db.session.commit()
            print(f"created rubric {DEMO_RUBRIC_TITLE!r} (id={rubric.id}) with criteria: "
                  f"{[c.name for c in rubric.criteria]}")
        else:
            print(f"rubric {DEMO_RUBRIC_TITLE!r} already exists (id={rubric.id})")


if __name__ == "__main__":
    main()
