"""Creates a demo teacher and the demo rubrics so there is something for
training_data/sample_answers.json to reference. Idempotent -- safe to run
again. Not needed once you have real rubrics from the app itself.

    ./venv/bin/python scripts/seed_demo_rubric.py

Two rubrics on different topics, with the same criterion names and points:
Essay 1 (social media regulation) and Essay 2 (AI tools in schoolwork). The
shared names are deliberate -- the feedback templates and the review UI key
off the name, so a second topic needs no code change. The *models* are
separate regardless, because they are keyed by criterion row id, not by name.

They are created in a fixed order so that after `reset_dev_data.py` (which
truncates with RESTART IDENTITY) the criterion ids are reproducible: Essay 1
takes 1-2, Essay 2 takes 3-4. Retrain after any reset -- models saved under
the old ids do not follow a rubric that was recreated.
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
DEMO_RUBRIC_TITLE = "Essay 1"  # kept: other scripts and docs refer to it by name

# Order matters: see the note above about reproducible criterion ids.
DEMO_RUBRICS = [
    {
        "title": "Essay 1",
        "description": "Should governments regulate social media? Argue one side.",
        "criteria": [
            ("Thesis", 5, "A clear, arguable claim."),
            ("Evidence", 10, "Specific, verifiable support for the claim."),
        ],
    },
    {
        "title": "Essay 2",
        "description": "Should students be allowed to use AI tools like ChatGPT for schoolwork?",
        "criteria": [
            ("Thesis", 5, "A clear, arguable claim."),
            ("Evidence", 10, "Specific, verifiable support for the claim."),
        ],
    },
]


def seed(quiet=False):
    """Ensures the demo teacher and both demo rubrics exist.
    Returns (teacher, [rubric, ...]) in DEMO_RUBRICS order.
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

    rubrics = []
    for spec in DEMO_RUBRICS:
        existing = db.session.query(Rubric).filter_by(title=spec["title"]).all()
        if len(existing) > 1:
            # train_grader.py resolves a training row's rubric by title, so two
            # rubrics sharing one would bind the data to whichever came back
            # first. Refuse rather than pick.
            raise SystemExit(
                f"{len(existing)} rubrics are titled {spec['title']!r} (ids "
                f"{[r.id for r in existing]}). Training data is matched to a rubric by "
                "title, so this is ambiguous. Rename or delete the duplicates first.")

        rubric = existing[0] if existing else None
        if not rubric:
            rubric = Rubric(title=spec["title"], description=spec["description"],
                            type="text", created_by=teacher.id)
            rubric.criteria = [
                RubricCriterion(name=name, max_points=points, position=i, description=description)
                for i, (name, points, description) in enumerate(spec["criteria"])
            ]
            db.session.add(rubric)
            db.session.commit()
            say(f"created rubric {spec['title']!r} (id={rubric.id}) with criteria "
                f"{[(c.id, c.name) for c in rubric.criteria]}")
        else:
            say(f"rubric {spec['title']!r} already exists (id={rubric.id}, criteria "
                f"{[(c.id, c.name) for c in rubric.criteria]})")
        rubrics.append(rubric)

    return teacher, rubrics


def main():
    app = create_app()
    with app.app_context():
        seed()


if __name__ == "__main__":
    main()
