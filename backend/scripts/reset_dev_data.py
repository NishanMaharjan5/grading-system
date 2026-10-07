"""Wipes the development database and re-seeds the demo teacher and rubric.

    ./venv/bin/python scripts/reset_dev_data.py --yes

This DELETES EVERY ROW in the database DATABASE_URL points at -- all users,
rubrics, submissions and grades. It exists to clear the throwaway accounts and
rubrics that pile up from manual and browser testing before a demo.

Two guards, because this is unrecoverable:
  * --yes is required; without it the script only reports what it would delete.
  * A database whose name contains neither "dev" nor "test" is refused unless
    --force is also given.

Tables are truncated with RESTART IDENTITY, so ids begin at 1 again and the
re-seeded rubrics' criteria land on the same ids the retired Ridge models in
ml_models/*.joblib were saved under.

**This no longer risks orphaning the grader that students are actually
scored by.** That model -- the fine-tuned one in ml_models/bert_rubric_scorer/
-- is looked up by criterion *name*, not by row id, precisely so that a
reseed cannot silently disconnect it. Re-seed as often as you like; Thesis and
Evidence keep grading. The guarantee is tested, not assumed: see
TestCriterionLookupIsByName in tests/test_grading.py, which recreates a rubric
under different row ids and asserts the scores are unchanged, and separately
runs this script's own TRUNCATE ... RESTART IDENTITY and checks grading still
works.
"""

import argparse
import os
import sys

from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extensions import db
from app.models import Grade, Rubric, RubricCriterion, Submission, User
from scripts.seed_demo_rubric import seed

COUNTED = [("users", User), ("rubrics", Rubric), ("criteria", RubricCriterion),
           ("submissions", Submission), ("grades", Grade)]


def database_name(uri):
    return uri.rsplit("/", 1)[-1].split("?")[0]


def main():
    parser = argparse.ArgumentParser(description="Wipe and re-seed the development database.")
    parser.add_argument("--yes", action="store_true", help="actually delete (otherwise this is a dry run)")
    parser.add_argument("--force", action="store_true", help="allow a database not named *dev* or *test*")
    args = parser.parse_args()

    app = create_app()
    name = database_name(app.config["SQLALCHEMY_DATABASE_URI"])

    # Checked before connecting: a database this script may not wipe should not
    # even be opened.
    if not ("dev" in name or "test" in name) and not args.force:
        raise SystemExit(
            f"Refusing to wipe {name!r}: its name contains neither 'dev' nor 'test'.\n"
            "Pass --force if you are certain this is a throwaway database."
        )

    with app.app_context():
        print(f"database: {name}")
        for label, model in COUNTED:
            print(f"  {label:<12} {db.session.query(model).count()}")

        if not args.yes:
            print("\nDry run. Nothing was deleted. Re-run with --yes to wipe and re-seed.")
            return

        tables = ", ".join(table.name for table in db.metadata.sorted_tables)
        db.session.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
        db.session.commit()
        print("\nAll rows deleted, ids reset.")

        teacher, rubrics = seed()
        print(f"\nSeeded: {teacher.email} / password1")
        for rubric in rubrics:
            print(f"  rubric {rubric.title!r} (criteria {[(c.id, c.name) for c in rubric.criteria]})")
        print("\nThe text grader needs no retraining after this: it is keyed by criterion\n"
              "name, not by row id, so Thesis and Evidence keep working. Only the retired\n"
              "Ridge baseline is id-keyed, and nothing grades students with it:\n"
              "    make train   # optional, seconds, benchmark/probe baseline only")


if __name__ == "__main__":
    main()
