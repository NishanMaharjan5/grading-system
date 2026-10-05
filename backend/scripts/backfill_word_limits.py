"""Sets a word range on existing text rubrics that have none.

    ./venv/bin/python scripts/backfill_word_limits.py            # dry run
    ./venv/bin/python scripts/backfill_word_limits.py --apply

Rubrics created before word limits existed accept any length, which means a
long essay is graded on roughly its first 226 words with nothing to show the
rest was dropped. This closes that for the rubrics already in a database.

Deliberately conservative:

- Only text rubrics. A code rubric is graded by running tests.
- Only rubrics with *no* limit set. A teacher who already chose a range keeps
  it, including a range above the grader's limit.
- Submissions already recorded are never touched or re-checked. Work that was
  acceptable when it was made stays acceptable.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extensions import db
from app.models import Rubric
from app.word_limits import DEFAULT_MAX_WORDS, GRADER_WORD_LIMIT, describe_range

DEFAULT_MIN_WORDS = 20


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write the changes (otherwise dry run)")
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--title", action="append",
                        help="limit to these titles (repeatable); default: every unlimited text rubric")
    args = parser.parse_args()

    if args.max_words > GRADER_WORD_LIMIT:
        print(f"note: {args.max_words} is above the grader's {GRADER_WORD_LIMIT}-word read, "
              "so the end of these essays would not reach the model.\n")

    app = create_app()
    with app.app_context():
        query = db.session.query(Rubric).filter(
            Rubric.type == "text", Rubric.min_words.is_(None), Rubric.max_words.is_(None),
        )
        if args.title:
            query = query.filter(Rubric.title.in_(args.title))
        targets = query.order_by(Rubric.id).all()

        skipped = db.session.query(Rubric).filter(
            Rubric.type == "text",
            db.or_(Rubric.min_words.isnot(None), Rubric.max_words.isnot(None)),
        ).count()

        if not targets:
            print("Nothing to do: no text rubric is missing a word range"
                  + (f" ({skipped} already have one)" if skipped else "") + ".")
            return

        print(f"{'Setting' if args.apply else 'Would set'} "
              f"{describe_range(args.min_words, args.max_words)} on {len(targets)} rubric(s):")
        for rubric in targets:
            print(f"  id={rubric.id} {rubric.title!r} ({len(rubric.submissions)} existing submission(s), untouched)")
        if skipped:
            print(f"\nLeaving {skipped} rubric(s) that already have a range of their own.")

        if not args.apply:
            print("\nDry run. Nothing was written. Re-run with --apply.")
            return

        for rubric in targets:
            rubric.min_words = args.min_words
            rubric.max_words = args.max_words
        db.session.commit()
        print(f"\nDone. {len(targets)} rubric(s) updated; no submission was modified.")


if __name__ == "__main__":
    main()
