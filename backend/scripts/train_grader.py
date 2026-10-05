"""Trains one small regression model per rubric criterion on top of frozen
Sentence-BERT (all-MiniLM-L6-v2) embeddings plus hand-crafted features.

Usage (from backend/, with the venv active):
    ./venv/bin/python scripts/train_grader.py

Reads training_data/sample_answers.json -- a list of
{"rubric": <title>, "criterion": <name>, "text": <answer>, "score": <number>}
objects. Examples reference a rubric/criterion by name rather than raw id, so
the file stays readable and doesn't break if a rubric gets recreated; both
must already exist in the database this connects to (DATABASE_URL / .env).

Why regression rather than a classifier: rubric scores are ordinal. A
classifier treats them as unrelated labels, so predicting 0 for a true 5 costs
exactly as much as predicting 4. Measured on the first 35 labeled examples,
LogisticRegression on embeddings scored *worse than always guessing the mean*
(MAE 2.50 vs 1.39 on Thesis); Ridge on embeddings + hand-crafted features cut
that to 1.06.

Every criterion is scored against that same "always predict the mean" baseline
below, because an accuracy number without a baseline is unreadable -- and with
this few examples, a model that loses to the baseline is a real possibility
worth seeing plainly.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extensions import db
from app.grading.engine import build_matrix
from app.grading.metrics import quadratic_weighted_kappa
from app.grading.model_store import save_model
from app.models import Rubric

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training_data", "sample_answers.json"
)

RIDGE_ALPHA = 1.0


def make_model():
    # StandardScaler matters here: the hand-crafted features (word counts,
    # 0/1 flags) live on wildly different scales from the embedding dimensions.
    return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))


def load_examples(path=None):
    with open(path or DATA_PATH) as f:
        rows = json.load(f)
    if not rows:
        raise SystemExit(f"{path or DATA_PATH} has no examples yet -- add some labeled answers first.")
    return rows


def resolve_criterion(row, rubric_cache):
    """Matches one training row to the criterion row it will train.

    Returns (criterion, None), or (None, reason) when the row cannot be
    resolved. Returning the reason rather than raising lets the caller collect
    every bad row and report them together -- fixing them one run at a time is
    tedious when a file has hundreds.

    An ambiguous title is refused outright. Models are saved under a criterion
    *row id*, so picking either of two rubrics sharing a title would train real
    models against the wrong criteria, silently and with no error. That is not
    hypothetical: two rubrics titled "Essay 1" existed in the dev database in
    early October 2026.
    """
    title = row["rubric"]
    if title not in rubric_cache:
        matches = db.session.query(Rubric).filter_by(title=title).order_by(Rubric.id).all()
        if not matches:
            return None, f"no rubric titled {title!r} exists -- create it first"
        if len(matches) > 1:
            return None, (
                f"{len(matches)} rubrics are titled {title!r} (ids {[r.id for r in matches]}); "
                "training data is matched to a rubric by title, so this is ambiguous -- "
                "rename or delete the duplicates"
            )
        rubric_cache[title] = matches[0]
    rubric = rubric_cache[title]

    criterion = next((c for c in rubric.criteria if c.name == row["criterion"]), None)
    if not criterion:
        available = ", ".join(repr(c.name) for c in rubric.criteria) or "none"
        return None, (f"rubric {title!r} has no criterion named {row['criterion']!r} "
                      f"(it has: {available})")
    return criterion, None


def resolve_all(examples):
    """Resolves every row before any training starts. Raises SystemExit listing
    every unresolvable row, so one run surfaces all of them."""
    rubric_cache = {}
    grouped = defaultdict(list)  # RubricCriterion -> [(text, score), ...]
    blocked = []  # (row number, rubric title, criterion name, reason)

    for number, row in enumerate(examples, start=1):
        criterion, reason = resolve_criterion(row, rubric_cache)
        if reason:
            blocked.append((number, row.get("rubric"), row.get("criterion"), reason))
            continue
        grouped[criterion].append((row["text"], row["score"]))

    if blocked:
        # Group by reason: one duplicate title usually blocks many rows, and a
        # list of 50 identical messages buries the one thing to fix.
        by_reason = defaultdict(list)
        for number, title, name, reason in blocked:
            by_reason[reason].append((number, title, name))

        lines = [f"{len(blocked)} of {len(examples)} training rows could not be resolved. "
                 "Nothing was trained and no model file was written.\n"]
        for reason, rows in by_reason.items():
            numbers = [n for n, _, _ in rows]
            shown = ", ".join(str(n) for n in numbers[:10])
            more = f" and {len(numbers) - 10} more" if len(numbers) > 10 else ""
            lines.append(f"  {reason}")
            lines.append(f"    blocked rows (1-based): {shown}{more}")
            lines.append(f"    first one: rubric {rows[0][1]!r}, criterion {rows[0][2]!r}\n")
        raise SystemExit("\n".join(lines))

    return grouped


def train_one(criterion, texts, scores, save=True):
    max_points = float(criterion.max_points)

    bad = [s for s in scores if not (0 <= s <= max_points)]
    if bad:
        raise SystemExit(f"{criterion.name}: score(s) {bad} are outside 0..{max_points:g}")

    distinct = sorted(set(scores))
    if len(distinct) < 2:
        print(f"  skip {criterion.name!r}: every example scores {distinct[0]} -- nothing to learn from")
        return

    X = build_matrix(texts)
    y = np.array(scores, dtype=float)

    def to_points(raw):
        return np.clip(np.rint(raw), 0, max_points)

    # Leave-one-out rather than k-fold: with this few examples every held-out
    # point counts, and k-fold can leave whole scores unrepresented in training.
    predicted = to_points(cross_val_predict(make_model(), X, y, cv=LeaveOneOut()))
    mae = np.abs(predicted - y).mean()
    within_one = np.mean(np.abs(predicted - y) <= 1)
    exact = np.mean(predicted == y)

    # QWK on the same held-out predictions: chance-corrected agreement that
    # punishes being far off more than being slightly off, which is what an
    # ordinal rubric score needs. MAE alone can't distinguish a model that is
    # reliably one point out from one that is occasionally five points out.
    kappa = quadratic_weighted_kappa(y, predicted, max_points)

    # The number any real model has to beat.
    baseline = to_points(np.full_like(y, y.mean()))
    baseline_mae = np.abs(baseline - y).mean()
    baseline_within_one = np.mean(np.abs(baseline - y) <= 1)
    baseline_exact = np.mean(baseline == y)

    if save:
        model = make_model().fit(X, y)
        save_model(criterion.id, {"model": model, "max_points": max_points, "n_examples": len(y)})

    verdict = "beats baseline" if mae < baseline_mae else "NO BETTER THAN GUESSING -- needs more/better examples"
    print(f"  {criterion.name!r} (0-{max_points:g}): n={len(y)}, distinct scores {distinct}")
    print(f"      leave-one-out MAE {mae:.2f}  vs  baseline {baseline_mae:.2f}   -- {verdict}")
    print(f"      exact {exact:.0%} (baseline {baseline_exact:.0%}),  "
          f"within 1 point {within_one:.0%} (baseline {baseline_within_one:.0%})")
    print(f"      QWK {kappa:.2f}   (baseline QWK is 0.00 by construction: constant predictions)")
    if not save:
        print("      [--no-save] metrics only; models on disk were not touched")


def parse_args():
    parser = argparse.ArgumentParser(description="Train (or just score) the per-criterion graders.")
    parser.add_argument("--data", default=None,
                        help="labeled examples to use (default: training_data/sample_answers.json)")
    parser.add_argument("--no-save", action="store_true",
                        help="report leave-one-out metrics without writing models -- use to measure a "
                             "dataset without disturbing what the app is serving")
    return parser.parse_args()


def main():
    args = parse_args()
    app = create_app()
    with app.app_context():
        examples = load_examples(args.data)
        grouped = resolve_all(examples)

        verb = "Scoring" if args.no_save else "Training on"
        print(f"{verb} {len(examples)} examples from {args.data or DATA_PATH}\n")
        for criterion, pairs in grouped.items():
            texts = [t for t, _ in pairs]
            scores = [s for _, s in pairs]
            train_one(criterion, texts, scores, save=not args.no_save)

        print("\nNote: with double-digit example counts these numbers move several points "
              "if one example changes. Treat them as a smoke test, not a validation.")


if __name__ == "__main__":
    main()
