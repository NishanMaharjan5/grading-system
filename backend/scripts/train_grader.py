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
    title = row["rubric"]
    if title not in rubric_cache:
        rubric = db.session.query(Rubric).filter_by(title=title).first()
        if not rubric:
            raise SystemExit(f"No rubric titled {title!r} exists -- create it first.")
        rubric_cache[title] = rubric
    rubric = rubric_cache[title]

    criterion = next((c for c in rubric.criteria if c.name == row["criterion"]), None)
    if not criterion:
        raise SystemExit(f"Rubric {title!r} has no criterion named {row['criterion']!r}.")
    return criterion


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
        rubric_cache = {}
        grouped = defaultdict(list)  # RubricCriterion -> [(text, score), ...]
        for row in examples:
            criterion = resolve_criterion(row, rubric_cache)
            grouped[criterion].append((row["text"], row["score"]))

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
