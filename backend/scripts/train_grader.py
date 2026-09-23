"""Trains one small classifier per rubric criterion on top of frozen
Sentence-BERT (all-MiniLM-L6-v2) embeddings.

Usage (from backend/, with the venv active):
    ./venv/bin/python scripts/train_grader.py

Reads training_data/sample_answers.json -- a list of
{"rubric": <title>, "criterion": <name>, "text": <answer>, "score": <number>}
objects. Examples reference a rubric/criterion by name rather than raw id, so
the file stays readable and doesn't break if a rubric gets recreated; both
must already exist in the database this connects to (DATABASE_URL / .env).

Each criterion needs at least two distinct score values in its examples --
a classifier can't learn from a single class. Cross-validated accuracy is
only reported when there's enough data per class to do it honestly; with a
handful of self-written examples, treat these numbers as a sanity check that
the pipeline works, not proof the grader generalizes.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extensions import db
from app.grading.embedder import embed
from app.grading.model_store import save_classifier
from app.models import Rubric

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training_data", "sample_answers.json"
)


def load_examples():
    with open(DATA_PATH) as f:
        rows = json.load(f)
    if not rows:
        raise SystemExit(f"{DATA_PATH} has no examples yet -- add some labeled answers first.")
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


def train_one(criterion, texts, scores):
    bad = [s for s in scores if not (0 <= s <= float(criterion.max_points))]
    if bad:
        raise SystemExit(f"{criterion.name}: score(s) {bad} are outside 0..{criterion.max_points}")

    classes = sorted(set(scores))
    if len(classes) < 2:
        print(f"  skip {criterion.name!r}: only one distinct score ({classes[0]}) seen -- add some variety")
        return

    X = embed(texts)
    y = np.array(scores)
    min_count = min(y.tolist().count(c) for c in classes)

    clf = LogisticRegression(max_iter=2000)
    if min_count >= 2:
        cv = StratifiedKFold(n_splits=min(3, min_count))
        acc = cross_val_score(clf, X, y, cv=cv).mean()
        note = f"cv accuracy {acc:.0%} over {cv.get_n_splits()} folds"
    else:
        note = "not enough examples per class for cross-validation -- accuracy not reported"

    clf.fit(X, y)
    save_classifier(criterion.id, {"model": clf, "classes": classes, "max_points": float(criterion.max_points)})
    print(f"  {criterion.name!r}: {len(texts)} examples, classes {classes} -- {note}")


def main():
    app = create_app()
    with app.app_context():
        examples = load_examples()
        rubric_cache = {}
        grouped = defaultdict(list)  # RubricCriterion -> [(text, score), ...]
        for row in examples:
            criterion = resolve_criterion(row, rubric_cache)
            grouped[criterion].append((row["text"], row["score"]))

        print(f"Training on {len(examples)} examples from {DATA_PATH}\n")
        for criterion, pairs in grouped.items():
            texts = [t for t, _ in pairs]
            scores = [s for _, s in pairs]
            train_one(criterion, texts, scores)


if __name__ == "__main__":
    main()
