"""Scores a held-out, team-labeled evaluation set against the models currently
saved in ml_models/, without touching training at all.

Usage (from backend/, with the venv active):
    ./venv/bin/python scripts/evaluate_holdout.py

Reads training_data/holdout_team_scored.json -- same {"rubric", "criterion",
"text", "score"} schema as sample_answers.json, plus an "essay_id" field so
rows can be grouped back into essays. This script only ever loads models with
model_store.load_model(); it never calls train_grader.py, never calls
save_model(), and never writes to sample_answers.json. Re-running it twice in
a row must produce identical numbers.

Leakage guard
-------------
Before scoring, every holdout row is compared against every training row for
the same criterion two ways:

1. Text similarity (difflib.SequenceMatcher on lowercased, punctuation-
   stripped text). Catches copy-pasted or lightly-edited duplicates.
2. Embedding similarity (cosine distance between frozen SBERT vectors --
   the actual feature space the models were fit on). Catches a paraphrase
   that reads differently but would land in nearly the same place for the
   model, which plain text similarity misses entirely.

A hard match on either (text ratio >= DUPLICATE_TEXT_THRESHOLD, or cosine >=
DUPLICATE_EMBED_THRESHOLD) stops the run -- that row is not a fair holdout.
A softer embedding match (>= NOTABLE_EMBED_THRESHOLD) does not stop anything;
it's printed as a disclosure, because a holdout essay can legitimately reuse
a real, well-known fact (the same statistic, the same law) in a new argument
without being a duplicate of the training example that already used it.
"""

import json
import os
import sys
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.grading.embedder import embed
from app.grading.engine import build_matrix
from app.grading.model_store import load_model
from app.models import Rubric

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BACKEND_DIR, "training_data", "sample_answers.json")
HOLDOUT_PATH = os.path.join(BACKEND_DIR, "training_data", "holdout_team_scored.json")

DUPLICATE_TEXT_THRESHOLD = 0.85
DUPLICATE_EMBED_THRESHOLD = 0.90
NOTABLE_EMBED_THRESHOLD = 0.60

# The essay-length grouping the caller wants checked, by essay_id.
LONG_ESSAY_IDS = {1, 2, 6, 8, 10, 12}
SHORT_ESSAY_IDS = {3, 4, 5, 7, 9, 11}
KNOWN_WEAKNESS_IDS = {4, 11}  # facts stated without being tied to the essay's argument


def normalize(text):
    import re
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def load_json(path, label):
    if not os.path.exists(path):
        raise SystemExit(f"{path} does not exist -- nothing to {label}.")
    with open(path) as f:
        return json.load(f)


def check_leakage(holdout_rows, train_rows):
    """Compares every holdout row against every training row sharing its
    criterion. Raises SystemExit on a hard duplicate. Returns a list of
    (holdout_row, train_row, cosine) notable-but-not-disqualifying matches
    to print later."""
    by_criterion = defaultdict(list)
    for row in train_rows:
        by_criterion[row["criterion"]].append(row)

    # Embed everything once, in bulk, rather than per-comparison.
    train_texts_by_criterion = {c: [r["text"] for r in rows] for c, rows in by_criterion.items()}
    train_embed_by_criterion = {}
    for c, texts in train_texts_by_criterion.items():
        vecs = embed(texts)
        train_embed_by_criterion[c] = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    holdout_texts = [row["text"] for row in holdout_rows]
    holdout_vecs = embed(holdout_texts)
    holdout_vecs = holdout_vecs / np.linalg.norm(holdout_vecs, axis=1, keepdims=True)

    notable = []
    hard_matches = []

    for row, vec in zip(holdout_rows, holdout_vecs):
        criterion = row["criterion"]
        candidates = by_criterion.get(criterion, [])
        if not candidates:
            continue
        norm_row = normalize(row["text"])

        sims = train_embed_by_criterion[criterion] @ vec
        best_idx = int(np.argmax(sims))
        best_cosine = float(sims[best_idx])
        best_train_row = candidates[best_idx]

        best_ratio = max(
            SequenceMatcher(None, norm_row, normalize(t)).ratio() for t in train_texts_by_criterion[criterion]
        )

        if best_ratio >= DUPLICATE_TEXT_THRESHOLD or best_cosine >= DUPLICATE_EMBED_THRESHOLD:
            hard_matches.append((row, best_train_row, best_ratio, best_cosine))
        elif best_cosine >= NOTABLE_EMBED_THRESHOLD:
            notable.append((row, best_train_row, best_cosine))

    if hard_matches:
        print("LEAKAGE GUARD: STOPPING -- the following holdout rows look like duplicates of training data:\n")
        for row, train_row, ratio, cosine in hard_matches:
            print(f"  essay {row.get('essay_id', '?')} / {row['criterion']}: text ratio {ratio:.2f}, cosine {cosine:.2f}")
            print(f"    holdout : {row['text'][:110]}")
            print(f"    training: {train_row['text'][:110]}")
        raise SystemExit(
            "\nFix or remove these rows before evaluating -- a holdout essay that duplicates a training "
            "example isn't held out."
        )

    return notable


def resolve_criterion(rubric, name):
    criterion = next((c for c in rubric.criteria if c.name == name), None)
    if not criterion:
        raise SystemExit(f"Rubric {rubric.title!r} has no criterion named {name!r}.")
    return criterion


def predict_for(criterion, texts):
    """Mirrors app/grading/engine.py grade_text_submission exactly: same
    featurisation, same round-then-clip, so this reports what the app would
    actually have scored, not an idealised version of the model."""
    payload = load_model(criterion.id)
    if payload is None:
        raise SystemExit(f"No trained model for criterion {criterion.name!r} -- run scripts/train_grader.py first.")
    X = build_matrix(texts)
    raw = payload["model"].predict(X)
    max_points = payload["max_points"]
    y_pred = np.clip(np.round(raw), 0, max_points)
    y_pred = np.where(y_pred == 0, 0.0, y_pred)  # drop the sign bit on -0.0 so it doesn't print as "-0"
    return y_pred, max_points


def metrics_for(y_true, y_pred, max_points):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.abs(y_pred - y_true).mean())
    within_one = float(np.mean(np.abs(y_pred - y_true) <= 1))
    exact = float(np.mean(y_pred == y_true))
    labels = list(range(0, int(max_points) + 1))
    qwk = float(cohen_kappa_score(y_true.astype(int), y_pred.astype(int), weights="quadratic", labels=labels))
    return {"mae": mae, "within_one": within_one, "exact": exact, "qwk": qwk}


def baseline_for(y_true, max_points):
    y_true = np.asarray(y_true, dtype=float)
    constant = float(np.clip(round(y_true.mean()), 0, max_points))
    y_pred = np.full_like(y_true, constant)
    return metrics_for(y_true, y_pred, max_points), constant


def print_metrics_row(label, m):
    print(f"    {label:<18} MAE {m['mae']:.2f}   within-1 {m['within_one']:.0%}   exact {m['exact']:.0%}   QWK {m['qwk']:.2f}")


def print_essay_table(criterion_name, essay_ids, y_true, y_pred):
    print(f"\n  {criterion_name} -- true vs. predicted, sorted by |error| (worst first):")
    print(f"    {'essay':<6}{'true':<6}{'pred':<6}{'error':<7}")
    rows = list(zip(essay_ids, y_true, y_pred))
    rows.sort(key=lambda r: -abs(r[2] - r[1]))
    for essay_id, true, pred in rows:
        error = pred - true
        sign = "+" if error > 0 else ("" if error == 0 else "")
        print(f"    {essay_id:<6}{true:<6.0f}{pred:<6.0f}{sign}{error:<6.0f}")


def length_breakdown(criterion_name, essay_ids, y_true, y_pred):
    essay_ids = np.array(essay_ids)
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    abs_err = np.abs(y_pred - y_true)

    long_mask = np.isin(essay_ids, list(LONG_ESSAY_IDS))
    short_mask = np.isin(essay_ids, list(SHORT_ESSAY_IDS))

    long_mae = abs_err[long_mask].mean() if long_mask.any() else float("nan")
    short_mae = abs_err[short_mask].mean() if short_mask.any() else float("nan")

    print(f"\n  {criterion_name} -- error by essay length:")
    print(f"    paragraph-length essays {sorted(LONG_ESSAY_IDS)}: MAE {long_mae:.2f}")
    print(f"    short essays            {sorted(SHORT_ESSAY_IDS)}: MAE {short_mae:.2f}")

    for essay_id in sorted(KNOWN_WEAKNESS_IDS):
        if essay_id in essay_ids:
            idx = int(np.where(essay_ids == essay_id)[0][0])
            print(
                f"    essay {essay_id} (known weakness: facts stated without being tied to the argument): "
                f"true {y_true[idx]:.0f}, predicted {y_pred[idx]:.0f}, error {y_pred[idx] - y_true[idx]:+.0f}"
            )

    gap = short_mae - long_mae
    if abs(gap) < 0.5:
        verdict = "no meaningful length gap -- error is roughly flat across short and paragraph-length essays"
    elif gap > 0:
        verdict = "short essays have the larger error here"
    else:
        verdict = "paragraph-length essays have the larger error here"
    print(f"    verdict: {verdict}")


def main():
    app = create_app()
    with app.app_context():
        train_rows = load_json(TRAIN_PATH, "compare against")
        holdout_rows = load_json(HOLDOUT_PATH, "evaluate")

        print(f"Loaded {len(holdout_rows)} holdout rows from {HOLDOUT_PATH}")
        print(f"Checking against {len(train_rows)} training rows in {TRAIN_PATH}\n")

        notable = check_leakage(holdout_rows, train_rows)
        if notable:
            print("Leakage guard: no duplicates found, but these holdout rows share a specific real-world")
            print("fact with a training example (same statistic/law, different wording and argument) --")
            print("not disqualifying, but worth knowing about when reading the Evidence results:\n")
            for row, train_row, cosine in notable:
                print(f"  essay {row.get('essay_id', '?')} / {row['criterion']}: cosine {cosine:.2f}")
                print(f"    holdout : {row['text'][:110]}")
                print(f"    training: {train_row['text'][:110]}\n")
        else:
            print("Leakage guard: no duplicate or near-duplicate rows found.\n")

        rubric = None
        by_criterion = defaultdict(list)  # name -> [(essay_id, text, score)]
        for row in holdout_rows:
            if rubric is None:
                rubric = Rubric.query.filter_by(title=row["rubric"]).first()
                if not rubric:
                    raise SystemExit(f"No rubric titled {row['rubric']!r} exists.")
            by_criterion[row["criterion"]].append((row["essay_id"], row["text"], row["score"]))

        print(f"Evaluating against models trained on {TRAIN_PATH.rsplit('/', 1)[-1]} "
              f"(re-run scripts/train_grader.py first if you've retrained since this data was written).\n")

        for name, rows in by_criterion.items():
            criterion = resolve_criterion(rubric, name)
            essay_ids = [r[0] for r in rows]
            texts = [r[1] for r in rows]
            y_true = [r[2] for r in rows]

            y_pred, max_points = predict_for(criterion, texts)
            m = metrics_for(y_true, y_pred, max_points)
            b, baseline_constant = baseline_for(y_true, max_points)

            print(f"=== {name} (0-{max_points:g}), n={len(rows)} ===")
            print_metrics_row("model", m)
            print_metrics_row(f"baseline ({baseline_constant:g})", b)
            print_essay_table(name, essay_ids, y_true, y_pred)
            length_breakdown(name, essay_ids, y_true, y_pred)
            print()

        print(
            "Note: 12 essays is too few for QWK to be more than indicative -- treat these numbers as a "
            "read on direction and rough size, not a precise estimate."
        )


if __name__ == "__main__":
    main()
