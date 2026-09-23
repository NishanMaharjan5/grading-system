"""Scores a submission against each rubric criterion using the per-criterion
models trained by scripts/train_grader.py.

Text-only -- code submissions are graded by a separate sandboxed test runner
that doesn't exist yet.

The models are Ridge *regressions*, not classifiers: rubric scores are ordinal,
and classification throws that ordering away (predicting 0 for a true 5 costs
exactly as much as predicting 4). See features.py for why the embedding alone
isn't enough.
"""

import numpy as np

from app.grading.embedder import embed
from app.grading.features import extract as extract_features
from app.grading.model_store import load_model


class GradingError(Exception):
    """A rubric can't be auto-graded yet (e.g. a criterion has no trained
    model). Callers fall back to status='grading_failed' so a teacher can
    grade by hand."""


def build_matrix(texts):
    """Embedding + hand-crafted features, in the one order every model is
    trained on. The training script imports this too, so the two paths can't
    silently drift apart."""
    return np.hstack([embed(texts), extract_features(texts)])


def grade_text_submission(content, criteria):
    """criteria: RubricCriterion rows. Returns {criterion_id: predicted_score}.
    The submission is featurised once and reused across every criterion's
    model, since it's the same text being scored against each rubric line."""
    missing = [c.name for c in criteria if load_model(c.id) is None]
    if missing:
        raise GradingError(f"No trained model for: {', '.join(missing)}")

    X = build_matrix([content])

    scores = {}
    for criterion in criteria:
        payload = load_model(criterion.id)
        raw = float(payload["model"].predict(X)[0])
        # Ridge predicts a continuous value; clamp it into the criterion's range
        # and round to whole points, which is how rubric scores are expressed.
        scores[criterion.id] = float(np.clip(round(raw), 0, payload["max_points"]))
    return scores
