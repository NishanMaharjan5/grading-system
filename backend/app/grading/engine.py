"""Runs the trained per-criterion classifiers against a submission's text.
Text-only -- code submissions are graded by a separate sandboxed test runner
that doesn't exist yet."""

from app.grading.embedder import embed
from app.grading.model_store import load_classifier


class GradingError(Exception):
    """A rubric can't be auto-graded yet (e.g. a criterion has no trained
    classifier). Callers fall back to status='grading_failed' so a teacher
    can grade by hand."""


def grade_text_submission(content, criteria):
    """criteria: RubricCriterion rows. Returns {criterion_id: predicted_score}.
    The submission is embedded once and reused across every criterion's
    classifier, since it's the same text being scored against each rubric line."""
    missing = [c.name for c in criteria if load_classifier(c.id) is None]
    if missing:
        raise GradingError(f"No trained model for: {', '.join(missing)}")

    vector = embed([content])[0]
    return {c.id: float(load_classifier(c.id)["model"].predict([vector])[0]) for c in criteria}
