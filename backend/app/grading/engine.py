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

from app.grading import feedback
from app.grading.code_runner import run_test_cases
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
    """criteria: RubricCriterion rows. Scores the text and writes the matching
    commentary in one pass, so a stored score can never end up without feedback.

    Returns {"scores": {criterion_id: score},
             "feedback": {criterion_id: text},
             "summary": str}

    The submission is featurised once and reused across every criterion's
    model, since it's the same text being scored against each rubric line.
    """
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

    return {
        "scores": scores,
        "feedback": {c.id: feedback.for_criterion(c, scores[c.id]) for c in criteria},
        "summary": feedback.summary(criteria, scores),
    }


def grade_code_submission(content, criteria):
    """Runs the submitted program against each criterion's test cases.

    A criterion is worth the fraction of its tests the program passes. Failing
    tests -- wrong output, a crash, a timeout -- are a grade, not a grading
    failure; only the harness breaking raises (CodeRunnerError), which the
    caller turns into grading_failed so a teacher grades by hand.
    """
    untestable = [c.name for c in criteria if not c.test_cases]
    if untestable:
        raise GradingError(f"No test cases defined for: {', '.join(untestable)}")

    scores, texts = {}, {}
    for criterion in criteria:
        results = run_test_cases(content, list(criterion.test_cases))
        passed = sum(1 for result in results if result["passed"])
        scores[criterion.id] = round(passed / len(results) * float(criterion.max_points), 2)
        texts[criterion.id] = feedback.for_code_criterion(criterion, results)

    return {"scores": scores, "feedback": texts, "summary": feedback.summary(criteria, scores)}


def grade_submission(rubric, content):
    """Picks the engine that fits the rubric."""
    if rubric.type == "code":
        return grade_code_submission(content, list(rubric.criteria))
    return grade_text_submission(content, list(rubric.criteria))
