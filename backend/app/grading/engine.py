"""Scores a submission against each rubric criterion.

Text is scored by the fine-tuned DistilBERT in bert_scorer.py -- one shared
model reading the essay and the criterion's description as a sentence pair.
Code is scored by running the criterion's test cases in a sandbox, which this
change does not touch at all.

Criteria are matched to the text model **by name**, not by database row id.
The Ridge models this replaced were keyed by row id, so re-seeding the dev
database orphaned every one of them; that cost seconds to fix then and would
cost a GPU and ~20 minutes now. A criterion whose name the model has not been
validated on is not guessed at -- it raises GradingError and the submission
becomes grading_failed for a teacher to grade by hand.

The Ridge path (build_matrix, model_store, features.py, scripts/train_grader.py)
is left intact but is no longer reached from here. See
ml_models/bert_rubric_scorer/PROVENANCE.md for what the swap was measured to
buy, and what it was measured not to.
"""

import numpy as np

from app.grading import bert_scorer, feedback
from app.grading.code_runner import run_test_cases
from app.grading.embedder import embed
from app.grading.features import extract as extract_features


class GradingError(Exception):
    """A rubric can't be auto-graded yet (e.g. a criterion has no trained
    model). Callers fall back to status='grading_failed' so a teacher can
    grade by hand."""


def build_matrix(texts):
    """Embedding + hand-crafted features, in the one order every Ridge model
    was trained on.

    No longer on the live grading path -- bert_scorer.py replaced it -- but
    kept, along with model_store and scripts/train_grader.py, because the
    benchmark and probe scripts still compare against that pipeline and
    deleting it would throw away the thing the new model is measured against.
    """
    return np.hstack([embed(texts), extract_features(texts)])


def grade_text_submission(content, criteria):
    """criteria: RubricCriterion rows. Scores the text and writes the matching
    commentary in one pass, so a stored score can never end up without feedback.

    Returns {"scores": {criterion_id: score},
             "feedback": {criterion_id: text},
             "summary": str}

    Scored by the fine-tuned model in bert_scorer.py, which matches criteria by
    name. A criterion the model was never validated on raises rather than
    getting a plausible-looking guess: the caller turns that into
    grading_failed, and a teacher grades it by hand. The scores are returned
    keyed by criterion id, as every caller already expects.
    """
    criteria = list(criteria)
    unvalidated = [c.name for c in criteria if bert_scorer.describes(c.name) is None]
    if unvalidated:
        raise GradingError(
            f"The text grader has not been validated on: {', '.join(sorted(unvalidated))}")
    if not bert_scorer.is_available():
        raise GradingError(
            "The fine-tuned text grader is not installed "
            "(see ml_models/bert_rubric_scorer/PROVENANCE.md)")

    by_name = bert_scorer.score(content, criteria)
    scores = {c.id: by_name[c.name] for c in criteria}

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
