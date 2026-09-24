"""The grading engine: feature extraction, the model store, the score bands,
and the auto-grade wiring on submission."""

import numpy as np
import pytest

from app.grading import feedback, features


@pytest.fixture
def criteria(rubric, criterion_ids):
    """The persisted Thesis (/5, with a description) and Evidence (/10, without)."""
    from app.extensions import db
    from app.models import RubricCriterion

    return [db.session.get(RubricCriterion, criterion_id) for criterion_id in criterion_ids]


class TestFeatureExtraction:
    def test_shape_matches_the_declared_names(self):
        assert features.extract(["one", "two"]).shape == (2, len(features.FEATURE_NAMES))

    def test_counts_concrete_detail(self):
        column = {name: i for i, name in enumerate(features.FEATURE_NAMES)}
        row = features.extract(
            ["A 2021 Stanford study found that 32% of respondents agreed, which shows a clear effect."]
        )[0]
        assert row[column["year_count"]] == 1
        assert row[column["has_percent"]] == 1
        assert row[column["citation_count"]] >= 2   # "study", "found"
        assert row[column["reasoning_count"]] >= 1  # "which"/"shows"
        assert row[column["proper_noun_count"]] >= 1

    def test_counts_hedging(self):
        column = {name: i for i, name in enumerate(features.FEATURE_NAMES)}
        row = features.extract(["There are many things that could probably seem interesting overall."])[0]
        assert row[column["hedge_count"]] >= 4
        assert row[column["digit_count"]] == 0
        assert row[column["has_percent"]] == 0

    def test_handles_an_empty_string(self):
        row = features.extract([""])[0]
        assert row[features.FEATURE_NAMES.index("word_count")] == 0
        assert row[features.FEATURE_NAMES.index("mean_word_length")] == 0


class TestScoreBands:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (0, "needs significant improvement"),
            (2, "needs significant improvement"),
            (2.5, "partially meets expectations"),
            (3, "partially meets expectations"),
            (4, "solid"),
            (4.5, "excellent"),
            (5, "excellent"),
        ],
    )
    def test_bands_across_the_range(self, criteria, score, expected):
        thesis = criteria[0]
        assert expected in feedback.for_criterion(thesis, score)

    def test_names_the_criterion_and_the_score(self, criteria):
        text = feedback.for_criterion(criteria[0], 4)
        assert "Thesis" in text and "4/5" in text

    def test_quotes_the_criterion_when_there_is_room_to_improve(self, criteria):
        assert "A clear, arguable claim." in feedback.for_criterion(criteria[0], 1)

    def test_does_not_quote_it_when_the_score_is_strong(self, criteria):
        assert "arguable claim" not in feedback.for_criterion(criteria[0], 5)

    def test_copes_with_a_criterion_that_has_no_description(self, criteria):
        text = feedback.for_criterion(criteria[1], 3)
        assert "Evidence" in text and "3/10" in text

    def test_a_zero_point_criterion_does_not_divide_by_zero(self, criteria):
        thesis = criteria[0]
        thesis.max_points = 0
        assert feedback.for_criterion(thesis, 0)


class TestSummary:
    def test_leads_with_the_total_then_names_both_ends(self, criteria, criterion_ids):
        thesis, evidence = criterion_ids
        line = feedback.summary(criteria, {thesis: 4, evidence: 2})
        assert "Overall 6/15" in line
        assert "Strongest: Thesis" in line
        assert "Weakest: Evidence" in line

    def test_omits_both_ends_when_every_ratio_matches(self, criteria, criterion_ids):
        thesis, evidence = criterion_ids
        line = feedback.summary(criteria, {thesis: 5, evidence: 10})
        assert "Overall 15/15" in line
        assert "Strongest" not in line

    def test_omits_both_ends_for_a_single_criterion(self, criteria, criterion_ids):
        line = feedback.summary(criteria[:1], {criterion_ids[0]: 3})
        assert "Overall 3/5" in line
        assert "Strongest" not in line

    def test_is_empty_when_nothing_was_scored(self, criteria):
        assert feedback.summary(criteria, {}) == ""


class TestModelStore:
    def test_round_trips_a_model(self):
        from sklearn.linear_model import Ridge

        from app.grading.model_store import load_model, save_model

        model = Ridge().fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
        save_model(4242, {"model": model, "max_points": 5.0})

        loaded = load_model(4242)
        assert loaded["max_points"] == 5.0
        # The reloaded model must behave identically to the one that was saved.
        probe = np.array([[0.0], [0.5], [1.0]])
        assert loaded["model"].predict(probe) == pytest.approx(model.predict(probe))

    def test_returns_none_for_an_untrained_criterion(self):
        from app.grading.model_store import load_model

        assert load_model(999_999) is None

    def test_saving_again_replaces_the_cached_copy(self):
        from sklearn.linear_model import Ridge

        from app.grading.model_store import load_model, save_model

        save_model(4243, {"model": Ridge().fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0])), "max_points": 1.0})
        load_model(4243)  # warm the cache
        save_model(4243, {"model": Ridge().fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0])), "max_points": 9.0})
        assert load_model(4243)["max_points"] == 9.0

    def test_writes_into_the_isolated_directory_not_the_real_one(self):
        """Guards the fixture that stops a test run destroying trained models."""
        import os

        from app.grading import model_store

        real = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_models")
        assert os.path.abspath(model_store.MODEL_DIR) != real


class TestEngineWithoutModels:
    def test_raises_when_a_criterion_has_no_model(self, criteria):
        from app.grading.engine import GradingError, grade_text_submission

        with pytest.raises(GradingError) as excinfo:
            grade_text_submission("some text", criteria)
        assert "Thesis" in str(excinfo.value)

    def test_submission_is_marked_failed_and_gets_no_grades(self, client, auth, student, rubric, submit):
        submission_id = submit(student)
        body = client.get(f"/api/submissions/{submission_id}", headers=auth(student)).get_json()
        assert body["status"] == "grading_failed"
        assert body["grades"] == []

    def test_the_submission_itself_still_succeeds(self, client, auth, student, rubric):
        """Grading is best-effort -- a student's work is never lost to an engine fault."""
        response = client.post(
            "/api/submissions", json={"rubric_id": rubric["id"], "content": "text"}, headers=auth(student)
        )
        assert response.status_code == 201


@pytest.fixture
def trained_models(criteria):
    """Train a real model per criterion so the auto-grade path runs for real
    instead of being stubbed. Lands in the isolated model directory."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from app.grading.engine import build_matrix
    from app.grading.model_store import save_model

    texts = [
        "A 2021 study of 16000 people found a 13% improvement, which directly supports the claim.",
        "Research from Pew found that 64% of documented cases originated there, showing a clear link.",
        "Everyone knows this is true because it just makes sense if you think about it.",
        "There are many things people say about this topic and some of them seem interesting.",
    ]
    matrix = build_matrix(texts)
    for criterion in criteria:
        top = float(criterion.max_points)
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(matrix, np.array([top, top, 0.0, 0.0]))
        save_model(criterion.id, {"model": model, "max_points": top, "n_examples": 4})
    return criteria


class TestAutoGradeEndToEnd:
    """Exercises the real engine: SBERT embedding, the fitted models, the score
    clamp and the generated feedback, all through the submission endpoint."""

    def test_a_submission_is_scored_and_commented_on(
        self, client, auth, teacher, student, rubric, trained_models, criterion_ids
    ):
        response = client.post(
            "/api/submissions",
            json={"rubric_id": rubric["id"],
                  "content": "A 2020 study found that 41% of participants improved, which supports the claim."},
            headers=auth(student),
        )
        assert response.status_code == 201
        assert response.get_json()["status"] == "ai_graded"

        body = client.get(f"/api/submissions/{response.get_json()['id']}", headers=auth(teacher)).get_json()
        assert body["ai_summary"]
        for grade in body["grades"]:
            assert grade["ai_score"] is not None
            assert grade["ai_feedback"]

    def test_scores_stay_inside_each_criterions_range(
        self, client, auth, teacher, student, rubric, trained_models, criterion_ids
    ):
        thesis, evidence = criterion_ids
        submission_id = client.post(
            "/api/submissions", json={"rubric_id": rubric["id"], "content": "Some answer."}, headers=auth(student)
        ).get_json()["id"]

        body = client.get(f"/api/submissions/{submission_id}", headers=auth(teacher)).get_json()
        by_criterion = {g["criterion_id"]: g["ai_score"] for g in body["grades"]}
        assert 0 <= by_criterion[thesis] <= 5
        assert 0 <= by_criterion[evidence] <= 10

    def test_stronger_writing_scores_at_least_as_well_as_weaker(
        self, client, auth, teacher, student, other_student, rubric, trained_models
    ):
        strong = ("A 2021 study of 16000 people found a 13% improvement, which directly supports "
                  "the claim that the intervention works.")
        weak = "There are many things people say and some of them could probably seem interesting."

        totals = {}
        for token, text, label in ((student, strong, "strong"), (other_student, weak, "weak")):
            submission_id = client.post(
                "/api/submissions", json={"rubric_id": rubric["id"], "content": text}, headers=auth(token)
            ).get_json()["id"]
            totals[label] = client.get(
                f"/api/submissions/{submission_id}", headers=auth(teacher)
            ).get_json()["ai_total"]

        assert totals["strong"] >= totals["weak"]
