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


class TestTypography:
    """Generated text is shown to students, so it uses a real dash rather than
    the double hyphen it was written with."""

    @pytest.mark.parametrize("score", [0, 2, 3, 4, 5])
    def test_criterion_feedback_has_no_double_hyphen(self, criteria, score):
        assert "--" not in feedback.for_criterion(criteria[0], score)

    def test_summary_has_no_double_hyphen(self, criteria, criterion_ids):
        thesis, evidence = criterion_ids
        assert "--" not in feedback.summary(criteria, {thesis: 4, evidence: 2})


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


class TestEngineWithoutTheModel:
    """The weights are not in git, so "not installed" is a state a real clone
    is in, not a hypothetical. It must degrade to grading_failed, never to a
    guess."""

    def test_raises_when_the_scorer_is_not_installed(self, criteria):
        from app.grading.engine import GradingError, grade_text_submission

        with pytest.raises(GradingError) as excinfo:
            grade_text_submission("some text", criteria)
        assert "not installed" in str(excinfo.value)

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


class TestCriterionLookupIsByName:
    """The point of the Phase 2 rewrite: the text model is found by criterion
    *name*, so re-seeding the database cannot orphan it."""

    def test_an_unvalidated_criterion_name_is_refused_not_guessed(self, bert_model):
        from app.grading.bert_scorer import describes

        assert describes("Thesis") is not None
        assert describes("Evidence") is not None
        # The criterion the rubric-conditioning experiment showed this
        # architecture scores by general essay quality rather than by the
        # criterion itself. It must not be silently scored.
        assert describes("Counterargument") is None

    def test_engine_refuses_a_rubric_it_was_never_validated_on(self, app, bert_model):
        from app.extensions import db
        from app.grading.engine import GradingError, grade_text_submission
        from app.models import Rubric, RubricCriterion, User
        from app.security import hash_password

        teacher = User(name="T", email="lookup@example.com",
                       password_hash=hash_password("password1"), role="teacher")
        db.session.add(teacher)
        db.session.flush()
        rubric = Rubric(title="New rubric", type="text", created_by=teacher.id)
        rubric.criteria.append(RubricCriterion(name="Counterargument", max_points=5, position=0))
        db.session.add(rubric)
        db.session.commit()

        with pytest.raises(GradingError) as excinfo:
            grade_text_submission("An essay.", list(rubric.criteria))
        assert "Counterargument" in str(excinfo.value)

    def test_the_same_criterion_scores_the_same_under_different_row_ids(self, app, bert_model):
        """The reset-dev orphaning bug, directly.

        An identical rubric recreated under *different* criterion row ids must
        score identically. The old store was keyed `criterion_<id>.joblib`, so
        new ids meant no model file and everything fell to grading_failed. The
        ids have to actually differ for this to prove anything, so the second
        rubric is created after the first rather than after a truncate (which
        restarts the sequence and would hand back the same ids).
        """
        from app.extensions import db
        from app.grading.engine import grade_text_submission
        from app.models import Rubric, RubricCriterion, User
        from app.security import hash_password

        essay = ("Governments should require platforms to open their recommendation systems to "
                 "independent audit, because a 2021 disclosure showed internal research being "
                 "withheld, which means self-reporting cannot be relied on.")

        def build_and_score(email, title):
            teacher = User(name="T", email=email,
                           password_hash=hash_password("password1"), role="teacher")
            db.session.add(teacher)
            db.session.flush()
            rubric = Rubric(title=title, type="text", created_by=teacher.id)
            rubric.criteria.append(RubricCriterion(name="Thesis", max_points=5, position=0))
            rubric.criteria.append(RubricCriterion(name="Evidence", max_points=10, position=1))
            db.session.add(rubric)
            db.session.commit()
            criteria = list(rubric.criteria)
            result = grade_text_submission(essay, criteria)
            return {c.name: result["scores"][c.id] for c in criteria}, sorted(c.id for c in criteria)

        before, before_ids = build_and_score("reseed.a@example.com", "Essay 1")
        after, after_ids = build_and_score("reseed.b@example.com", "Essay 1 again")

        assert before_ids != after_ids, (
            "the two rubrics got the same criterion ids, so this test proved nothing")
        assert before == after, (
            f"the same essay scored differently under new row ids {before_ids} -> {after_ids}: "
            f"{before} -> {after}")
        assert set(before) == {"Thesis", "Evidence"}

    def test_a_truncate_and_reseed_still_grades(self, app, bert_model):
        """`make reset-dev` truncates with RESTART IDENTITY and re-seeds. The
        rubric comes back with fresh rows; grading must still work rather than
        silently falling to grading_failed as it did when models were
        id-keyed."""
        from app.extensions import db
        from app.grading.engine import grade_text_submission
        from app.models import Rubric, RubricCriterion, User
        from app.security import hash_password
        from sqlalchemy import text as sql_text

        tables = ", ".join(t.name for t in db.metadata.sorted_tables)
        db.session.execute(sql_text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
        db.session.commit()

        teacher = User(name="T", email="afterreset@example.com",
                       password_hash=hash_password("password1"), role="teacher")
        db.session.add(teacher)
        db.session.flush()
        rubric = Rubric(title="Essay 1", type="text", created_by=teacher.id)
        rubric.criteria.append(RubricCriterion(name="Thesis", max_points=5, position=0))
        rubric.criteria.append(RubricCriterion(name="Evidence", max_points=10, position=1))
        db.session.add(rubric)
        db.session.commit()

        criteria = list(rubric.criteria)
        result = grade_text_submission("A clear claim, supported by a 2021 study of 16,000 people.", criteria)
        assert set(result["scores"]) == {c.id for c in criteria}
        for criterion in criteria:
            assert 0 <= result["scores"][criterion.id] <= float(criterion.max_points)


class TestAutoGradeEndToEnd:
    """Exercises the real engine through the submission endpoint: the
    fine-tuned model, the score clamp and the generated feedback.

    The Ridge models these tests used to train first are gone from this path --
    the text scorer no longer consults them -- so the only thing that has to be
    present is the fine-tuned model, via the `bert_model` fixture.
    """

    def test_a_submission_is_scored_and_commented_on(
        self, client, auth, teacher, student, rubric, bert_model, criterion_ids
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
        self, client, auth, teacher, student, rubric, bert_model, criterion_ids
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
        self, client, auth, teacher, student, other_student, rubric, bert_model
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


class TestBertScorer:
    """The scoring module itself, below the engine."""

    def test_loads_once_and_is_reused(self, bert_model):
        model_a, tokenizer_a = bert_model.load()
        model_b, tokenizer_b = bert_model.load()
        assert model_a is model_b and tokenizer_a is tokenizer_b

    def test_is_in_eval_mode(self, bert_model):
        """Dropout at inference would make the same essay score differently on
        two submissions, which a student would experience as the grader
        changing its mind."""
        model, _ = bert_model.load()
        assert model.training is False

    def test_the_same_text_scores_identically_twice(self, bert_model):
        class C:
            def __init__(self, name, max_points):
                self.name, self.max_points = name, max_points

        criteria = [C("Thesis", 5), C("Evidence", 10)]
        text = "A clear claim supported by a 2021 study of 16,000 people, which shows the effect."
        assert bert_model.score(text, criteria) == bert_model.score(text, criteria)

    def test_scores_are_whole_numbers_inside_each_range(self, bert_model):
        class C:
            def __init__(self, name, max_points):
                self.name, self.max_points = name, max_points

        for text in ("", "x", "A clear claim supported by a 2021 study.", "word " * 600):
            scores = bert_model.score(text, [C("Thesis", 5), C("Evidence", 10)])
            assert scores["Thesis"] == int(scores["Thesis"]) and 0 <= scores["Thesis"] <= 5
            assert scores["Evidence"] == int(scores["Evidence"]) and 0 <= scores["Evidence"] <= 10

    def test_an_essay_longer_than_the_window_still_scores(self, bert_model):
        """Truncation is only_first, so a long essay is cut and the criterion
        segment survives -- the alternative silently scores against a
        half-eaten criterion description."""
        class C:
            def __init__(self, name, max_points):
                self.name, self.max_points = name, max_points

        scores = bert_model.score("regulation " * 2000, [C("Thesis", 5)])
        assert 0 <= scores["Thesis"] <= 5

    def test_refuses_an_unvalidated_criterion(self, bert_model):
        class C:
            def __init__(self, name, max_points):
                self.name, self.max_points = name, max_points

        with pytest.raises(KeyError):
            bert_model.score("An essay.", [C("Counterargument", 5)])

    def test_descriptions_match_what_the_model_was_trained_on(self, bert_model):
        """The descriptions are part of the model's input. If they drift from
        the ones in the training file, the model is being asked a question it
        was never trained on, silently."""
        import json
        import os

        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "training_data", "rubric_conditioning_train.json")
        trained_on = {row["criterion"]: row["criterion_description"] for row in json.load(open(path))}
        for name, description in bert_model.CRITERION_DESCRIPTIONS.items():
            assert trained_on[name] == description, f"{name}'s description has drifted from training"
