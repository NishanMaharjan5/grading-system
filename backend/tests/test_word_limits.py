"""Word limits on text rubrics.

These exist for a grading reason as well as a pedagogical one: the embedder
reads ~226 words and silently drops the rest, so without a cap a long essay is
scored on a partial read with nothing to show it happened.

The server is the authority. The submission form mirrors the same rule (see
frontend/src/grading/wordCount.js) so a student is told before pressing the
button, but every check below goes through the API.
"""

import pytest

from app.extensions import db
from app.models import Rubric
from app.word_limits import (DEFAULT_MAX_WORDS, GRADER_WORD_LIMIT, check_submission_length,
                             count_words, describe_range, validate_limits)


def words(n):
    """An n-word essay."""
    return " ".join(["word"] * n)


@pytest.fixture
def limited_rubric(client, auth, teacher):
    """A rubric accepting 20-200 words, the backfilled default."""
    response = client.post("/api/rubrics", headers=auth(teacher), json={
        "title": "Limited", "type": "text", "min_words": 20, "max_words": 200,
        "criteria": [{"name": "Thesis", "max_points": 5}],
    })
    assert response.status_code == 201, response.get_json()
    return response.get_json()


class TestCountingIsOneRule:
    """The counter in the box, the message and the server must agree, or a
    student cannot tell what the rule is."""

    @pytest.mark.parametrize("text,expected", [
        ("", 0), ("   ", 0), ("one", 1), ("one two three", 3),
        ("line\nbreaks\tand   runs", 4), ("  padded  ", 1), (None, 0),
    ])
    def test_counts_whitespace_separated_tokens(self, text, expected):
        assert count_words(text) == expected

    def test_describes_each_kind_of_range(self):
        assert describe_range(20, 200) == "between 20 and 200 words"
        assert describe_range(None, 200) == "at most 200 words"
        assert describe_range(20, None) == "at least 20 words"
        assert describe_range(None, None) is None


class TestTheDefaultStaysUnderTheGrader:
    def test_default_is_below_the_truncation_point(self):
        """The whole reason for the default: above this the grader stops
        reading and the score is based on a partial essay."""
        assert DEFAULT_MAX_WORDS < GRADER_WORD_LIMIT


class TestSubmissionEnforcement:
    def test_a_submission_inside_the_range_is_accepted(self, client, auth, student, limited_rubric):
        response = client.post("/api/submissions", headers=auth(student),
                               json={"rubric_id": limited_rubric["id"], "content": words(50)})
        assert response.status_code == 201

    def test_too_short_is_refused(self, client, auth, student, limited_rubric):
        response = client.post("/api/submissions", headers=auth(student),
                               json={"rubric_id": limited_rubric["id"], "content": words(19)})
        assert response.status_code == 422
        detail = response.get_json()["detail"]
        assert "between 20 and 200 words" in detail and "too short" in detail

    def test_too_long_is_refused(self, client, auth, student, limited_rubric):
        response = client.post("/api/submissions", headers=auth(student),
                               json={"rubric_id": limited_rubric["id"], "content": words(201)})
        assert response.status_code == 422
        detail = response.get_json()["detail"]
        assert "between 20 and 200 words" in detail and "too long" in detail

    def test_the_message_says_how_long_the_essay_actually_is(self, client, auth, student, limited_rubric):
        response = client.post("/api/submissions", headers=auth(student),
                               json={"rubric_id": limited_rubric["id"], "content": words(300)})
        assert "Yours is 300" in response.get_json()["detail"]

    @pytest.mark.parametrize("n", [20, 200])
    def test_the_boundaries_themselves_are_allowed(self, client, auth, register, limited_rubric, n):
        """Inclusive: "between 20 and 200" has to accept 20 and 200."""
        other = register(f"boundary{n}@example.com")  # a fresh student: one submission per rubric
        response = client.post("/api/submissions", headers=auth(other),
                               json={"rubric_id": limited_rubric["id"], "content": words(n)})
        assert response.status_code == 201, response.get_json()

    def test_a_refused_submission_is_not_stored(self, client, auth, student, limited_rubric):
        client.post("/api/submissions", headers=auth(student),
                    json={"rubric_id": limited_rubric["id"], "content": words(500)})
        assert client.get("/api/submissions", headers=auth(student)).get_json() == []

    def test_a_300_word_essay_is_refused_rather_than_silently_truncated(
            self, client, auth, student, limited_rubric):
        """The case that motivated the feature. Before this, a 300-word essay
        was accepted and graded on roughly its first 226 words, with nothing
        anywhere saying the rest had been ignored."""
        response = client.post("/api/submissions", headers=auth(student),
                               json={"rubric_id": limited_rubric["id"], "content": words(300)})
        assert response.status_code == 422
        assert client.get("/api/submissions", headers=auth(student)).get_json() == []


class TestRubricsWithoutLimitsAreUnrestricted:
    def test_a_rubric_with_null_limits_accepts_any_length(self, client, auth, student, rubric):
        """Every rubric that predates this feature has null limits and must
        keep working exactly as before."""
        assert rubric["min_words"] is None and rubric["max_words"] is None
        response = client.post("/api/submissions", headers=auth(student),
                               json={"rubric_id": rubric["id"], "content": words(1000)})
        assert response.status_code == 201

    def test_a_code_rubric_cannot_be_given_word_limits(self, client, auth, teacher):
        """Word counts say nothing about a program, and the limit would never
        be enforced -- so it is refused rather than stored and ignored."""
        response = client.post("/api/rubrics", headers=auth(teacher), json={
            "title": "Code with limits", "type": "code", "max_words": 200,
            "criteria": [{"name": "Correctness", "max_points": 10,
                          "test_cases": [{"stdin": "", "expected_output": "x"}]}],
        })
        assert response.status_code == 422
        assert "text rubrics" in response.get_json()["detail"]


class TestExistingSubmissionsAreLeftAlone:
    def test_a_limit_set_later_does_not_invalidate_earlier_work(
            self, client, auth, teacher, student, rubric):
        """A teacher adding a limit must not retroactively flag work that was
        acceptable when it was submitted."""
        long_essay = words(400)
        created = client.post("/api/submissions", headers=auth(student),
                              json={"rubric_id": rubric["id"], "content": long_essay})
        assert created.status_code == 201
        submission_id = created.get_json()["id"]

        db.session.get(Rubric, rubric["id"]).max_words = 200
        db.session.commit()

        fetched = client.get(f"/api/submissions/{submission_id}", headers=auth(student))
        assert fetched.status_code == 200
        assert fetched.get_json()["content"] == long_essay


class TestRubricValidation:
    def test_limits_are_returned_to_students_too(self, client, auth, student, limited_rubric):
        """The submission form needs the numbers to show the range and block."""
        body = client.get(f"/api/rubrics/{limited_rubric['id']}", headers=auth(student)).get_json()
        assert (body["min_words"], body["max_words"]) == (20, 200)

    def test_an_inverted_range_is_refused(self, client, auth, teacher):
        response = client.post("/api/rubrics", headers=auth(teacher), json={
            "title": "Inverted", "type": "text", "min_words": 300, "max_words": 100,
            "criteria": [{"name": "Thesis", "max_points": 5}],
        })
        assert response.status_code == 422
        assert "cannot be greater than" in response.get_json()["detail"]

    def test_a_negative_limit_is_refused(self, client, auth, teacher):
        response = client.post("/api/rubrics", headers=auth(teacher), json={
            "title": "Negative", "type": "text", "min_words": -5,
            "criteria": [{"name": "Thesis", "max_points": 5}],
        })
        assert response.status_code == 422

    def test_a_limit_above_the_grader_limit_is_allowed(self, client, auth, teacher):
        """The teacher's call. The form warns; the API does not refuse."""
        response = client.post("/api/rubrics", headers=auth(teacher), json={
            "title": "Long form", "type": "text", "max_words": 1000,
            "criteria": [{"name": "Thesis", "max_points": 5}],
        })
        assert response.status_code == 201
        assert response.get_json()["max_words"] == 1000

    def test_omitting_limits_leaves_them_null(self, client, auth, teacher):
        response = client.post("/api/rubrics", headers=auth(teacher), json={
            "title": "No limits", "type": "text",
            "criteria": [{"name": "Thesis", "max_points": 5}],
        })
        assert response.get_json()["min_words"] is None
        assert response.get_json()["max_words"] is None

    def test_an_edit_can_clear_a_limit(self, client, auth, teacher, limited_rubric):
        response = client.put(f"/api/rubrics/{limited_rubric['id']}", headers=auth(teacher),
                              json={"min_words": None, "max_words": None})
        assert response.status_code == 200
        assert response.get_json()["max_words"] is None

    def test_an_edit_that_ignores_the_fields_keeps_them(self, client, auth, teacher, limited_rubric):
        response = client.put(f"/api/rubrics/{limited_rubric['id']}", headers=auth(teacher),
                              json={"description": "Changed"})
        assert response.status_code == 200
        assert (response.get_json()["min_words"], response.get_json()["max_words"]) == (20, 200)


class TestValidateLimitsUnit:
    @pytest.mark.parametrize("min_words,max_words", [(None, None), (0, 1), (20, 200), (None, 5), (5, None)])
    def test_accepts_sensible_pairs(self, min_words, max_words):
        assert validate_limits(min_words, max_words) is None

    @pytest.mark.parametrize("min_words,max_words", [(-1, None), (None, 0), (10, 5), (True, None)])
    def test_rejects_impossible_pairs(self, min_words, max_words):
        assert validate_limits(min_words, max_words) is not None


class TestCheckSubmissionLength:
    class FakeRubric:
        def __init__(self, type="text", min_words=None, max_words=None):
            self.type, self.min_words, self.max_words = type, min_words, max_words

    def test_code_rubrics_are_never_length_checked(self):
        rubric = self.FakeRubric(type="code", max_words=5)
        assert check_submission_length(words(500), rubric) is None

    def test_no_limits_means_no_check(self):
        assert check_submission_length(words(5000), self.FakeRubric()) is None

    def test_only_a_maximum(self):
        rubric = self.FakeRubric(max_words=10)
        assert check_submission_length(words(5), rubric) is None
        assert "at most 10 words" in check_submission_length(words(11), rubric)

    def test_only_a_minimum(self):
        rubric = self.FakeRubric(min_words=10)
        assert check_submission_length(words(50), rubric) is None
        assert "at least 10 words" in check_submission_length(words(9), rubric)
