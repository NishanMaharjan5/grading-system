"""Rubric deadlines.

A rubric with no due date never blocks anything, which is what every rubric
created before deadlines were enforced keeps. The server is the authority: a
page left open across the deadline must not be able to submit.

Note there was no migration for this. `due_date` has been on the rubrics table
since the baseline migration and was already accepted and returned by the API;
what was missing was anywhere to set it and anything that honoured it.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.deadlines import check_deadline, is_past_due, parse_due_date
from app.extensions import db
from app.models import Rubric


def iso(**delta):
    return (datetime.now(timezone.utc) + timedelta(**delta)).isoformat()


@pytest.fixture
def open_rubric(client, auth, teacher):
    """Due in an hour."""
    response = client.post("/api/rubrics", headers=auth(teacher), json={
        "title": "Open", "type": "text", "due_date": iso(hours=1),
        "criteria": [{"name": "Thesis", "max_points": 5}],
    })
    assert response.status_code == 201, response.get_json()
    return response.get_json()


@pytest.fixture
def closed_rubric(client, auth, teacher):
    """Due an hour ago. Created through the API, so the stored value is one a
    teacher could really have set."""
    response = client.post("/api/rubrics", headers=auth(teacher), json={
        "title": "Closed", "type": "text", "due_date": iso(hours=-1),
        "criteria": [{"name": "Thesis", "max_points": 5}],
    })
    assert response.status_code == 201, response.get_json()
    return response.get_json()


class TestParsing:
    def test_blank_and_null_both_mean_no_deadline(self):
        for value in (None, "", "   "):
            assert parse_due_date(value) == (None, None)

    def test_reads_an_iso_instant(self):
        parsed, error = parse_due_date("2026-10-05T17:00:00+00:00")
        assert error is None and parsed.year == 2026 and parsed.tzinfo is not None

    def test_accepts_a_trailing_z(self):
        parsed, error = parse_due_date("2026-10-05T17:00:00Z")
        assert error is None and parsed.utcoffset().total_seconds() == 0

    def test_a_naive_time_is_read_as_utc_rather_than_rejected(self):
        """The browser's datetime-local field has no zone. Guessing UTC beats
        a 500."""
        parsed, error = parse_due_date("2026-10-05T17:00:00")
        assert error is None and parsed.tzinfo is not None

    @pytest.mark.parametrize("value", ["not a date", "2026-13-45T99:00", 12345, []])
    def test_nonsense_is_refused(self, value):
        parsed, error = parse_due_date(value)
        assert parsed is None and error is not None


class TestIsPastDue:
    class FakeRubric:
        def __init__(self, due_date):
            self.due_date = due_date

    def test_no_deadline_is_never_past_due(self):
        assert is_past_due(self.FakeRubric(None)) is False
        assert check_deadline(self.FakeRubric(None)) is None

    def test_a_future_deadline_is_open(self):
        assert is_past_due(self.FakeRubric(datetime.now(timezone.utc) + timedelta(hours=1))) is False

    def test_a_past_deadline_is_closed(self):
        assert is_past_due(self.FakeRubric(datetime.now(timezone.utc) - timedelta(seconds=1))) is True

    def test_exactly_on_the_deadline_is_still_open(self):
        """'> due', not '>=': a submission landing on the stroke of the
        deadline is on time."""
        moment = datetime.now(timezone.utc)
        assert is_past_due(self.FakeRubric(moment), now=moment) is False


class TestSubmissionEnforcement:
    def test_before_the_deadline_a_submission_succeeds(self, client, auth, student, open_rubric):
        response = client.post("/api/submissions", headers=auth(student),
                               json={"rubric_id": open_rubric["id"], "content": "An essay."})
        assert response.status_code == 201

    def test_after_the_deadline_a_submission_is_refused(self, client, auth, student, closed_rubric):
        response = client.post("/api/submissions", headers=auth(student),
                               json={"rubric_id": closed_rubric["id"], "content": "An essay."})
        assert response.status_code == 422
        assert "deadline" in response.get_json()["detail"].lower()

    def test_a_refused_late_submission_is_not_stored(self, client, auth, student, closed_rubric):
        client.post("/api/submissions", headers=auth(student),
                    json={"rubric_id": closed_rubric["id"], "content": "An essay."})
        assert client.get("/api/submissions", headers=auth(student)).get_json() == []

    def test_a_rubric_with_no_deadline_is_unaffected(self, client, auth, student, rubric):
        """The default, and what every rubric predating this keeps."""
        assert rubric["due_date"] is None
        response = client.post("/api/submissions", headers=auth(student),
                               json={"rubric_id": rubric["id"], "content": "An essay."})
        assert response.status_code == 201

    def test_the_deadline_is_checked_before_the_word_range(self, client, auth, teacher, student):
        """A closed assignment should say it is closed, not complain about
        length -- the length is no longer the student's problem."""
        made = client.post("/api/rubrics", headers=auth(teacher), json={
            "title": "Closed and limited", "type": "text",
            "due_date": iso(hours=-1), "min_words": 50, "max_words": 100,
            "criteria": [{"name": "Thesis", "max_points": 5}],
        }).get_json()
        response = client.post("/api/submissions", headers=auth(student),
                               json={"rubric_id": made["id"], "content": "Too short."})
        assert response.status_code == 422
        assert "deadline" in response.get_json()["detail"].lower()


class TestRubricRoutes:
    def test_a_due_date_round_trips(self, client, auth, teacher, open_rubric):
        body = client.get(f"/api/rubrics/{open_rubric['id']}", headers=auth(teacher)).get_json()
        assert body["due_date"] is not None

    def test_students_see_the_deadline(self, client, auth, student, open_rubric):
        """They need it to show the due date and to stop before submitting."""
        listed = client.get("/api/rubrics", headers=auth(student)).get_json()
        mine = next(r for r in listed if r["id"] == open_rubric["id"])
        assert mine["due_date"] is not None

    def test_a_deadline_can_be_cleared_by_sending_blank(self, client, auth, teacher, open_rubric):
        response = client.put(f"/api/rubrics/{open_rubric['id']}", headers=auth(teacher),
                              json={"due_date": ""})
        assert response.status_code == 200
        assert response.get_json()["due_date"] is None

    def test_an_unparseable_due_date_is_refused(self, client, auth, teacher):
        response = client.post("/api/rubrics", headers=auth(teacher), json={
            "title": "Bad date", "type": "text", "due_date": "whenever",
            "criteria": [{"name": "Thesis", "max_points": 5}],
        })
        assert response.status_code == 422
        assert "due_date" in response.get_json()["detail"]

    def test_omitting_a_due_date_leaves_it_null(self, client, auth, teacher):
        response = client.post("/api/rubrics", headers=auth(teacher), json={
            "title": "No deadline", "type": "text",
            "criteria": [{"name": "Thesis", "max_points": 5}],
        })
        assert response.get_json()["due_date"] is None
