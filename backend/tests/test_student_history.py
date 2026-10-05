"""The data behind a student's "your submissions" view.

The page lists everything one student has ever handed in, across every rubric,
with its current status and grade. It is built on GET /api/submissions, which
already scopes to the signed-in student -- these tests pin that scoping and
the fields the page depends on, because a leak here would show one student
another's work and grades.
"""

import pytest

from app.extensions import db
from app.models import Submission


@pytest.fixture
def second_rubric(client, auth, teacher):
    return client.post("/api/rubrics", headers=auth(teacher), json={
        "title": "Second assignment", "type": "text",
        "criteria": [{"name": "Thesis", "max_points": 5}],
    }).get_json()


def submit_to(client, auth, token, rubric_id, content="An essay for the history view."):
    return client.post("/api/submissions", headers=auth(token),
                       json={"rubric_id": rubric_id, "content": content})


class TestScoping:
    def test_a_student_sees_only_their_own(self, client, auth, student, other_student, rubric, submit):
        submit(student)
        submit(other_student)
        mine = client.get("/api/submissions", headers=auth(student)).get_json()
        assert len(mine) == 1
        assert all(s["student_id"] is not None for s in mine)

    def test_another_students_content_never_appears(
            self, client, auth, student, other_student, rubric):
        submit_to(client, auth, other_student, rubric["id"], "Their private essay.")
        submit_to(client, auth, student, rubric["id"], "My own essay.")
        mine = client.get("/api/submissions", headers=auth(student)).get_json()
        assert [s["content"] for s in mine] == ["My own essay."]

    def test_a_student_is_not_sent_ai_scores_for_their_own_work(
            self, client, auth, student, rubric, submit, seed_ai_grade, criterion_ids):
        """Unapproved AI scores stay invisible here too, not just on the
        single-submission view."""
        submission_id = submit(student)
        thesis, evidence = criterion_ids
        seed_ai_grade(submission_id, {thesis: 4, evidence: 9})
        row = client.get("/api/submissions", headers=auth(student)).get_json()[0]
        assert "ai_total" not in row and "ai_summary" not in row
        assert all("ai_score" not in g for g in row["grades"])


class TestSpansEveryRubric:
    def test_submissions_to_different_rubrics_all_appear(
            self, client, auth, student, rubric, second_rubric):
        submit_to(client, auth, student, rubric["id"])
        submit_to(client, auth, student, second_rubric["id"])
        rows = client.get("/api/submissions", headers=auth(student)).get_json()
        assert {r["rubric_id"] for r in rows} == {rubric["id"], second_rubric["id"]}

    def test_each_row_identifies_its_rubric(self, client, auth, student, rubric, submit):
        """The page needs this to show a title and a total."""
        submit(student)
        row = client.get("/api/submissions", headers=auth(student)).get_json()[0]
        assert row["rubric_id"] == rubric["id"]
        assert "status" in row and "created_at" in row


class TestStatusAndGrade:
    def test_an_ungraded_submission_has_no_final_total(self, client, auth, student, rubric, submit):
        submit(student)
        row = client.get("/api/submissions", headers=auth(student)).get_json()[0]
        assert row["final_total"] is None

    def test_an_approved_submission_shows_its_grade(
            self, client, auth, teacher, student, rubric, submit, seed_ai_grade, criterion_ids):
        submission_id = submit(student)
        thesis, evidence = criterion_ids
        seed_ai_grade(submission_id, {thesis: 3, evidence: 7})
        client.put(f"/api/submissions/{submission_id}/review", json={}, headers=auth(teacher))
        row = client.get("/api/submissions", headers=auth(student)).get_json()[0]
        assert row["status"] == "approved" and row["final_total"] == 10


class TestItKeepsUp:
    def test_it_reflects_a_resubmission(self, client, auth, student, rubric, submit):
        submit(student)
        submit_to(client, auth, student, rubric["id"], "The replacement essay.")
        rows = client.get("/api/submissions", headers=auth(student)).get_json()
        assert len(rows) == 1, "overwritten, not duplicated"
        assert rows[0]["content"] == "The replacement essay."

    def test_it_reflects_a_later_revision(
            self, client, auth, teacher, student, rubric, submit, seed_ai_grade, criterion_ids):
        """A corrected grade must show the corrected figure, not the one the
        student first saw."""
        submission_id = submit(student)
        thesis, evidence = criterion_ids
        seed_ai_grade(submission_id, {thesis: 3, evidence: 7})
        client.put(f"/api/submissions/{submission_id}/review", json={}, headers=auth(teacher))
        assert client.get("/api/submissions", headers=auth(student)).get_json()[0]["final_total"] == 10

        client.put(f"/api/submissions/{submission_id}/revise", headers=auth(teacher), json={
            "criterion_scores": [{"criterion_id": thesis, "final_score": 5},
                                 {"criterion_id": evidence, "final_score": 9}]})
        assert client.get("/api/submissions", headers=auth(student)).get_json()[0]["final_total"] == 14

    def test_a_student_with_nothing_submitted_gets_an_empty_list(self, client, auth, student):
        assert client.get("/api/submissions", headers=auth(student)).get_json() == []
