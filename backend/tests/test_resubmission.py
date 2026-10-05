"""Resubmission.

A student may replace their work until a teacher releases a grade. The model
is overwrite, not history: one row per student per rubric, re-graded in place,
only the latest attempt kept. Keeping every attempt would mean dropping the
unique constraint and adding an attempt number, which this deliberately does
not do.

Two things close resubmission off: the rubric's deadline passing, and the
teacher approving. Both are tested here.

(tests/test_submissions.py::test_submitting_again_overwrites_rather_than_adding_a_row
documents the behaviour that changed -- it used to assert a flat 409.)
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.extensions import db
from app.models import Grade, Rubric, Submission


def set_status(submission_id, status):
    db.session.get(Submission, submission_id).status = status
    db.session.commit()


def set_due_date(rubric_id, when):
    """Set the deadline directly. Not through the API on purpose: a rubric
    with submissions is edit-locked (409), so a teacher cannot move a deadline
    once anyone has submitted -- see
    TestDeadlinesCannotBeMovedAfterSubmissions below."""
    db.session.get(Rubric, rubric_id).due_date = when
    db.session.commit()


def resubmit(client, auth, student, rubric_id, content):
    return client.post("/api/submissions", headers=auth(student),
                       json={"rubric_id": rubric_id, "content": content})


class TestWhatResubmissionDoes:
    def test_it_overwrites_the_same_row(self, client, auth, student, rubric, submit):
        first = submit(student)
        response = resubmit(client, auth, student, rubric["id"], "A better essay.")
        assert response.status_code == 200
        assert response.get_json()["id"] == first
        assert response.get_json()["content"] == "A better essay."

    def test_the_student_still_has_exactly_one_submission(self, client, auth, student, rubric, submit):
        submit(student)
        resubmit(client, auth, student, rubric["id"], "Another go.")
        assert len(client.get("/api/submissions", headers=auth(student)).get_json()) == 1

    def test_a_stale_ai_grade_from_the_old_text_is_discarded(
            self, client, auth, student, rubric, submit, seed_ai_grade, criterion_ids):
        """The old scores describe text that no longer exists, so they must not
        survive the overwrite."""
        submission_id = submit(student)
        thesis, evidence = criterion_ids
        seed_ai_grade(submission_id, {thesis: 4, evidence: 9})
        stale = {g.id for g in db.session.query(Grade).filter_by(submission_id=submission_id)}
        assert stale

        resubmit(client, auth, student, rubric["id"], "Completely different text.")
        survivors = {g.id for g in db.session.query(Grade).filter_by(submission_id=submission_id)}
        assert not (stale & survivors), "grades for the previous content were kept"

    def test_the_old_summary_is_cleared(self, client, auth, student, rubric, submit, seed_ai_grade, criterion_ids):
        submission_id = submit(student)
        thesis, evidence = criterion_ids
        seed_ai_grade(submission_id, {thesis: 4, evidence: 9})
        db.session.get(Submission, submission_id).final_summary = "Released earlier."
        db.session.commit()

        resubmit(client, auth, student, rubric["id"], "New text entirely.")
        refreshed = db.session.get(Submission, submission_id)
        assert refreshed.final_summary is None

    def test_it_is_graded_again_rather_than_left_as_submitted(
            self, client, auth, student, rubric, submit):
        """Whatever the engine does with the new text, it must not be left in
        the 'submitted' limbo that no queue shows."""
        submit(student)
        body = resubmit(client, auth, student, rubric["id"], "New text to grade.").get_json()
        assert body["status"] in ("ai_graded", "grading_failed")


class TestWhenResubmissionIsAllowed:
    @pytest.mark.parametrize("status", ["ai_graded", "grading_failed", "submitted"])
    def test_allowed_before_a_teacher_releases_a_grade(
            self, client, auth, student, rubric, submit, status):
        submission_id = submit(student)
        set_status(submission_id, status)
        response = resubmit(client, auth, student, rubric["id"], "Revised essay.")
        assert response.status_code == 200, response.get_json()

    def test_refused_once_approved(self, client, auth, student, rubric, submit):
        submission_id = submit(student)
        set_status(submission_id, "approved")
        response = resubmit(client, auth, student, rubric["id"], "Too late.")
        assert response.status_code == 409
        assert response.get_json()["detail"] == "This has already been graded by your teacher"

    def test_an_approved_submission_keeps_its_content(self, client, auth, student, rubric, submit):
        submission_id = submit(student)
        original = db.session.get(Submission, submission_id).content
        set_status(submission_id, "approved")
        resubmit(client, auth, student, rubric["id"], "Should not land.")
        assert db.session.get(Submission, submission_id).content == original

    def test_refused_after_the_deadline(self, client, auth, teacher, student, submit):
        made = client.post("/api/rubrics", headers=auth(teacher), json={
            "title": "Closing soon", "type": "text",
            "due_date": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "criteria": [{"name": "Thesis", "max_points": 5}],
        }).get_json()

        first = client.post("/api/submissions", headers=auth(student),
                            json={"rubric_id": made["id"], "content": "In good time."})
        assert first.status_code == 201

        set_due_date(made["id"], datetime.now(timezone.utc) - timedelta(minutes=1))

        late = client.post("/api/submissions", headers=auth(student),
                           json={"rubric_id": made["id"], "content": "After the bell."})
        assert late.status_code == 422
        assert "deadline" in late.get_json()["detail"].lower()

    def test_the_deadline_closes_it_even_before_approval(self, client, auth, teacher, student):
        """Both gates are independent: being un-approved does not reopen a
        closed assignment."""
        made = client.post("/api/rubrics", headers=auth(teacher), json={
            "title": "Shut", "type": "text",
            "due_date": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "criteria": [{"name": "Thesis", "max_points": 5}],
        }).get_json()
        submission = client.post("/api/submissions", headers=auth(student),
                                 json={"rubric_id": made["id"], "content": "First attempt."}).get_json()
        set_status(submission["id"], "ai_graded")
        set_due_date(made["id"], datetime.now(timezone.utc) - timedelta(minutes=1))
        assert resubmit(client, auth, student, made["id"], "Second attempt.").status_code == 422


class TestResubmissionStillValidates:
    def test_the_word_range_applies_to_a_resubmission_too(self, client, auth, teacher, student):
        made = client.post("/api/rubrics", headers=auth(teacher), json={
            "title": "Ranged", "type": "text", "min_words": 5, "max_words": 10,
            "criteria": [{"name": "Thesis", "max_points": 5}],
        }).get_json()
        client.post("/api/submissions", headers=auth(student),
                    json={"rubric_id": made["id"], "content": "one two three four five six"})
        response = resubmit(client, auth, student, made["id"], "word " * 50)
        assert response.status_code == 422
        assert "too long" in response.get_json()["detail"]

    def test_empty_content_is_still_refused(self, client, auth, student, rubric, submit):
        submit(student)
        assert resubmit(client, auth, student, rubric["id"], "   ").status_code == 422

    def test_another_students_submission_is_untouched(
            self, client, auth, student, other_student, rubric, submit):
        mine = submit(student)
        theirs = submit(other_student)
        assert mine != theirs
        resubmit(client, auth, student, rubric["id"], "Only mine changes.")
        assert db.session.get(Submission, theirs).content != "Only mine changes."


class TestTeacherViewAfterResubmission:
    def test_the_queue_shows_the_new_text(self, client, auth, teacher, student, rubric, submit):
        submit(student)
        resubmit(client, auth, student, rubric["id"], "The text the teacher should read.")
        queue = client.get("/api/submissions/pending", headers=auth(teacher)).get_json()
        assert any(s["content"] == "The text the teacher should read." for s in queue)

    def test_a_resubmission_can_then_be_approved_normally(
            self, client, auth, teacher, student, rubric, submit, criterion_ids):
        submission_id = submit(student)
        resubmit(client, auth, student, rubric["id"], "Final answer.")
        thesis, evidence = criterion_ids
        response = client.put(f"/api/submissions/{submission_id}/review", headers=auth(teacher),
                              json={"criterion_scores": [
                                  {"criterion_id": thesis, "final_score": 4},
                                  {"criterion_id": evidence, "final_score": 8}]})
        assert response.status_code == 200
        assert db.session.get(Submission, submission_id).status == "approved"


class TestDeadlinesCanBeMovedAfterSubmissions:
    """CHANGED: this class used to be TestDeadlinesCannotBeMovedAfterSubmissions
    and recorded the opposite behaviour as a known limitation.

    The rubric edit-lock was all-or-nothing, so one submission froze the whole
    rubric including its deadline -- and a teacher could not extend one after
    work started arriving, which is exactly when they need to. The lock now
    covers only the scoring (criteria, points, type); see
    tests/test_rubric_edit_lock.py for the full split.

    What it protects is unchanged: a grade already recorded still cannot be
    invalidated underneath a student.
    """

    def test_a_deadline_can_be_extended_once_work_exists(self, client, auth, teacher, student):
        made = client.post("/api/rubrics", headers=auth(teacher), json={
            "title": "Extendable", "type": "text",
            "due_date": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "criteria": [{"name": "Thesis", "max_points": 5}],
        }).get_json()
        client.post("/api/submissions", headers=auth(student),
                    json={"rubric_id": made["id"], "content": "Early work."})

        extended = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        response = client.put(f"/api/rubrics/{made['id']}", headers=auth(teacher),
                              json={"due_date": extended})
        assert response.status_code == 200
        assert response.get_json()["due_date"] is not None

    def test_extending_a_passed_deadline_reopens_submissions(self, client, auth, teacher, student):
        """The point of allowing it: a student who missed the deadline can be
        given more time without the rubric having to be recreated."""
        made = client.post("/api/rubrics", headers=auth(teacher), json={
            "title": "Reopened", "type": "text",
            "due_date": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "criteria": [{"name": "Thesis", "max_points": 5}],
        }).get_json()
        client.post("/api/submissions", headers=auth(student),
                    json={"rubric_id": made["id"], "content": "First attempt."})

        set_due_date(made["id"], datetime.now(timezone.utc) - timedelta(minutes=1))
        assert resubmit(client, auth, student, made["id"], "Blocked while closed.").status_code == 422

        client.put(f"/api/rubrics/{made['id']}", headers=auth(teacher),
                   json={"due_date": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()})
        assert resubmit(client, auth, student, made["id"], "Allowed again.").status_code == 200

    def test_the_scoring_is_still_frozen(self, client, auth, teacher, student):
        """Narrowing the lock must not have opened the thing it existed for."""
        made = client.post("/api/rubrics", headers=auth(teacher), json={
            "title": "Still frozen", "type": "text",
            "criteria": [{"name": "Thesis", "max_points": 5}],
        }).get_json()
        client.post("/api/submissions", headers=auth(student),
                    json={"rubric_id": made["id"], "content": "Work exists now."})

        response = client.put(f"/api/rubrics/{made['id']}", headers=auth(teacher),
                              json={"criteria": [{"name": "Thesis", "max_points": 99}]})
        assert response.status_code == 409
