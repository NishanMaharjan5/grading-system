"""Revising a grade after it has been released.

This is the fix for what the README called the "approval is terminal" gap: a
teacher who released a wrong score had no way to correct it. Allowing a
correction is only safe if every one is recorded, so each change writes a
GradeRevision row saying what the score was, what it became, who changed it
and when.

Validation is deliberately the same as the original override: every criterion
scored, each within its own maximum.
"""

import pytest

from app.extensions import db
from app.models import Grade, GradeRevision, Submission


@pytest.fixture
def approved(client, auth, teacher, student, submit, criterion_ids, seed_ai_grade):
    """An approved submission: Thesis 3/5, Evidence 7/10."""
    submission_id = submit(student)
    thesis, evidence = criterion_ids
    seed_ai_grade(submission_id, {thesis: 3, evidence: 7})
    response = client.put(f"/api/submissions/{submission_id}/review", json={}, headers=auth(teacher))
    assert response.status_code == 200, response.get_json()
    return submission_id


def revise(client, auth, token, submission_id, scores, **extra):
    return client.put(f"/api/submissions/{submission_id}/revise", headers=auth(token),
                      json={"criterion_scores": scores, **extra})


def scores_for(criterion_ids, thesis_score, evidence_score, **feedback):
    thesis, evidence = criterion_ids
    return [
        {"criterion_id": thesis, "final_score": thesis_score, **feedback.get("thesis", {})},
        {"criterion_id": evidence, "final_score": evidence_score, **feedback.get("evidence", {})},
    ]


class TestRevisingWorks:
    def test_an_approved_grade_can_be_corrected(self, client, auth, teacher, approved, criterion_ids):
        response = revise(client, auth, teacher, approved, scores_for(criterion_ids, 5, 9))
        assert response.status_code == 200
        assert response.get_json()["final_total"] == 14

    def test_the_live_grade_actually_changes(self, client, auth, teacher, approved, criterion_ids):
        thesis, _ = criterion_ids
        revise(client, auth, teacher, approved, scores_for(criterion_ids, 5, 9))
        grade = db.session.query(Grade).filter_by(submission_id=approved, criterion_id=thesis).one()
        assert float(grade.final_score) == 5

    def test_it_stays_approved(self, client, auth, teacher, approved, criterion_ids):
        revise(client, auth, teacher, approved, scores_for(criterion_ids, 5, 9))
        assert db.session.get(Submission, approved).status == "approved"

    def test_the_student_sees_the_corrected_grade(self, client, auth, student, teacher, approved, criterion_ids):
        revise(client, auth, teacher, approved, scores_for(criterion_ids, 5, 9))
        body = client.get(f"/api/submissions/{approved}", headers=auth(student)).get_json()
        assert body["final_total"] == 14

    def test_feedback_is_regenerated_when_the_teacher_gives_none(
            self, client, auth, teacher, approved, criterion_ids):
        """Keeping the old sentence would describe a score no longer on the page."""
        thesis, _ = criterion_ids
        before = db.session.query(Grade).filter_by(submission_id=approved, criterion_id=thesis).one().final_feedback
        revise(client, auth, teacher, approved, scores_for(criterion_ids, 5, 9))
        after = db.session.query(Grade).filter_by(submission_id=approved, criterion_id=thesis).one().final_feedback
        assert after != before

    def test_the_teachers_own_wording_is_kept(self, client, auth, teacher, approved, criterion_ids):
        thesis, evidence = criterion_ids
        revise(client, auth, teacher, approved, [
            {"criterion_id": thesis, "final_score": 5, "final_feedback": "Much better, well argued."},
            {"criterion_id": evidence, "final_score": 9},
        ])
        grade = db.session.query(Grade).filter_by(submission_id=approved, criterion_id=thesis).one()
        assert grade.final_feedback == "Much better, well argued."


class TestTheAuditTrail:
    def test_a_revision_row_is_written_per_changed_criterion(
            self, client, auth, teacher, approved, criterion_ids):
        revise(client, auth, teacher, approved, scores_for(criterion_ids, 5, 9))
        rows = db.session.query(GradeRevision).filter_by(submission_id=approved).all()
        assert len(rows) == 2

    def test_it_records_what_the_score_was_and_became(
            self, client, auth, teacher, approved, criterion_ids):
        thesis, _ = criterion_ids
        revise(client, auth, teacher, approved, scores_for(criterion_ids, 5, 9))
        row = db.session.query(GradeRevision).filter_by(
            submission_id=approved, criterion_id=thesis).one()
        assert float(row.old_final_score) == 3 and float(row.new_final_score) == 5

    def test_it_records_who_and_when(self, client, auth, teacher, approved, criterion_ids):
        revise(client, auth, teacher, approved, scores_for(criterion_ids, 5, 9))
        row = db.session.query(GradeRevision).filter_by(submission_id=approved).first()
        assert row.revised_by is not None and row.revised_at is not None

    def test_it_records_the_old_feedback_too(self, client, auth, teacher, approved, criterion_ids):
        thesis, _ = criterion_ids
        original = db.session.query(Grade).filter_by(
            submission_id=approved, criterion_id=thesis).one().final_feedback
        revise(client, auth, teacher, approved, scores_for(criterion_ids, 5, 9))
        row = db.session.query(GradeRevision).filter_by(
            submission_id=approved, criterion_id=thesis).one()
        assert row.old_final_feedback == original

    def test_revising_twice_appends_rather_than_replacing(
            self, client, auth, teacher, approved, criterion_ids):
        """Re-revising is allowed, and the first correction is not erased."""
        revise(client, auth, teacher, approved, scores_for(criterion_ids, 5, 9))
        second = revise(client, auth, teacher, approved, scores_for(criterion_ids, 2, 4))
        assert second.status_code == 200
        rows = db.session.query(GradeRevision).filter_by(submission_id=approved).all()
        assert len(rows) == 4

    def test_the_second_revision_starts_from_the_first_result(
            self, client, auth, teacher, approved, criterion_ids):
        thesis, _ = criterion_ids
        revise(client, auth, teacher, approved, scores_for(criterion_ids, 5, 9))
        revise(client, auth, teacher, approved, scores_for(criterion_ids, 2, 4))
        rows = (db.session.query(GradeRevision)
                .filter_by(submission_id=approved, criterion_id=thesis)
                .order_by(GradeRevision.id).all())
        assert float(rows[0].new_final_score) == 5
        assert float(rows[1].old_final_score) == 5, "the second revision's 'before' is the first's 'after'"

    def test_an_unchanged_criterion_writes_no_row(self, client, auth, teacher, approved, criterion_ids):
        """Only what actually moved is recorded, so the history stays readable."""
        response = revise(client, auth, teacher, approved, scores_for(criterion_ids, 3, 9))
        assert response.get_json()["revised_criteria"] == 1
        rows = db.session.query(GradeRevision).filter_by(submission_id=approved).all()
        assert len(rows) == 1

    def test_the_history_is_returned_to_the_teacher(self, client, auth, teacher, approved, criterion_ids):
        revise(client, auth, teacher, approved, scores_for(criterion_ids, 5, 9))
        body = client.get(f"/api/submissions/{approved}", headers=auth(teacher)).get_json()
        assert len(body["revisions"]) == 2
        assert body["revisions"][0]["criterion_name"] in ("Thesis", "Evidence")
        assert body["revisions"][0]["revised_by_name"] is not None

    def test_a_student_is_not_shown_the_revision_history(
            self, client, auth, student, teacher, approved, criterion_ids):
        """They see the grade that stands, not a log of the teacher's corrections."""
        revise(client, auth, teacher, approved, scores_for(criterion_ids, 5, 9))
        body = client.get(f"/api/submissions/{approved}", headers=auth(student)).get_json()
        assert "revisions" not in body


class TestAccessControl:
    def test_a_student_cannot_revise(self, client, auth, student, approved, criterion_ids):
        assert revise(client, auth, student, approved, scores_for(criterion_ids, 5, 9)).status_code == 403

    def test_another_teacher_cannot_revise(self, client, auth, other_teacher, approved, criterion_ids):
        response = revise(client, auth, other_teacher, approved, scores_for(criterion_ids, 5, 9))
        assert response.status_code == 403
        assert db.session.query(GradeRevision).count() == 0

    def test_an_unapproved_submission_cannot_be_revised(
            self, client, auth, teacher, student, submit, criterion_ids):
        submission_id = submit(student)
        response = revise(client, auth, teacher, submission_id, scores_for(criterion_ids, 5, 9))
        assert response.status_code == 409
        assert "review this submission instead" in response.get_json()["detail"]

    def test_an_unknown_submission_is_404(self, client, auth, teacher, criterion_ids):
        assert revise(client, auth, teacher, 999999, scores_for(criterion_ids, 5, 9)).status_code == 404


class TestValidationMatchesTheOriginalReview:
    def test_a_score_above_the_maximum_is_refused(self, client, auth, teacher, approved, criterion_ids):
        response = revise(client, auth, teacher, approved, scores_for(criterion_ids, 99, 9))
        assert response.status_code == 422
        assert "outside" in response.get_json()["detail"]

    def test_a_negative_score_is_refused(self, client, auth, teacher, approved, criterion_ids):
        assert revise(client, auth, teacher, approved, scores_for(criterion_ids, -1, 9)).status_code == 422

    def test_every_criterion_must_be_scored(self, client, auth, teacher, approved, criterion_ids):
        thesis, _ = criterion_ids
        response = revise(client, auth, teacher, approved, [{"criterion_id": thesis, "final_score": 5}])
        assert response.status_code == 422

    def test_a_criterion_from_another_rubric_is_refused(self, client, auth, teacher, approved):
        response = revise(client, auth, teacher, approved, [{"criterion_id": 999999, "final_score": 5}])
        assert response.status_code == 422

    def test_a_failed_revision_changes_nothing(self, client, auth, teacher, approved, criterion_ids):
        thesis, _ = criterion_ids
        before = db.session.query(Grade).filter_by(submission_id=approved, criterion_id=thesis).one().final_score
        revise(client, auth, teacher, approved, scores_for(criterion_ids, 99, 9))
        after = db.session.query(Grade).filter_by(submission_id=approved, criterion_id=thesis).one().final_score
        assert after == before
        assert db.session.query(GradeRevision).count() == 0
