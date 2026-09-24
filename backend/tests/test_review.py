"""The teacher review queue and the approve/override flow."""

import pytest


@pytest.fixture
def graded(submit, student, seed_ai_grade, criterion_ids):
    """A submission in ai_graded with Thesis 4/5 and Evidence 2/10."""
    thesis, evidence = criterion_ids
    return seed_ai_grade(submit(student), {thesis: 4, evidence: 2})


@pytest.fixture
def failed(submit, other_student, seed_grading_failed):
    """A submission in grading_failed: no Grade rows at all."""
    return seed_grading_failed(submit(other_student))


class TestPendingQueue:
    def test_lists_work_awaiting_the_owning_teacher(self, client, auth, teacher, graded, failed):
        response = client.get("/api/submissions/pending", headers=auth(teacher))
        assert response.status_code == 200
        assert sorted(s["id"] for s in response.get_json()) == sorted([graded, failed])

    def test_includes_the_ai_numbers(self, client, auth, teacher, graded):
        entry = next(s for s in client.get("/api/submissions/pending", headers=auth(teacher)).get_json()
                     if s["id"] == graded)
        assert entry["ai_total"] == 6.0
        assert sorted(g["ai_score"] for g in entry["grades"]) == [2.0, 4.0]

    def test_another_teacher_sees_an_empty_queue(self, client, auth, other_teacher, graded, failed):
        assert client.get("/api/submissions/pending", headers=auth(other_teacher)).get_json() == []

    def test_a_student_is_refused(self, client, auth, student, graded):
        assert client.get("/api/submissions/pending", headers=auth(student)).status_code == 403

    def test_drains_once_everything_is_approved(self, client, auth, teacher, criterion_ids, graded, failed):
        thesis, evidence = criterion_ids
        client.put(f"/api/submissions/{graded}/review", json={}, headers=auth(teacher))
        client.put(
            f"/api/submissions/{failed}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": 2},
                                       {"criterion_id": evidence, "final_score": 6}]},
            headers=auth(teacher),
        )
        assert client.get("/api/submissions/pending", headers=auth(teacher)).get_json() == []

    def test_a_merely_submitted_item_is_not_in_the_queue(self, client, auth, teacher, student, submit):
        submit(student)  # left in whatever state grading produced, not forced to ai_graded
        queued = client.get("/api/submissions/pending", headers=auth(teacher)).get_json()
        assert all(s["status"] in ("ai_graded", "grading_failed") for s in queued)


class TestAccessControl:
    def test_a_student_cannot_review(self, client, auth, student, graded):
        assert client.put(f"/api/submissions/{graded}/review", json={}, headers=auth(student)).status_code == 403

    def test_another_teacher_cannot_review(self, client, auth, other_teacher, graded):
        assert client.put(f"/api/submissions/{graded}/review", json={},
                          headers=auth(other_teacher)).status_code == 403

    def test_missing_submission_is_404(self, client, auth, teacher):
        assert client.put("/api/submissions/99999/review", json={}, headers=auth(teacher)).status_code == 404


class TestApproveAsIs:
    def test_copies_the_ai_scores_into_the_final_grade(self, client, auth, teacher, graded):
        body = client.put(f"/api/submissions/{graded}/review", json={}, headers=auth(teacher)).get_json()
        assert body["status"] == "approved"
        assert sorted(g["final_score"] for g in body["grades"]) == [2.0, 4.0]
        assert body["final_total"] == 6.0

    def test_marks_every_criterion_as_accepted(self, client, auth, teacher, graded):
        body = client.put(f"/api/submissions/{graded}/review", json={}, headers=auth(teacher)).get_json()
        assert [g["ai_accepted"] for g in body["grades"]] == [True, True]

    def test_records_who_approved_it(self, client, auth, teacher, graded):
        from app.extensions import db
        from app.models import Grade, Submission

        client.put(f"/api/submissions/{graded}/review", json={}, headers=auth(teacher))
        submission = db.session.get(Submission, graded)
        approver = db.session.query(Grade).filter_by(submission_id=graded).first()
        assert submission.status == "approved"
        assert approver.approved_by is not None and approver.approved_at is not None

    def test_is_refused_a_second_time(self, client, auth, teacher, graded):
        client.put(f"/api/submissions/{graded}/review", json={}, headers=auth(teacher))
        assert client.put(f"/api/submissions/{graded}/review", json={}, headers=auth(teacher)).status_code == 409

    def test_is_refused_when_there_is_no_ai_score(self, client, auth, teacher, failed):
        response = client.put(f"/api/submissions/{failed}/review", json={}, headers=auth(teacher))
        assert response.status_code == 422
        assert "criterion_scores" in response.get_json()["detail"]


class TestOverrideValidation:
    def _payload(self, criterion_ids, **overrides):
        thesis, evidence = criterion_ids
        scores = [{"criterion_id": thesis, "final_score": 3}, {"criterion_id": evidence, "final_score": 5}]
        scores = overrides.get("criterion_scores", scores)
        return {"criterion_scores": scores}

    def test_rejects_a_score_above_the_maximum(self, client, auth, teacher, graded, criterion_ids):
        thesis, evidence = criterion_ids
        response = client.put(
            f"/api/submissions/{graded}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": 99},
                                       {"criterion_id": evidence, "final_score": 5}]},
            headers=auth(teacher),
        )
        assert response.status_code == 422
        assert "Thesis" in response.get_json()["detail"]

    def test_rejects_a_negative_score(self, client, auth, teacher, graded, criterion_ids):
        thesis, evidence = criterion_ids
        response = client.put(
            f"/api/submissions/{graded}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": -1},
                                       {"criterion_id": evidence, "final_score": 5}]},
            headers=auth(teacher),
        )
        assert response.status_code == 422

    def test_rejects_a_missing_criterion(self, client, auth, teacher, graded, criterion_ids):
        thesis, _ = criterion_ids
        response = client.put(
            f"/api/submissions/{graded}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": 3}]},
            headers=auth(teacher),
        )
        assert response.status_code == 422
        assert "Evidence" in response.get_json()["detail"]

    def test_rejects_a_criterion_from_another_rubric(self, client, auth, teacher, graded, criterion_ids):
        _, evidence = criterion_ids
        response = client.put(
            f"/api/submissions/{graded}/review",
            json={"criterion_scores": [{"criterion_id": 99999, "final_score": 3},
                                       {"criterion_id": evidence, "final_score": 5}]},
            headers=auth(teacher),
        )
        assert response.status_code == 422

    def test_rejects_a_non_numeric_score(self, client, auth, teacher, graded, criterion_ids):
        thesis, evidence = criterion_ids
        response = client.put(
            f"/api/submissions/{graded}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": "abc"},
                                       {"criterion_id": evidence, "final_score": 5}]},
            headers=auth(teacher),
        )
        assert response.status_code == 422

    def test_rejects_a_repeated_criterion(self, client, auth, teacher, graded, criterion_ids):
        thesis, _ = criterion_ids
        response = client.put(
            f"/api/submissions/{graded}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": 3},
                                       {"criterion_id": thesis, "final_score": 4}]},
            headers=auth(teacher),
        )
        assert response.status_code == 422

    @pytest.mark.parametrize("scores", [[], "not-a-list", [123]], ids=["empty", "not-a-list", "not-an-object"])
    def test_rejects_a_malformed_payload(self, client, auth, teacher, graded, scores):
        response = client.put(
            f"/api/submissions/{graded}/review", json={"criterion_scores": scores}, headers=auth(teacher)
        )
        assert response.status_code == 422

    def test_rejects_a_non_string_summary(self, client, auth, teacher, graded, criterion_ids):
        thesis, evidence = criterion_ids
        response = client.put(
            f"/api/submissions/{graded}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": 3},
                                       {"criterion_id": evidence, "final_score": 5}],
                  "summary": 42},
            headers=auth(teacher),
        )
        assert response.status_code == 422

    def test_a_rejected_override_leaves_it_unapproved(self, client, auth, teacher, graded, criterion_ids):
        thesis, evidence = criterion_ids
        client.put(
            f"/api/submissions/{graded}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": 99},
                                       {"criterion_id": evidence, "final_score": 5}]},
            headers=auth(teacher),
        )
        assert client.get(f"/api/submissions/{graded}", headers=auth(teacher)).get_json()["status"] == "ai_graded"


class TestOverride:
    def test_stores_the_teachers_numbers(self, client, auth, teacher, graded, criterion_ids):
        thesis, evidence = criterion_ids
        body = client.put(
            f"/api/submissions/{graded}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": 4},
                                       {"criterion_id": evidence, "final_score": 9}]},
            headers=auth(teacher),
        ).get_json()
        assert body["status"] == "approved"
        assert body["final_total"] == 13.0

    def test_records_acceptance_per_criterion(self, client, auth, teacher, graded, criterion_ids):
        """4 matches the AI's 4 -> accepted; 9 replaces the AI's 2 -> overridden."""
        thesis, evidence = criterion_ids
        body = client.put(
            f"/api/submissions/{graded}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": 4},
                                       {"criterion_id": evidence, "final_score": 9}]},
            headers=auth(teacher),
        ).get_json()
        by_criterion = {g["criterion_id"]: g for g in body["grades"]}
        assert by_criterion[thesis]["ai_accepted"] is True
        assert by_criterion[evidence]["ai_accepted"] is False

    def test_never_overwrites_what_the_ai_suggested(self, client, auth, teacher, graded, criterion_ids):
        thesis, evidence = criterion_ids
        body = client.put(
            f"/api/submissions/{graded}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": 4},
                                       {"criterion_id": evidence, "final_score": 9}]},
            headers=auth(teacher),
        ).get_json()
        by_criterion = {g["criterion_id"]: g for g in body["grades"]}
        assert by_criterion[evidence]["ai_score"] == 2.0  # still the original suggestion
        assert by_criterion[evidence]["final_score"] == 9.0

    def test_keeps_the_teachers_own_wording(self, client, auth, teacher, graded, criterion_ids):
        thesis, evidence = criterion_ids
        body = client.put(
            f"/api/submissions/{graded}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": 4, "final_feedback": "My words."},
                                       {"criterion_id": evidence, "final_score": 9}],
                  "summary": "My overall note."},
            headers=auth(teacher),
        ).get_json()
        by_criterion = {g["criterion_id"]: g for g in body["grades"]}
        assert by_criterion[thesis]["final_feedback"] == "My words."
        assert body["final_summary"] == "My overall note."


class TestGradingFailedPath:
    def test_can_be_graded_by_hand(self, client, auth, teacher, failed, criterion_ids):
        thesis, evidence = criterion_ids
        body = client.put(
            f"/api/submissions/{failed}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": 2},
                                       {"criterion_id": evidence, "final_score": 6}]},
            headers=auth(teacher),
        ).get_json()
        assert body["status"] == "approved"
        assert body["final_total"] == 8.0

    def test_creates_the_grade_rows_that_never_existed(self, client, auth, teacher, failed, criterion_ids):
        thesis, evidence = criterion_ids
        before = client.get(f"/api/submissions/{failed}", headers=auth(teacher)).get_json()
        assert before["grades"] == []

        body = client.put(
            f"/api/submissions/{failed}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": 2},
                                       {"criterion_id": evidence, "final_score": 6}]},
            headers=auth(teacher),
        ).get_json()
        assert len(body["grades"]) == 2

    def test_acceptance_is_unknown_when_there_was_no_suggestion(self, client, auth, teacher, failed, criterion_ids):
        thesis, evidence = criterion_ids
        body = client.put(
            f"/api/submissions/{failed}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": 2},
                                       {"criterion_id": evidence, "final_score": 6}]},
            headers=auth(teacher),
        ).get_json()
        assert [g["ai_accepted"] for g in body["grades"]] == [None, None]
        assert body["ai_total"] is None


class TestFeedbackOnApproval:
    """Feedback must never contradict the number printed beside it."""

    def test_approving_as_is_copies_the_ai_wording(self, client, auth, teacher, graded):
        body = client.put(f"/api/submissions/{graded}/review", json={}, headers=auth(teacher)).get_json()
        for grade in body["grades"]:
            assert grade["final_feedback"] == grade["ai_feedback"]
        assert body["final_summary"] == body["ai_summary"]

    def test_an_override_regenerates_wording_for_the_new_score(self, client, auth, teacher, graded, criterion_ids):
        """Evidence goes 2 -> 9, so copying the AI's "needs significant
        improvement" would sit next to a 9/10."""
        thesis, evidence = criterion_ids
        body = client.put(
            f"/api/submissions/{graded}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": 4},
                                       {"criterion_id": evidence, "final_score": 9}]},
            headers=auth(teacher),
        ).get_json()

        rewritten = next(g for g in body["grades"] if g["criterion_id"] == evidence)
        assert "9/10" in rewritten["final_feedback"]
        assert "excellent" in rewritten["final_feedback"]
        assert rewritten["final_feedback"] != rewritten["ai_feedback"]

    def test_an_unchanged_score_reproduces_the_same_wording(self, client, auth, teacher, graded, criterion_ids):
        thesis, evidence = criterion_ids
        body = client.put(
            f"/api/submissions/{graded}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": 4},
                                       {"criterion_id": evidence, "final_score": 9}]},
            headers=auth(teacher),
        ).get_json()

        untouched = next(g for g in body["grades"] if g["criterion_id"] == thesis)
        assert untouched["final_feedback"] == untouched["ai_feedback"]

    def test_the_summary_is_regenerated_from_the_final_scores(self, client, auth, teacher, graded, criterion_ids):
        thesis, evidence = criterion_ids
        body = client.put(
            f"/api/submissions/{graded}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": 4},
                                       {"criterion_id": evidence, "final_score": 9}]},
            headers=auth(teacher),
        ).get_json()
        assert "Overall 13/15" in body["final_summary"]
        assert body["final_summary"] != body["ai_summary"]

    def test_a_hand_graded_submission_still_gets_wording(self, client, auth, teacher, failed, criterion_ids):
        """grading_failed has no AI text to fall back on, so the same
        score-derived template runs rather than leaving the student a bare number."""
        thesis, evidence = criterion_ids
        body = client.put(
            f"/api/submissions/{failed}/review",
            json={"criterion_scores": [{"criterion_id": thesis, "final_score": 2},
                                       {"criterion_id": evidence, "final_score": 6}]},
            headers=auth(teacher),
        ).get_json()

        assert all(grade["ai_feedback"] is None for grade in body["grades"])
        assert all(grade["final_feedback"] for grade in body["grades"])
        assert "6/10" in next(g for g in body["grades"] if g["criterion_id"] == evidence)["final_feedback"]

    def test_feedback_is_never_left_empty(self, client, auth, teacher, graded):
        body = client.put(f"/api/submissions/{graded}/review", json={}, headers=auth(teacher)).get_json()
        assert all(grade["final_feedback"] for grade in body["grades"])
        assert body["final_summary"]


class TestStudentVisibilityAfterApproval:
    def test_the_student_sees_the_final_grade(self, client, auth, teacher, student, graded):
        client.put(f"/api/submissions/{graded}/review", json={}, headers=auth(teacher))
        body = client.get(f"/api/submissions/{graded}", headers=auth(student)).get_json()
        assert body["status"] == "approved"
        assert body["final_total"] == 6.0
        assert sorted(g["final_score"] for g in body["grades"]) == [2.0, 4.0]

    def test_the_student_still_never_sees_the_ai_working(self, client, auth, teacher, student, graded):
        client.put(f"/api/submissions/{graded}/review", json={}, headers=auth(teacher))
        body = client.get(f"/api/submissions/{graded}", headers=auth(student)).get_json()
        assert "ai_total" not in body
        assert "ai_summary" not in body
        assert not any(key.startswith("ai_") for grade in body["grades"] for key in grade)
