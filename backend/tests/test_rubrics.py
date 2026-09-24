"""Rubric CRUD, its validation rules, ownership, and the edit lock."""

import pytest


class TestCreate:
    def test_creates_a_rubric_with_criteria(self, rubric):
        assert rubric["total_points"] == 15.0
        assert [c["name"] for c in rubric["criteria"]] == ["Thesis", "Evidence"]
        assert [c["position"] for c in rubric["criteria"]] == [0, 1]
        assert all(c["id"] for c in rubric["criteria"])

    def test_creates_a_code_rubric(self, client, auth, teacher):
        """A code criterion needs test cases -- without them there is nothing to
        grade against. See test_code_grading.py for the rejection case."""
        response = client.post(
            "/api/rubrics",
            json={
                "title": "Sum two numbers",
                "type": "code",
                "criteria": [{
                    "name": "Adds correctly",
                    "max_points": 10,
                    "test_cases": [{"stdin": "2 3", "expected_output": "5"}],
                }],
            },
            headers=auth(teacher),
        )
        assert response.status_code == 201, response.get_json()
        assert response.get_json()["type"] == "code"
        assert len(response.get_json()["criteria"][0]["test_cases"]) == 1

    def test_a_student_cannot_create_one(self, client, auth, student):
        response = client.post(
            "/api/rubrics",
            json={"title": "R", "type": "text", "criteria": [{"name": "C", "max_points": 1}]},
            headers=auth(student),
        )
        assert response.status_code == 403

    @pytest.mark.parametrize(
        "body",
        [
            {"title": "R", "type": "text", "criteria": []},
            {"title": "R", "type": "text"},
            {"title": "", "type": "text", "criteria": [{"name": "C", "max_points": 1}]},
            {"title": "R", "type": "essay", "criteria": [{"name": "C", "max_points": 1}]},
            {"title": "R", "type": "text", "criteria": [{"name": "C", "max_points": 0}]},
            {"title": "R", "type": "text", "criteria": [{"name": "C", "max_points": -5}]},
            {"title": "R", "type": "text", "criteria": [{"name": "C", "max_points": "lots"}]},
            {"title": "R", "type": "text", "criteria": [{"name": "", "max_points": 1}]},
            {"title": "R", "type": "text", "criteria": [{"name": "A", "max_points": 1}, {"name": "A", "max_points": 2}]},
            {"title": "R", "type": "text", "criteria": "not-a-list"},
            {"title": "R", "type": "text", "criteria": ["not-an-object"]},
        ],
        ids=[
            "empty-criteria", "missing-criteria", "empty-title", "bad-type", "zero-points",
            "negative-points", "non-numeric-points", "empty-criterion-name", "duplicate-names",
            "criteria-not-a-list", "criterion-not-an-object",
        ],
    )
    def test_rejects_invalid_input(self, client, auth, teacher, body):
        assert client.post("/api/rubrics", json=body, headers=auth(teacher)).status_code == 422

    def test_duplicate_names_are_caught_regardless_of_case(self, client, auth, teacher):
        response = client.post(
            "/api/rubrics",
            json={"title": "R", "type": "text",
                  "criteria": [{"name": "Thesis", "max_points": 1}, {"name": "THESIS", "max_points": 2}]},
            headers=auth(teacher),
        )
        assert response.status_code == 422


class TestRead:
    def test_a_student_can_list_rubrics(self, client, auth, student, rubric):
        response = client.get("/api/rubrics", headers=auth(student))
        assert response.status_code == 200
        assert [r["id"] for r in response.get_json()] == [rubric["id"]]

    def test_a_teacher_sees_only_their_own(self, client, auth, teacher, other_teacher, rubric):
        client.post(
            "/api/rubrics",
            json={"title": "Someone else's", "type": "text", "criteria": [{"name": "C", "max_points": 1}]},
            headers=auth(other_teacher),
        )
        mine = client.get("/api/rubrics", headers=auth(teacher)).get_json()
        assert [r["id"] for r in mine] == [rubric["id"]]

    def test_a_student_still_sees_every_rubric(self, client, auth, student, other_teacher, rubric):
        client.post(
            "/api/rubrics",
            json={"title": "Another", "type": "text", "criteria": [{"name": "C", "max_points": 1}]},
            headers=auth(other_teacher),
        )
        assert len(client.get("/api/rubrics", headers=auth(student)).get_json()) == 2


class TestOwnerOnlyFields:
    """How many classmates have submitted is the author's business."""

    def test_a_student_listing_does_not_see_the_counts(self, client, auth, student, rubric):
        listed = client.get("/api/rubrics", headers=auth(student)).get_json()[0]
        assert "submission_count" not in listed
        assert "locked" not in listed

    def test_a_student_reading_one_does_not_see_the_counts(self, client, auth, student, rubric):
        body = client.get(f"/api/rubrics/{rubric['id']}", headers=auth(student)).get_json()
        assert "submission_count" not in body
        assert "locked" not in body

    def test_a_student_still_gets_what_they_need_to_submit(self, client, auth, student, rubric):
        body = client.get(f"/api/rubrics/{rubric['id']}", headers=auth(student)).get_json()
        assert body["title"] == "Essay 1"
        assert body["total_points"] == 15.0
        assert [c["name"] for c in body["criteria"]] == ["Thesis", "Evidence"]

    def test_another_teacher_does_not_see_them_either(self, client, auth, other_teacher, rubric):
        body = client.get(f"/api/rubrics/{rubric['id']}", headers=auth(other_teacher)).get_json()
        assert "submission_count" not in body

    def test_the_owner_does_see_them(self, client, auth, teacher, rubric):
        body = client.get(f"/api/rubrics/{rubric['id']}", headers=auth(teacher)).get_json()
        assert body["submission_count"] == 0
        assert body["locked"] is False


class TestLockedFlag:
    def test_an_untouched_rubric_is_unlocked(self, client, auth, teacher, rubric):
        listed = client.get("/api/rubrics", headers=auth(teacher)).get_json()[0]
        assert listed["locked"] is False
        assert listed["submission_count"] == 0

    def test_a_submission_locks_it(self, client, auth, teacher, student, submit, rubric):
        submit(student)
        listed = client.get("/api/rubrics", headers=auth(teacher)).get_json()[0]
        assert listed["locked"] is True
        assert listed["submission_count"] == 1

    def test_the_count_rises_with_each_submission(self, client, auth, teacher, student, other_student, submit):
        submit(student)
        submit(other_student)
        assert client.get("/api/rubrics", headers=auth(teacher)).get_json()[0]["submission_count"] == 2

    def test_the_single_rubric_view_reports_it_too(self, client, auth, teacher, student, submit, rubric):
        submit(student)
        body = client.get(f"/api/rubrics/{rubric['id']}", headers=auth(teacher)).get_json()
        assert body["locked"] is True
        assert body["submission_count"] == 1

    def test_locked_matches_what_editing_actually_does(self, client, auth, teacher, student, submit, rubric):
        """The flag has to agree with the 409, or the UI disables the wrong thing."""
        assert client.get(f"/api/rubrics/{rubric['id']}", headers=auth(teacher)).get_json()["locked"] is False
        assert client.put(f"/api/rubrics/{rubric['id']}", json={"title": "ok"}, headers=auth(teacher)).status_code == 200

        submit(student)
        assert client.get(f"/api/rubrics/{rubric['id']}", headers=auth(teacher)).get_json()["locked"] is True
        assert client.put(f"/api/rubrics/{rubric['id']}", json={"title": "no"}, headers=auth(teacher)).status_code == 409
        assert client.delete(f"/api/rubrics/{rubric['id']}", headers=auth(teacher)).status_code == 409

    def test_a_student_can_read_one(self, client, auth, student, rubric):
        assert client.get(f"/api/rubrics/{rubric['id']}", headers=auth(student)).status_code == 200

    def test_missing_rubric_is_404(self, client, auth, student):
        assert client.get("/api/rubrics/99999", headers=auth(student)).status_code == 404

    def test_listing_requires_a_login(self, client):
        assert client.get("/api/rubrics").status_code == 401


class TestUpdate:
    def test_the_owner_can_edit(self, client, auth, teacher, rubric):
        response = client.put(f"/api/rubrics/{rubric['id']}", json={"title": "Essay 1 (revised)"},
                              headers=auth(teacher))
        assert response.status_code == 200
        assert response.get_json()["title"] == "Essay 1 (revised)"

    def test_replacing_criteria_recomputes_the_total(self, client, auth, teacher, rubric):
        response = client.put(
            f"/api/rubrics/{rubric['id']}",
            json={"criteria": [{"name": "Only", "max_points": 7}]},
            headers=auth(teacher),
        )
        assert response.status_code == 200
        assert response.get_json()["total_points"] == 7.0

    def test_can_edit_while_keeping_a_criterion_name(self, client, auth, teacher, rubric):
        """The ordinary edit: change something else and resend the same criteria.
        Assigning over the list inserts before deleting, so the unique
        (rubric_id, name) constraint used to reject this with a 400."""
        response = client.put(
            f"/api/rubrics/{rubric['id']}",
            json={
                "title": "Essay 1 (revised)",
                "criteria": [
                    {"name": "Thesis", "max_points": 5},      # unchanged name
                    {"name": "Evidence", "max_points": 12},   # unchanged name, new points
                ],
            },
            headers=auth(teacher),
        )
        assert response.status_code == 200, response.get_json()
        assert response.get_json()["total_points"] == 17.0
        assert [c["name"] for c in response.get_json()["criteria"]] == ["Thesis", "Evidence"]

    def test_resending_identical_criteria_is_accepted(self, client, auth, teacher, rubric):
        response = client.put(
            f"/api/rubrics/{rubric['id']}",
            json={"criteria": [{"name": c["name"], "max_points": c["max_points"]} for c in rubric["criteria"]]},
            headers=auth(teacher),
        )
        assert response.status_code == 200, response.get_json()
        assert response.get_json()["total_points"] == 15.0

    def test_editing_leaves_no_orphaned_criteria(self, client, auth, teacher, rubric):
        from app.extensions import db
        from app.models import RubricCriterion

        client.put(
            f"/api/rubrics/{rubric['id']}",
            json={"criteria": [{"name": "Thesis", "max_points": 5}]},
            headers=auth(teacher),
        )
        remaining = db.session.query(RubricCriterion).filter_by(rubric_id=rubric["id"]).all()
        assert [c.name for c in remaining] == ["Thesis"]

    def test_another_teacher_cannot_edit(self, client, auth, other_teacher, rubric):
        response = client.put(f"/api/rubrics/{rubric['id']}", json={"title": "Mine now"},
                              headers=auth(other_teacher))
        assert response.status_code == 403

    def test_a_student_cannot_edit(self, client, auth, student, rubric):
        assert client.put(f"/api/rubrics/{rubric['id']}", json={"title": "x"},
                          headers=auth(student)).status_code == 403

    def test_missing_rubric_is_404(self, client, auth, teacher):
        assert client.put("/api/rubrics/99999", json={"title": "x"}, headers=auth(teacher)).status_code == 404

    def test_invalid_criteria_are_still_rejected(self, client, auth, teacher, rubric):
        response = client.put(f"/api/rubrics/{rubric['id']}", json={"criteria": []}, headers=auth(teacher))
        assert response.status_code == 422


class TestLockingOnceSubmitted:
    """Changing points under a submission would invalidate grades already
    recorded against it, so both edit and delete are refused."""

    def test_edit_is_refused(self, client, auth, teacher, student, rubric, submit):
        submit(student)
        response = client.put(f"/api/rubrics/{rubric['id']}", json={"title": "too late"}, headers=auth(teacher))
        assert response.status_code == 409

    def test_delete_is_refused(self, client, auth, teacher, student, rubric, submit):
        submit(student)
        assert client.delete(f"/api/rubrics/{rubric['id']}", headers=auth(teacher)).status_code == 409


class TestDelete:
    def test_the_owner_can_delete_an_unused_rubric(self, client, auth, teacher, rubric):
        assert client.delete(f"/api/rubrics/{rubric['id']}", headers=auth(teacher)).status_code == 204
        assert client.get(f"/api/rubrics/{rubric['id']}", headers=auth(teacher)).status_code == 404

    def test_deleting_takes_the_criteria_with_it(self, client, auth, teacher, rubric):
        from app.extensions import db
        from app.models import RubricCriterion

        criterion_id = rubric["criteria"][0]["id"]
        client.delete(f"/api/rubrics/{rubric['id']}", headers=auth(teacher))
        assert db.session.get(RubricCriterion, criterion_id) is None

    def test_another_teacher_cannot_delete(self, client, auth, other_teacher, rubric):
        assert client.delete(f"/api/rubrics/{rubric['id']}", headers=auth(other_teacher)).status_code == 403

    def test_a_student_cannot_delete(self, client, auth, student, rubric):
        assert client.delete(f"/api/rubrics/{rubric['id']}", headers=auth(student)).status_code == 403

    def test_missing_rubric_is_404(self, client, auth, teacher):
        assert client.delete("/api/rubrics/99999", headers=auth(teacher)).status_code == 404
