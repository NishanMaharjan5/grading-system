"""Rubric CRUD, its validation rules, ownership, and the edit lock."""

import pytest


class TestCreate:
    def test_creates_a_rubric_with_criteria(self, rubric):
        assert rubric["total_points"] == 15.0
        assert [c["name"] for c in rubric["criteria"]] == ["Thesis", "Evidence"]
        assert [c["position"] for c in rubric["criteria"]] == [0, 1]
        assert all(c["id"] for c in rubric["criteria"])

    def test_creates_a_code_rubric(self, client, auth, teacher):
        response = client.post(
            "/api/rubrics",
            json={"title": "Sum two numbers", "type": "code", "criteria": [{"name": "Tests pass", "max_points": 10}]},
            headers=auth(teacher),
        )
        assert response.status_code == 201
        assert response.get_json()["type"] == "code"

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
