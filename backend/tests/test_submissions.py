"""Submission creation and who is allowed to see what."""

import pytest


class TestCreate:
    def test_a_student_can_submit(self, client, auth, student, rubric):
        response = client.post(
            "/api/submissions", json={"rubric_id": rubric["id"], "content": "My essay."}, headers=auth(student)
        )
        assert response.status_code == 201
        assert response.get_json()["rubric_id"] == rubric["id"]

    def test_a_teacher_cannot_submit(self, client, auth, teacher, rubric):
        response = client.post(
            "/api/submissions", json={"rubric_id": rubric["id"], "content": "x"}, headers=auth(teacher)
        )
        assert response.status_code == 403

    def test_only_one_submission_per_student_per_rubric(self, client, auth, student, rubric, submit):
        submit(student)
        response = client.post(
            "/api/submissions", json={"rubric_id": rubric["id"], "content": "again"}, headers=auth(student)
        )
        assert response.status_code == 409

    def test_two_students_can_each_submit(self, submit, student, other_student):
        assert submit(student) != submit(other_student)

    def test_unknown_rubric_is_404(self, client, auth, student):
        response = client.post(
            "/api/submissions", json={"rubric_id": 99999, "content": "x"}, headers=auth(student)
        )
        assert response.status_code == 404

    @pytest.mark.parametrize(
        "body",
        [
            {"rubric_id": None, "content": "x"},
            {"content": "x"},
            {"rubric_id": "1", "content": "x"},
            {"rubric_id": 1, "content": ""},
            {"rubric_id": 1, "content": "   "},
            {"rubric_id": 1},
        ],
        ids=["null-rubric", "missing-rubric", "string-rubric", "empty-content", "blank-content", "missing-content"],
    )
    def test_rejects_invalid_input(self, client, auth, student, rubric, body):
        if body.get("rubric_id") == 1:
            body["rubric_id"] = rubric["id"]
        assert client.post("/api/submissions", json=body, headers=auth(student)).status_code == 422

    def test_rejects_content_over_the_limit(self, client, auth, student, rubric):
        response = client.post(
            "/api/submissions",
            json={"rubric_id": rubric["id"], "content": "x" * 50_001},
            headers=auth(student),
        )
        assert response.status_code == 422


class TestListScoping:
    def test_a_student_sees_only_their_own(self, client, auth, student, other_student, submit):
        mine = submit(student)
        submit(other_student)
        response = client.get("/api/submissions", headers=auth(student))
        assert [s["id"] for s in response.get_json()] == [mine]

    def test_a_teacher_sees_submissions_on_their_rubrics(self, client, auth, teacher, student, other_student, submit):
        expected = sorted([submit(student), submit(other_student)])
        response = client.get("/api/submissions", headers=auth(teacher))
        assert sorted(s["id"] for s in response.get_json()) == expected

    def test_a_teacher_sees_nothing_on_rubrics_they_do_not_own(self, client, auth, other_teacher, student, submit):
        submit(student)
        assert client.get("/api/submissions", headers=auth(other_teacher)).get_json() == []

    def test_can_filter_by_rubric(self, client, auth, teacher, student, rubric, submit):
        submit(student)
        assert len(client.get(f"/api/submissions?rubric_id={rubric['id']}", headers=auth(teacher)).get_json()) == 1
        assert client.get("/api/submissions?rubric_id=99999", headers=auth(teacher)).get_json() == []

    def test_requires_a_login(self, client):
        assert client.get("/api/submissions").status_code == 401


class TestReadOne:
    def test_a_student_can_read_their_own(self, client, auth, student, submit):
        submission_id = submit(student)
        assert client.get(f"/api/submissions/{submission_id}", headers=auth(student)).status_code == 200

    def test_a_student_cannot_read_someone_elses(self, client, auth, student, other_student, submit):
        theirs = submit(other_student)
        assert client.get(f"/api/submissions/{theirs}", headers=auth(student)).status_code == 403

    def test_the_owning_teacher_can_read_it(self, client, auth, teacher, student, submit):
        submission_id = submit(student)
        assert client.get(f"/api/submissions/{submission_id}", headers=auth(teacher)).status_code == 200

    def test_another_teacher_cannot_read_it(self, client, auth, other_teacher, student, submit):
        submission_id = submit(student)
        assert client.get(f"/api/submissions/{submission_id}", headers=auth(other_teacher)).status_code == 403

    def test_missing_submission_is_404(self, client, auth, teacher):
        assert client.get("/api/submissions/99999", headers=auth(teacher)).status_code == 404


class TestAiFieldsAreTeacherOnly:
    """The student view must not carry the AI's working, approved or not."""

    def test_student_view_has_no_ai_keys(self, client, auth, student, submit, seed_ai_grade, criterion_ids):
        thesis, evidence = criterion_ids
        submission_id = seed_ai_grade(submit(student), {thesis: 4, evidence: 2})

        body = client.get(f"/api/submissions/{submission_id}", headers=auth(student)).get_json()
        assert "ai_total" not in body
        assert "ai_summary" not in body
        assert not any(key.startswith("ai_") for grade in body["grades"] for key in grade)

    def test_teacher_view_has_the_ai_keys(self, client, auth, teacher, student, submit, seed_ai_grade, criterion_ids):
        thesis, evidence = criterion_ids
        submission_id = seed_ai_grade(submit(student), {thesis: 4, evidence: 2})

        body = client.get(f"/api/submissions/{submission_id}", headers=auth(teacher)).get_json()
        assert body["ai_total"] == 6.0
        assert body["ai_summary"]
        assert all("ai_score" in grade for grade in body["grades"])
