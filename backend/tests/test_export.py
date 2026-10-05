"""The gradebook CSV export.

One row per submission, a score column per criterion, scoped to rubrics the
requesting teacher owns. An empty gradebook is a real answer rather than an
error: a rubric nobody has submitted to still returns a file with its header.
"""

import csv
import io

import pytest

from app.extensions import db
from app.models import Submission


def read_csv(response):
    return list(csv.reader(io.StringIO(response.get_data(as_text=True))))


def export(client, auth, token, rubric_id):
    return client.get(f"/api/rubrics/{rubric_id}/export", headers=auth(token))


@pytest.fixture
def graded_rubric(client, auth, teacher, student, submit, criterion_ids, seed_ai_grade, rubric):
    """One approved submission scoring Thesis 3, Evidence 7."""
    submission_id = submit(student)
    thesis, evidence = criterion_ids
    seed_ai_grade(submission_id, {thesis: 3, evidence: 7})
    assert client.put(f"/api/submissions/{submission_id}/review", json={},
                      headers=auth(teacher)).status_code == 200
    return rubric, submission_id


class TestContents:
    def test_the_header_names_every_criterion(self, client, auth, teacher, rubric):
        rows = read_csv(export(client, auth, teacher, rubric["id"]))
        header = rows[0]
        assert "student_name" in header and "student_email" in header
        assert any("Thesis" in column for column in header)
        assert any("Evidence" in column for column in header)
        assert header[-4:] == ["final_total", "total_possible", "approved_at", "revised"]

    def test_a_graded_submission_appears_with_its_scores(self, client, auth, teacher, graded_rubric):
        rubric, _ = graded_rubric
        rows = read_csv(export(client, auth, teacher, rubric["id"]))
        assert len(rows) == 2
        body = rows[1]
        assert body[0] == "Student" or body[1] == "student@example.com"
        assert "3" in body and "7" in body
        assert "10" in body, "final total"

    def test_the_student_is_named_and_identified(self, client, auth, teacher, graded_rubric):
        rubric, _ = graded_rubric
        header, row = read_csv(export(client, auth, teacher, rubric["id"]))
        assert row[header.index("student_email")] == "student@example.com"

    def test_ai_accepted_is_recorded_per_criterion(self, client, auth, teacher, graded_rubric):
        """Approving as-is means the AI's number was taken for every criterion."""
        rubric, _ = graded_rubric
        header, row = read_csv(export(client, auth, teacher, rubric["id"]))
        accepted = [row[i] for i, name in enumerate(header) if name.endswith("ai_accepted")]
        assert accepted == ["true", "true"]

    def test_approved_at_is_filled_for_a_released_grade(self, client, auth, teacher, graded_rubric):
        rubric, _ = graded_rubric
        header, row = read_csv(export(client, auth, teacher, rubric["id"]))
        assert row[header.index("approved_at")] != ""

    def test_an_unapproved_submission_has_blank_scores_rather_than_zeros(
            self, client, auth, teacher, student, submit, rubric):
        """A missing grade is not a grade of nothing."""
        submit(student)
        header, row = read_csv(export(client, auth, teacher, rubric["id"]))
        assert row[header.index("final_total")] == ""
        assert row[header.index("approved_at")] == ""
        assert row[header.index("revised")] == "no"

    def test_a_revised_submission_is_flagged(self, client, auth, teacher, graded_rubric, criterion_ids):
        rubric, submission_id = graded_rubric
        thesis, evidence = criterion_ids
        assert client.put(f"/api/submissions/{submission_id}/revise", headers=auth(teacher), json={
            "criterion_scores": [{"criterion_id": thesis, "final_score": 5},
                                 {"criterion_id": evidence, "final_score": 9}]}).status_code == 200
        header, row = read_csv(export(client, auth, teacher, rubric["id"]))
        assert row[header.index("revised")] == "yes"
        assert row[header.index("final_total")] == "14", "the corrected total, not the original"

    def test_every_submission_gets_a_row(self, client, auth, teacher, student, other_student, submit, rubric):
        submit(student)
        submit(other_student)
        rows = read_csv(export(client, auth, teacher, rubric["id"]))
        assert len(rows) == 3

    def test_rows_cover_only_this_rubric(self, client, auth, teacher, student, submit, rubric):
        """A second rubric's work must not leak into this gradebook."""
        other = client.post("/api/rubrics", headers=auth(teacher), json={
            "title": "Other", "type": "text",
            "criteria": [{"name": "Thesis", "max_points": 5}]}).get_json()
        submit(student)
        client.post("/api/submissions", headers=auth(student),
                    json={"rubric_id": other["id"], "content": "Different assignment."})
        assert len(read_csv(export(client, auth, teacher, rubric["id"]))) == 2
        assert len(read_csv(export(client, auth, teacher, other["id"]))) == 2


class TestEmptyGradebook:
    def test_a_rubric_with_no_submissions_is_not_an_error(self, client, auth, teacher, rubric):
        response = export(client, auth, teacher, rubric["id"])
        assert response.status_code == 200

    def test_it_still_has_its_header_row(self, client, auth, teacher, rubric):
        rows = read_csv(export(client, auth, teacher, rubric["id"]))
        assert len(rows) == 1 and rows[0][0] == "student_name"


class TestAccessControl:
    def test_another_teacher_cannot_export_it(self, client, auth, other_teacher, rubric):
        assert export(client, auth, other_teacher, rubric["id"]).status_code == 403

    def test_a_student_cannot_export(self, client, auth, student, rubric):
        assert export(client, auth, student, rubric["id"]).status_code == 403

    def test_an_unknown_rubric_is_404(self, client, auth, teacher):
        assert export(client, auth, teacher, 999999).status_code == 404


class TestResponseShape:
    def test_it_is_served_as_a_csv_download(self, client, auth, teacher, rubric):
        response = export(client, auth, teacher, rubric["id"])
        assert response.mimetype == "text/csv"
        assert "attachment" in response.headers["Content-Disposition"]

    def test_the_filename_names_the_rubric(self, client, auth, teacher, rubric):
        disposition = export(client, auth, teacher, rubric["id"]).headers["Content-Disposition"]
        assert "gradebook" in disposition and ".csv" in disposition

    def test_an_awkward_title_still_produces_a_safe_filename(self, client, auth, teacher):
        made = client.post("/api/rubrics", headers=auth(teacher), json={
            "title": 'Essay "1"/2: drafts', "type": "text",
            "criteria": [{"name": "Thesis", "max_points": 5}]}).get_json()
        disposition = export(client, auth, teacher, made["id"]).headers["Content-Disposition"]
        assert '"' not in disposition.split("filename=")[1].strip('"')
        assert "/" not in disposition.split("filename=")[1]
