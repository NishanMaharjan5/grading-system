"""Which parts of a rubric freeze once work has been submitted.

The rule used to be all-or-nothing: one submission and the whole rubric was
read-only. That protected recorded grades, but it also made the thing a teacher
most often needs after work starts arriving -- extending a deadline --
impossible, which is a worse failure than the one it prevented.

Now only the scoring freezes: criteria names, points, order, test cases, and
the rubric's type. Those decide what a mark means, so changing them under an
existing grade would quietly make that grade describe something else. The
title, description, due date and word limits touch no recorded grade and stay
editable.

Deleting is still refused outright -- that would destroy the students' work.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.extensions import db
from app.models import Grade, Rubric


def put(client, auth, teacher, rubric_id, body):
    return client.put(f"/api/rubrics/{rubric_id}", json=body, headers=auth(teacher))


def in_hours(n):
    return (datetime.now(timezone.utc) + timedelta(hours=n)).isoformat()


@pytest.fixture
def submitted(client, auth, teacher, student, rubric, submit):
    """A rubric with one submission against it, so the scoring is frozen."""
    submit(student)
    return rubric


class TestNonScoringFieldsStayEditable:
    def test_the_due_date_can_be_moved(self, client, auth, teacher, submitted):
        """The case this was built for: extending a deadline after work has
        started arriving."""
        response = put(client, auth, teacher, submitted["id"], {"due_date": in_hours(48)})
        assert response.status_code == 200
        assert response.get_json()["due_date"] is not None

    def test_a_deadline_can_be_added_where_there_was_none(self, client, auth, teacher, submitted):
        assert submitted["due_date"] is None
        assert put(client, auth, teacher, submitted["id"], {"due_date": in_hours(24)}).status_code == 200

    def test_a_deadline_can_be_cleared(self, client, auth, teacher, submitted):
        put(client, auth, teacher, submitted["id"], {"due_date": in_hours(24)})
        response = put(client, auth, teacher, submitted["id"], {"due_date": ""})
        assert response.status_code == 200 and response.get_json()["due_date"] is None

    def test_the_title_can_be_changed(self, client, auth, teacher, submitted):
        response = put(client, auth, teacher, submitted["id"], {"title": "Essay 1 (revised brief)"})
        assert response.status_code == 200
        assert response.get_json()["title"] == "Essay 1 (revised brief)"

    def test_the_description_can_be_changed(self, client, auth, teacher, submitted):
        response = put(client, auth, teacher, submitted["id"], {"description": "A clearer brief."})
        assert response.status_code == 200
        assert response.get_json()["description"] == "A clearer brief."

    def test_word_limits_can_be_changed(self, client, auth, teacher, submitted):
        """They gate future submissions, not recorded grades, so they travel
        with the deadline rather than with the scoring."""
        response = put(client, auth, teacher, submitted["id"], {"min_words": 10, "max_words": 150})
        assert response.status_code == 200
        assert (response.get_json()["min_words"], response.get_json()["max_words"]) == (10, 150)

    def test_several_non_scoring_fields_at_once(self, client, auth, teacher, submitted):
        response = put(client, auth, teacher, submitted["id"], {
            "title": "New title", "description": "New brief", "due_date": in_hours(72), "max_words": 300,
        })
        assert response.status_code == 200

    def test_the_change_actually_persists(self, client, auth, teacher, submitted):
        put(client, auth, teacher, submitted["id"], {"due_date": in_hours(48)})
        assert db.session.get(Rubric, submitted["id"]).due_date is not None


class TestScoringStaysFrozen:
    def test_changing_max_points_is_refused(self, client, auth, teacher, submitted, criteria_payload):
        repriced = [dict(c, max_points=99) for c in criteria_payload]
        response = put(client, auth, teacher, submitted["id"], {"criteria": repriced})
        assert response.status_code == 409
        assert "criteria, points and type" in response.get_json()["detail"]

    def test_renaming_a_criterion_is_refused(self, client, auth, teacher, submitted, criteria_payload):
        renamed = [dict(c, name="Argument") if c["name"] == "Thesis" else c for c in criteria_payload]
        assert put(client, auth, teacher, submitted["id"], {"criteria": renamed}).status_code == 409

    def test_adding_a_criterion_is_refused(self, client, auth, teacher, submitted, criteria_payload):
        extended = criteria_payload + [{"name": "Structure", "max_points": 5}]
        assert put(client, auth, teacher, submitted["id"], {"criteria": extended}).status_code == 409

    def test_removing_a_criterion_is_refused(self, client, auth, teacher, submitted, criteria_payload):
        assert put(client, auth, teacher, submitted["id"], {"criteria": criteria_payload[:1]}).status_code == 409

    def test_reordering_criteria_is_refused(self, client, auth, teacher, submitted, criteria_payload):
        """Position is part of the signature: grades are reported in rubric
        order, so reordering changes what a mark sheet means."""
        assert put(client, auth, teacher, submitted["id"],
                   {"criteria": list(reversed(criteria_payload))}).status_code == 409

    def test_changing_the_type_is_refused(self, client, auth, teacher, submitted):
        assert put(client, auth, teacher, submitted["id"], {"type": "code"}).status_code == 409

    def test_a_refused_edit_changes_nothing(self, client, auth, teacher, submitted, criteria_payload):
        repriced = [dict(c, max_points=99) for c in criteria_payload]
        put(client, auth, teacher, submitted["id"], {"title": "Should not stick", "criteria": repriced})
        stored = db.session.get(Rubric, submitted["id"])
        assert stored.title == "Essay 1"
        assert sorted(float(c.max_points) for c in stored.criteria) == [5.0, 10.0]

    def test_deleting_is_still_refused(self, client, auth, teacher, submitted):
        """Unchanged: this would destroy the students' work."""
        assert client.delete(f"/api/rubrics/{submitted['id']}", headers=auth(teacher)).status_code == 409


class TestTheFormCanStillSaveTheWholeRubric:
    """The edit form submits every field each time, including criteria it did
    not touch. Refusing on key presence alone would make a pure due-date change
    impossible from the UI, so the check compares against what is stored."""

    def test_resending_identical_criteria_is_allowed(self, client, auth, teacher, submitted, criteria_payload):
        response = put(client, auth, teacher, submitted["id"], {
            "title": "Essay 1", "type": "text", "criteria": criteria_payload, "due_date": in_hours(48),
        })
        assert response.status_code == 200, response.get_json()

    def test_resending_identical_criteria_with_the_type_is_allowed(
            self, client, auth, teacher, submitted, criteria_payload):
        response = put(client, auth, teacher, submitted["id"],
                       {"type": "text", "criteria": criteria_payload})
        assert response.status_code == 200

    def test_a_criterion_description_may_be_reworded(self, client, auth, teacher, submitted, criteria_payload):
        """Rewording "a clear, arguable claim" does not change what five points
        is worth, so it is not part of the scoring signature."""
        reworded = [dict(c, description="Reworded guidance for students.") for c in criteria_payload]
        assert put(client, auth, teacher, submitted["id"], {"criteria": reworded}).status_code == 200


class TestBeforeAnySubmission:
    """Nothing is frozen until work exists."""

    def test_scoring_can_be_changed_freely(self, client, auth, teacher, rubric, criteria_payload):
        repriced = [dict(c, max_points=50) for c in criteria_payload]
        assert put(client, auth, teacher, rubric["id"], {"criteria": repriced}).status_code == 200

    def test_the_type_can_be_changed(self, client, auth, teacher, rubric):
        response = put(client, auth, teacher, rubric["id"], {
            "type": "code",
            "criteria": [{"name": "Correctness", "max_points": 10,
                          "test_cases": [{"stdin": "", "expected_output": "x"}]}],
        })
        assert response.status_code == 200


class TestOwnershipIsUnchanged:
    def test_another_teacher_still_cannot_edit(self, client, auth, other_teacher, submitted):
        response = client.put(f"/api/rubrics/{submitted['id']}",
                              json={"due_date": in_hours(48)}, headers=auth(other_teacher))
        assert response.status_code == 403

    def test_a_student_still_cannot_edit(self, client, auth, student, submitted):
        response = client.put(f"/api/rubrics/{submitted['id']}",
                              json={"due_date": in_hours(48)}, headers=auth(student))
        assert response.status_code == 403


class TestGradedWorkSurvivesANonScoringEdit:
    """The case that broke first time round.

    The edit form resends the whole rubric, so a due-date change arrives with
    the criteria attached. The handler used to clear and rebuild those rows on
    every save, which is fine while nothing points at them -- and a hard FK
    violation once grades do, because a grade references its criterion by id.
    Every test here has real Grade rows, which the plain `submit` fixture does
    not produce when the engine has no model for the criteria.
    """

    @pytest.fixture
    def graded(self, client, auth, teacher, student, rubric, submit, seed_ai_grade, criterion_ids):
        submission_id = submit(student)
        thesis, evidence = criterion_ids
        seed_ai_grade(submission_id, {thesis: 4, evidence: 8})
        assert db.session.query(Grade).filter_by(submission_id=submission_id).count() == 2
        return rubric, submission_id

    def test_a_due_date_change_carrying_the_criteria_succeeds(
            self, client, auth, teacher, graded, criteria_payload):
        rubric, _ = graded
        response = put(client, auth, teacher, rubric["id"], {
            "title": rubric["title"], "type": "text",
            "criteria": criteria_payload, "due_date": in_hours(72),
        })
        assert response.status_code == 200, response.get_json()

    def test_the_grades_still_point_at_their_criteria(
            self, client, auth, teacher, graded, criteria_payload):
        rubric, submission_id = graded
        before = {(g.criterion_id, float(g.ai_score))
                  for g in db.session.query(Grade).filter_by(submission_id=submission_id)}
        put(client, auth, teacher, rubric["id"], {
            "title": "Retitled", "type": "text", "criteria": criteria_payload, "due_date": in_hours(72),
        })
        after = {(g.criterion_id, float(g.ai_score))
                 for g in db.session.query(Grade).filter_by(submission_id=submission_id)}
        assert after == before, "grades were orphaned or rewritten by a non-scoring edit"

    def test_the_criterion_rows_keep_their_ids(self, client, auth, teacher, graded, criteria_payload):
        """Replacing a row with an identical one would still break every grade
        and every trained model, which are keyed by criterion id."""
        rubric, _ = graded
        before = sorted(c["id"] for c in rubric["criteria"])
        put(client, auth, teacher, rubric["id"], {
            "type": "text", "criteria": criteria_payload, "due_date": in_hours(72),
        })
        after = sorted(c.id for c in db.session.get(Rubric, rubric["id"]).criteria)
        assert after == before

    def test_a_description_reword_keeps_the_ids_too(self, client, auth, teacher, graded, criteria_payload):
        rubric, _ = graded
        before = sorted(c["id"] for c in rubric["criteria"])
        reworded = [dict(c, description="Reworded.") for c in criteria_payload]
        response = put(client, auth, teacher, rubric["id"], {"criteria": reworded})
        assert response.status_code == 200
        stored = db.session.get(Rubric, rubric["id"])
        assert sorted(c.id for c in stored.criteria) == before
        assert all(c.description == "Reworded." for c in stored.criteria)
