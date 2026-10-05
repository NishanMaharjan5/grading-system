"""Tests for how train_grader.py matches a training row to a criterion row.

This is the one place the pipeline crosses from strings (a rubric title and a
criterion name in sample_answers.json) to database row ids, and models are
saved under those ids. A wrong match here trains a real model against the
wrong criterion, silently. Nothing below trains anything or writes a model.
"""

import importlib.util
import os

import pytest

from app.extensions import db
from app.models import Rubric, RubricCriterion, User

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_train_grader():
    """scripts/ isn't a package, so load the module from its path."""
    path = os.path.join(BACKEND_DIR, "scripts", "train_grader.py")
    spec = importlib.util.spec_from_file_location("train_grader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_grader = _load_train_grader()


@pytest.fixture
def teacher_row(app):
    teacher = User(name="T", email="resolver@example.com", password_hash="x", role="teacher")
    db.session.add(teacher)
    db.session.commit()
    return teacher


def make_rubric(teacher, title, criteria=(("Thesis", 5), ("Evidence", 10))):
    rubric = Rubric(title=title, type="text", created_by=teacher.id)
    rubric.criteria = [RubricCriterion(name=n, max_points=p, position=i)
                       for i, (n, p) in enumerate(criteria)]
    db.session.add(rubric)
    db.session.commit()
    return rubric


def row(title="Essay 1", criterion="Thesis", text="An answer.", score=3):
    return {"rubric": title, "criterion": criterion, "text": text, "score": score}


class TestResolvesToTheRightRow:
    def test_matches_by_title_and_name(self, app, teacher_row):
        rubric = make_rubric(teacher_row, "Essay 1")
        criterion, reason = train_grader.resolve_criterion(row(), {})
        assert reason is None
        assert criterion.id == rubric.criteria[0].id

    def test_two_rubrics_with_shared_criterion_names_stay_separate(self, app, teacher_row):
        """The whole point of keying on row id: Essay 1 and Essay 2 both have a
        'Thesis', and they must resolve to different rows."""
        one = make_rubric(teacher_row, "Essay 1")
        two = make_rubric(teacher_row, "Essay 2")
        first, _ = train_grader.resolve_criterion(row("Essay 1", "Thesis"), {})
        second, _ = train_grader.resolve_criterion(row("Essay 2", "Thesis"), {})
        assert first.id == one.criteria[0].id
        assert second.id == two.criteria[0].id
        assert first.id != second.id

    def test_groups_rows_by_criterion_row(self, app, teacher_row):
        make_rubric(teacher_row, "Essay 1")
        make_rubric(teacher_row, "Essay 2")
        grouped = train_grader.resolve_all([
            row("Essay 1", "Thesis", "a", 1), row("Essay 1", "Thesis", "b", 2),
            row("Essay 2", "Thesis", "c", 3),
        ])
        assert sorted(len(v) for v in grouped.values()) == [1, 2]


class TestDuplicateTitlesAreRefused:
    """Models are saved under a criterion row id, so picking either of two
    rubrics sharing a title would train against the wrong criteria with no
    error. Two rubrics titled 'Essay 1' really did exist in the dev database."""

    def test_an_ambiguous_title_is_not_resolved(self, app, teacher_row):
        make_rubric(teacher_row, "Essay 1")
        make_rubric(teacher_row, "Essay 1")  # a duplicate, as happened for real
        criterion, reason = train_grader.resolve_criterion(row(), {})
        assert criterion is None
        assert "2 rubrics are titled" in reason

    def test_the_reason_names_both_rubric_ids(self, app, teacher_row):
        a = make_rubric(teacher_row, "Essay 1")
        b = make_rubric(teacher_row, "Essay 1")
        _, reason = train_grader.resolve_criterion(row(), {})
        assert str(a.id) in reason and str(b.id) in reason

    def test_training_stops_rather_than_picking_one(self, app, teacher_row):
        make_rubric(teacher_row, "Essay 1")
        make_rubric(teacher_row, "Essay 1")
        with pytest.raises(SystemExit) as caught:
            train_grader.resolve_all([row(), row()])
        assert "no model file was written" in str(caught.value)

    def test_a_unique_title_is_unaffected_by_a_duplicate_elsewhere(self, app, teacher_row):
        make_rubric(teacher_row, "Essay 1")
        make_rubric(teacher_row, "Essay 1")
        unique = make_rubric(teacher_row, "Essay 2")
        criterion, reason = train_grader.resolve_criterion(row("Essay 2", "Thesis"), {})
        assert reason is None and criterion.id == unique.criteria[0].id


class TestBlockedRowsAreReported:
    """'Which rows are wrong' is the thing you need to fix the file; one run
    should surface all of them."""

    def test_lists_the_blocked_row_numbers(self, app, teacher_row):
        make_rubric(teacher_row, "Essay 1")
        rows = [row(), row("Nope"), row(), row("Nope")]  # rows 2 and 4 are bad
        with pytest.raises(SystemExit) as caught:
            train_grader.resolve_all(rows)
        message = str(caught.value)
        assert "2 of 4 training rows could not be resolved" in message
        assert "2, 4" in message

    def test_reports_every_distinct_reason_not_just_the_first(self, app, teacher_row):
        make_rubric(teacher_row, "Essay 1")
        with pytest.raises(SystemExit) as caught:
            train_grader.resolve_all([row("Missing"), row("Essay 1", "Nonexistent")])
        message = str(caught.value)
        assert "no rubric titled 'Missing' exists" in message
        assert "has no criterion named 'Nonexistent'" in message

    def test_an_unknown_criterion_lists_what_is_available(self, app, teacher_row):
        make_rubric(teacher_row, "Essay 1")
        _, reason = train_grader.resolve_criterion(row(criterion="Structure"), {})
        assert "'Thesis'" in reason and "'Evidence'" in reason

    def test_a_long_run_of_bad_rows_is_truncated(self, app, teacher_row):
        make_rubric(teacher_row, "Essay 1")
        with pytest.raises(SystemExit) as caught:
            train_grader.resolve_all([row("Missing") for _ in range(25)])
        assert "and 15 more" in str(caught.value)

    def test_nothing_is_grouped_when_any_row_is_blocked(self, app, teacher_row):
        """A partial train would write models from a file the author thinks is
        whole, so it is all or nothing."""
        make_rubric(teacher_row, "Essay 1")
        with pytest.raises(SystemExit):
            train_grader.resolve_all([row(), row("Missing")])
