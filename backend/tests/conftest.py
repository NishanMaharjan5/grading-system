"""Shared fixtures.

Two things worth knowing before adding tests here:

* The schema under test is built by running the Alembic migrations, not
  db.create_all(). That way the migrations are exercised on every run and
  cannot quietly drift away from the models.
* The model store is redirected at a throwaway directory for the whole
  session. Without that, any test which trains a model would overwrite the
  real backend/ml_models/*.joblib files, and -- because each test truncates
  with RESTART IDENTITY, so criterion ids begin at 1 again -- tests would
  otherwise load whatever model the last real training run left behind.
"""

import os

import pytest
from sqlalchemy import text

# app.config.Config reads these at import time, so they have to be set first.
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg2://grading_test:testpass@localhost:5432/grading_system_test",
)
os.environ.setdefault("JWT_SECRET", "test-secret-not-a-real-key")
os.environ.setdefault("TEACHER_SIGNUP_CODE", "")  # gate off by default; individual tests switch it on


@pytest.fixture(scope="session")
def app():
    from flask_migrate import upgrade

    from app import create_app
    from app.extensions import db

    application = create_app()
    with application.app_context():
        # Start from nothing, then let the migrations build the schema.
        db.drop_all()
        db.session.execute(text("DROP TABLE IF EXISTS alembic_version"))
        db.session.commit()
        upgrade()
    return application


@pytest.fixture(scope="session", autouse=True)
def _isolate_model_store(app, tmp_path_factory):
    from app.grading import model_store

    original = model_store.MODEL_DIR
    model_store.MODEL_DIR = str(tmp_path_factory.mktemp("ml_models"))
    model_store._cache.clear()
    yield
    model_store.MODEL_DIR = original
    model_store._cache.clear()


@pytest.fixture(autouse=True)
def clean_db(app):
    """Every test starts from an empty database with ids restarting at 1, so no
    test depends on what ran before it. The app context stays open for the
    duration of the test so tests can touch db.session directly."""
    from app.extensions import db
    from app.grading import model_store

    with app.app_context():
        tables = ", ".join(table.name for table in db.metadata.sorted_tables)
        db.session.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
        db.session.commit()
        model_store._cache.clear()  # cache is keyed by criterion id, which just restarted
        yield


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def auth():
    def _auth(token):
        return {"Authorization": f"Bearer {token}"}

    return _auth


@pytest.fixture
def register(client):
    def _register(email, role="student", password="password1", expect=201, **extra):
        body = {"name": email.split("@")[0], "email": email, "password": password, "role": role, **extra}
        response = client.post("/api/auth/register", json=body)
        assert response.status_code == expect, response.get_json()
        return response.get_json().get("access_token")

    return _register


@pytest.fixture
def teacher(register):
    return register("teacher@example.com", "teacher")


@pytest.fixture
def other_teacher(register):
    return register("other.teacher@example.com", "teacher")


@pytest.fixture
def student(register):
    return register("student@example.com")


@pytest.fixture
def other_student(register):
    return register("other.student@example.com")


@pytest.fixture
def rubric(client, auth, teacher):
    """Thesis out of 5 (with a description) and Evidence out of 10 (without),
    so tests can cover both description-quoting branches of the feedback."""
    response = client.post(
        "/api/rubrics",
        json={
            "title": "Essay 1",
            "type": "text",
            "criteria": [
                {"name": "Thesis", "max_points": 5, "description": "A clear, arguable claim."},
                {"name": "Evidence", "max_points": 10},
            ],
        },
        headers=auth(teacher),
    )
    assert response.status_code == 201, response.get_json()
    return response.get_json()


@pytest.fixture
def criterion_ids(rubric):
    """(thesis_id, evidence_id)"""
    return tuple(criterion["id"] for criterion in rubric["criteria"])


@pytest.fixture
def submit(client, auth, rubric):
    def _submit(token, content="An essay submitted by a student."):
        response = client.post(
            "/api/submissions", json={"rubric_id": rubric["id"], "content": content}, headers=auth(token)
        )
        assert response.status_code == 201, response.get_json()
        return response.get_json()["id"]

    return _submit


@pytest.fixture
def seed_ai_grade(app):
    """Force a submission into ai_graded with known scores, so the review tests
    assert against fixed numbers rather than whatever the model predicts."""

    def _seed(submission_id, scores):
        from app.extensions import db
        from app.grading import feedback
        from app.models import Grade, Submission

        submission = db.session.get(Submission, submission_id)
        submission.grades.clear()
        db.session.flush()

        criteria = list(submission.rubric.criteria)
        for criterion in criteria:
            db.session.add(
                Grade(
                    submission_id=submission.id,
                    criterion_id=criterion.id,
                    ai_score=scores[criterion.id],
                    ai_feedback=feedback.for_criterion(criterion, scores[criterion.id]),
                )
            )
        submission.ai_summary = feedback.summary(criteria, scores)
        submission.status = "ai_graded"
        db.session.commit()
        return submission_id

    return _seed


@pytest.fixture
def seed_grading_failed(app):
    """Force a submission into grading_failed: status set, no Grade rows at all,
    which is exactly what the engine leaves behind when it cannot grade."""

    def _seed(submission_id):
        from app.extensions import db
        from app.models import Submission

        submission = db.session.get(Submission, submission_id)
        submission.grades.clear()
        submission.ai_summary = None
        submission.status = "grading_failed"
        db.session.commit()
        return submission_id

    return _seed
