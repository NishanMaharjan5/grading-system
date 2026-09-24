"""Registration, login, /me, and the role guard."""

import pytest


class TestRegister:
    def test_registers_a_student_by_default(self, client):
        response = client.post(
            "/api/auth/register", json={"name": "S", "email": "s@example.com", "password": "password1"}
        )
        assert response.status_code == 201
        assert response.get_json()["role"] == "student"
        assert response.get_json()["token_type"] == "bearer"

    def test_registers_a_teacher(self, register):
        assert register("t@example.com", "teacher") is not None

    def test_rejects_a_duplicate_email(self, client, register):
        register("dupe@example.com")
        response = client.post(
            "/api/auth/register", json={"name": "S", "email": "dupe@example.com", "password": "password1"}
        )
        assert response.status_code == 400

    def test_duplicate_check_ignores_case(self, client, register):
        register("Mixed@Example.com")
        response = client.post(
            "/api/auth/register", json={"name": "S", "email": "mixed@example.com", "password": "password1"}
        )
        assert response.status_code == 400

    @pytest.mark.parametrize(
        "body",
        [
            {"name": "", "email": "a@example.com", "password": "password1"},
            {"name": "S", "email": "not-an-email", "password": "password1"},
            {"name": "S", "email": "a@example.com", "password": "short"},
            # bcrypt only reads the first 72 bytes, so anything longer is rejected
            # rather than silently truncated
            {"name": "S", "email": "a@example.com", "password": "x" * 73},
            {"name": "S", "email": "a@example.com", "password": "password1", "role": "admin"},
            {},
        ],
    )
    def test_rejects_invalid_input(self, client, body):
        assert client.post("/api/auth/register", json=body).status_code == 422

    def test_stores_a_hash_not_the_password(self, register):
        from app.extensions import db
        from app.models import User

        register("hash@example.com")
        user = db.session.query(User).filter_by(email="hash@example.com").one()
        assert user.password_hash != "password1"
        assert user.password_hash.startswith("$2")


class TestTeacherSignupCode:
    def test_code_not_required_when_unset(self, register):
        assert register("free@example.com", "teacher") is not None

    def test_rejects_a_teacher_without_the_code(self, client, app, monkeypatch):
        monkeypatch.setitem(app.config, "TEACHER_SIGNUP_CODE", "letmein")
        response = client.post(
            "/api/auth/register",
            json={"name": "T", "email": "t@example.com", "password": "password1", "role": "teacher"},
        )
        assert response.status_code == 403

    def test_rejects_a_wrong_code(self, client, app, monkeypatch):
        monkeypatch.setitem(app.config, "TEACHER_SIGNUP_CODE", "letmein")
        response = client.post(
            "/api/auth/register",
            json={
                "name": "T", "email": "t@example.com", "password": "password1",
                "role": "teacher", "teacher_code": "wrong",
            },
        )
        assert response.status_code == 403

    def test_accepts_the_right_code(self, client, app, monkeypatch):
        monkeypatch.setitem(app.config, "TEACHER_SIGNUP_CODE", "letmein")
        response = client.post(
            "/api/auth/register",
            json={
                "name": "T", "email": "t@example.com", "password": "password1",
                "role": "teacher", "teacher_code": "letmein",
            },
        )
        assert response.status_code == 201

    def test_the_gate_does_not_apply_to_students(self, client, app, monkeypatch):
        monkeypatch.setitem(app.config, "TEACHER_SIGNUP_CODE", "letmein")
        response = client.post(
            "/api/auth/register", json={"name": "S", "email": "s@example.com", "password": "password1"}
        )
        assert response.status_code == 201


class TestLogin:
    def test_logs_in(self, client, register):
        register("login@example.com")
        response = client.post("/api/auth/login", json={"email": "login@example.com", "password": "password1"})
        assert response.status_code == 200
        assert response.get_json()["access_token"]

    def test_email_is_case_insensitive(self, client, register):
        register("Case@Example.com")
        response = client.post("/api/auth/login", json={"email": "CASE@EXAMPLE.COM", "password": "password1"})
        assert response.status_code == 200

    def test_rejects_a_wrong_password(self, client, register):
        register("login@example.com")
        response = client.post("/api/auth/login", json={"email": "login@example.com", "password": "nope"})
        assert response.status_code == 401

    def test_rejects_an_unknown_email(self, client):
        response = client.post("/api/auth/login", json={"email": "ghost@example.com", "password": "password1"})
        assert response.status_code == 401


class TestMe:
    def test_returns_the_current_user(self, client, auth, student):
        response = client.get("/api/auth/me", headers=auth(student))
        assert response.status_code == 200
        assert response.get_json()["email"] == "student@example.com"
        assert response.get_json()["role"] == "student"

    def test_rejects_a_missing_token(self, client):
        assert client.get("/api/auth/me").status_code == 401

    def test_rejects_a_malformed_token(self, client, auth):
        assert client.get("/api/auth/me", headers=auth("not.a.jwt")).status_code == 401

    def test_rejects_a_token_signed_with_another_key(self, client, auth, app):
        from datetime import datetime, timedelta, timezone

        from jose import jwt

        forged = jwt.encode(
            {"sub": "1", "email": "x@example.com", "role": "teacher",
             "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
            "a-different-secret",
            algorithm=app.config["JWT_ALGORITHM"],
        )
        assert client.get("/api/auth/me", headers=auth(forged)).status_code == 401

    def test_rejects_an_expired_token(self, client, auth, app):
        from datetime import datetime, timedelta, timezone

        from jose import jwt

        expired = jwt.encode(
            {"sub": "1", "email": "x@example.com", "role": "student",
             "exp": datetime.now(timezone.utc) - timedelta(minutes=1)},
            app.config["JWT_SECRET"],
            algorithm=app.config["JWT_ALGORITHM"],
        )
        assert client.get("/api/auth/me", headers=auth(expired)).status_code == 401


class TestRoleGuard:
    """require_role, exercised through a real teacher-only endpoint."""

    def test_a_student_is_refused(self, client, auth, student):
        response = client.post(
            "/api/rubrics",
            json={"title": "R", "type": "text", "criteria": [{"name": "C", "max_points": 1}]},
            headers=auth(student),
        )
        assert response.status_code == 403

    def test_a_teacher_is_allowed(self, client, auth, teacher):
        response = client.post(
            "/api/rubrics",
            json={"title": "R", "type": "text", "criteria": [{"name": "C", "max_points": 1}]},
            headers=auth(teacher),
        )
        assert response.status_code == 201

    def test_an_anonymous_caller_is_refused(self, client):
        response = client.post(
            "/api/rubrics", json={"title": "R", "type": "text", "criteria": [{"name": "C", "max_points": 1}]}
        )
        assert response.status_code == 401
