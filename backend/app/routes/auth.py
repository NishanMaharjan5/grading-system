from flask import Blueprint, current_app, g, jsonify, request
from sqlalchemy.exc import IntegrityError

from app.extensions import db
from app.models.user import User
from app.security import create_token, hash_password, login_required, verify_password

auth_bp = Blueprint("auth", __name__)

VALID_ROLES = {"student", "teacher"}


def _token_response(user, status_code):
    token = create_token(user.id, user.email, user.role)
    return jsonify(access_token=token, token_type="bearer", name=user.name, role=user.role), status_code


@auth_bp.post("/register")
def register():
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    role = body.get("role", "student")
    teacher_code = body.get("teacher_code")

    if not name or len(name) > 100:
        return jsonify(detail="Name is required (max 100 characters)"), 422
    if "@" not in email or "." not in email.split("@")[-1]:
        return jsonify(detail="A valid email is required"), 422
    if not (8 <= len(password) <= 72):
        # bcrypt only uses the first 72 bytes, so cap the length rather than silently truncate
        return jsonify(detail="Password must be 8-72 characters"), 422
    if role not in VALID_ROLES:
        return jsonify(detail="role must be 'student' or 'teacher'"), 422

    if role == "teacher":
        # Without this gate anyone could self-register as a teacher and approve their own grades.
        required = current_app.config.get("TEACHER_SIGNUP_CODE")
        if required and teacher_code != required:
            return jsonify(detail="Invalid teacher signup code"), 403

    user = User(name=name, email=email, password_hash=hash_password(password), role=role)
    db.session.add(user)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify(detail="Email already registered"), 400

    return _token_response(user, 201)


@auth_bp.post("/login")
def login():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    user = db.session.query(User).filter_by(email=email).first()
    if not user or not verify_password(password, user.password_hash):
        return jsonify(detail="Invalid email or password"), 401

    return _token_response(user, 200)


@auth_bp.get("/me")
@login_required
def me():
    user = db.session.get(User, int(g.current_user["sub"]))
    if not user:
        return jsonify(detail="User no longer exists"), 401
    return jsonify(id=user.id, name=user.name, email=user.email, role=user.role), 200
