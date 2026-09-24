from flask import Blueprint, g, jsonify, request
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from app.extensions import db
from app.models import Rubric, RubricCriterion, Submission
from app.security import login_required, require_role

rubrics_bp = Blueprint("rubrics", __name__)

VALID_TYPES = {"text", "code"}


def _criterion_to_dict(c):
    return {
        "id": c.id,
        "name": c.name,
        "description": c.description,
        "max_points": float(c.max_points),
        "position": c.position,
    }


def _rubric_to_dict(r, submission_count=None, for_owner=True):
    """`locked` is the rule, not a hint: once work has been submitted against a
    rubric, editing or deleting it is refused (409). Naming it here means the
    frontend disables those actions from the server's answer instead of
    re-deriving the rule and drifting out of step with it.

    Both counts are the author's business, not a classmate's, so they are left
    out unless the caller owns the rubric.

    Callers listing many rubrics should pass submission_count to avoid a
    per-rubric count query."""
    if submission_count is None:
        submission_count = len(r.submissions)

    owner_fields = {"submission_count": submission_count, "locked": submission_count > 0} if for_owner else {}

    return {
        **owner_fields,
        "id": r.id,
        "title": r.title,
        "description": r.description,
        "type": r.type,
        "due_date": r.due_date.isoformat() if r.due_date else None,
        "created_by": r.created_by,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "total_points": float(r.total_points),
        "criteria": [_criterion_to_dict(c) for c in sorted(r.criteria, key=lambda c: c.position)],
    }


def _parse_criteria(raw):
    """Validate the incoming criteria list. Returns (criteria_kwargs_list, error_message)."""
    if not isinstance(raw, list) or not raw:
        return None, "criteria must be a non-empty list"

    parsed = []
    seen_names = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            return None, f"criteria[{i}] must be an object"
        name = (item.get("name") or "").strip()
        if not name or len(name) > 100:
            return None, f"criteria[{i}].name is required (max 100 characters)"
        if name.lower() in seen_names:
            return None, f"criteria[{i}].name '{name}' is duplicated in this rubric"
        seen_names.add(name.lower())
        try:
            max_points = float(item.get("max_points"))
        except (TypeError, ValueError):
            return None, f"criteria[{i}].max_points must be a number"
        if max_points <= 0:
            return None, f"criteria[{i}].max_points must be greater than 0"
        parsed.append({
            "name": name,
            "description": (item.get("description") or "").strip(),
            "max_points": max_points,
            "position": i,
        })
    return parsed, None


@rubrics_bp.post("")
@require_role("teacher")
def create_rubric():
    body = request.get_json(silent=True) or {}
    title = (body.get("title") or "").strip()
    rtype = body.get("type", "text")
    description = (body.get("description") or "").strip()
    due_date = body.get("due_date")  # ISO string or None; stored as-is via SQLAlchemy's DateTime coercion

    if not title or len(title) > 200:
        return jsonify(detail="title is required (max 200 characters)"), 422
    if rtype not in VALID_TYPES:
        return jsonify(detail="type must be 'text' or 'code'"), 422

    criteria, err = _parse_criteria(body.get("criteria"))
    if err:
        return jsonify(detail=err), 422

    rubric = Rubric(
        title=title, description=description, type=rtype,
        due_date=due_date, created_by=int(g.current_user["sub"]),
    )
    rubric.criteria = [RubricCriterion(**c) for c in criteria]

    db.session.add(rubric)
    db.session.commit()
    return jsonify(_rubric_to_dict(rubric)), 201


@rubrics_bp.get("")
@login_required
def list_rubrics():
    """Teachers get the rubrics they authored; students get everything they
    could submit against."""
    query = db.session.query(Rubric)
    if g.current_user.get("role") == "teacher":
        query = query.filter(Rubric.created_by == int(g.current_user["sub"]))
    rubrics = query.order_by(Rubric.created_at.desc()).all()

    # One grouped count rather than a lazy load per rubric.
    counts = dict(
        db.session.query(Submission.rubric_id, func.count(Submission.id)).group_by(Submission.rubric_id).all()
    )
    viewer_id = int(g.current_user["sub"])
    return jsonify([
        _rubric_to_dict(r, counts.get(r.id, 0), for_owner=r.created_by == viewer_id) for r in rubrics
    ]), 200


@rubrics_bp.get("/<int:rubric_id>")
@login_required
def get_rubric(rubric_id):
    rubric = db.session.get(Rubric, rubric_id)
    if not rubric:
        return jsonify(detail="Rubric not found"), 404
    return jsonify(_rubric_to_dict(rubric, for_owner=rubric.created_by == int(g.current_user["sub"]))), 200


@rubrics_bp.put("/<int:rubric_id>")
@require_role("teacher")
def update_rubric(rubric_id):
    rubric = db.session.get(Rubric, rubric_id)
    if not rubric:
        return jsonify(detail="Rubric not found"), 404
    if rubric.created_by != int(g.current_user["sub"]):
        return jsonify(detail="You can only edit rubrics you created"), 403

    # Once a student has submitted against this rubric, changing point totals would
    # silently invalidate any grades already recorded against it.
    if rubric.submissions:
        return jsonify(detail="This rubric already has submissions and can no longer be edited"), 409

    body = request.get_json(silent=True) or {}

    if "title" in body:
        title = (body.get("title") or "").strip()
        if not title or len(title) > 200:
            return jsonify(detail="title must be 1-200 characters"), 422
        rubric.title = title
    if "description" in body:
        rubric.description = (body.get("description") or "").strip()
    if "due_date" in body:
        rubric.due_date = body.get("due_date")
    if "type" in body:
        if body["type"] not in VALID_TYPES:
            return jsonify(detail="type must be 'text' or 'code'"), 422
        rubric.type = body["type"]
    if "criteria" in body:
        criteria, err = _parse_criteria(body.get("criteria"))
        if err:
            return jsonify(detail=err), 422
        # Clear and flush before adding: assigning straight over the list makes
        # SQLAlchemy insert the replacements before deleting the originals, and
        # uq_criteria_rubric_name then rejects any criterion whose name is being
        # kept -- which is most edits (retitling, changing points).
        rubric.criteria.clear()
        db.session.flush()
        rubric.criteria = [RubricCriterion(**c) for c in criteria]

    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify(detail="Could not save rubric (constraint violated)"), 400

    return jsonify(_rubric_to_dict(rubric)), 200


@rubrics_bp.delete("/<int:rubric_id>")
@require_role("teacher")
def delete_rubric(rubric_id):
    rubric = db.session.get(Rubric, rubric_id)
    if not rubric:
        return jsonify(detail="Rubric not found"), 404
    if rubric.created_by != int(g.current_user["sub"]):
        return jsonify(detail="You can only delete rubrics you created"), 403

    db.session.delete(rubric)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify(detail="This rubric has submissions and cannot be deleted"), 409

    return "", 204
