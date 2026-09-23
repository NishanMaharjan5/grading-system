from flask import Blueprint, current_app, g, jsonify, request
from sqlalchemy.exc import IntegrityError

from app.extensions import db
from app.grading.engine import GradingError, grade_text_submission
from app.models import Grade, Rubric, Submission
from app.security import login_required, require_role

submissions_bp = Blueprint("submissions", __name__)

MAX_CONTENT_LENGTH = 50_000


def _auto_grade(submission, rubric):
    """Best-effort: the submission itself is already committed either way. On
    success it becomes 'ai_graded' with a Grade row per criterion; on any
    failure it becomes 'grading_failed' so a teacher can grade it by hand."""
    try:
        scores = grade_text_submission(submission.content, rubric.criteria)
    except GradingError as e:
        current_app.logger.info("Auto-grade skipped for submission %s: %s", submission.id, e)
        submission.status = "grading_failed"
        db.session.commit()
        return
    except Exception:
        current_app.logger.exception("Auto-grade failed for submission %s", submission.id)
        db.session.rollback()
        submission.status = "grading_failed"
        db.session.commit()
        return

    for criterion in rubric.criteria:
        db.session.add(Grade(submission_id=submission.id, criterion_id=criterion.id, ai_score=scores[criterion.id]))
    submission.status = "ai_graded"
    db.session.commit()


def _grade_to_dict(grade, include_ai):
    d = {
        "criterion_id": grade.criterion_id,
        "final_score": float(grade.final_score) if grade.final_score is not None else None,
        "final_feedback": grade.final_feedback,
    }
    if include_ai:
        d["ai_score"] = float(grade.ai_score) if grade.ai_score is not None else None
        d["ai_feedback"] = grade.ai_feedback
    return d


def _submission_to_dict(sub, *, for_teacher):
    """Teachers see the AI's suggestion alongside the final grade; students only
    ever see the approved final grade -- never an unapproved AI score."""
    base = {
        "id": sub.id,
        "rubric_id": sub.rubric_id,
        "student_id": sub.student_id,
        "content": sub.content,
        "status": sub.status,
        "created_at": sub.created_at.isoformat() if sub.created_at else None,
        "final_total": float(sub.final_total) if sub.final_total is not None else None,
        "grades": [_grade_to_dict(gr, include_ai=for_teacher) for gr in sub.grades],
    }
    if for_teacher:
        base["ai_total"] = float(sub.ai_total) if sub.ai_total is not None else None
    return base


@submissions_bp.post("")
@require_role("student")
def create_submission():
    body = request.get_json(silent=True) or {}
    rubric_id = body.get("rubric_id")
    content = (body.get("content") or "")

    if not isinstance(rubric_id, int):
        return jsonify(detail="rubric_id is required"), 422
    if not content.strip():
        return jsonify(detail="content is required"), 422
    if len(content) > MAX_CONTENT_LENGTH:
        return jsonify(detail=f"content must be at most {MAX_CONTENT_LENGTH} characters"), 422

    rubric = db.session.get(Rubric, rubric_id)
    if not rubric:
        return jsonify(detail="Rubric not found"), 404

    submission = Submission(
        rubric_id=rubric_id,
        student_id=int(g.current_user["sub"]),
        content=content,
        status="submitted",
    )
    db.session.add(submission)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify(detail="You have already submitted for this rubric"), 409

    # Code submissions are graded by a separate sandboxed test runner, not built yet
    if rubric.type == "text":
        _auto_grade(submission, rubric)

    return jsonify(_submission_to_dict(submission, for_teacher=False)), 201


@submissions_bp.get("")
@login_required
def list_submissions():
    role = g.current_user.get("role")
    user_id = int(g.current_user["sub"])
    rubric_id = request.args.get("rubric_id", type=int)

    query = db.session.query(Submission)
    if role == "student":
        query = query.filter(Submission.student_id == user_id)
    else:
        # Teachers only see submissions for rubrics they created
        query = query.join(Rubric).filter(Rubric.created_by == user_id)
    if rubric_id is not None:
        query = query.filter(Submission.rubric_id == rubric_id)

    submissions = query.order_by(Submission.created_at.desc()).all()
    return jsonify([_submission_to_dict(s, for_teacher=(role == "teacher")) for s in submissions]), 200


@submissions_bp.get("/<int:submission_id>")
@login_required
def get_submission(submission_id):
    submission = db.session.get(Submission, submission_id)
    if not submission:
        return jsonify(detail="Submission not found"), 404

    role = g.current_user.get("role")
    user_id = int(g.current_user["sub"])
    if role == "student" and submission.student_id != user_id:
        return jsonify(detail="You can only view your own submissions"), 403
    if role == "teacher" and submission.rubric.created_by != user_id:
        return jsonify(detail="You can only view submissions for rubrics you created"), 403

    return jsonify(_submission_to_dict(submission, for_teacher=(role == "teacher"))), 200
