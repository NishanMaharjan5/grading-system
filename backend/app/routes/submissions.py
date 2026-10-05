from datetime import datetime, timezone

from flask import Blueprint, current_app, g, jsonify, request
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload

from app.extensions import db
from app.grading import feedback as feedback_templates
from app.grading.engine import GradingError, grade_submission
from app.models import Grade, GradeRevision, Rubric, Submission
from app.security import login_required, require_role
from app.deadlines import check_deadline
from app.word_limits import check_submission_length

submissions_bp = Blueprint("submissions", __name__)

MAX_CONTENT_LENGTH = 50_000


def _auto_grade(submission, rubric):
    """Best-effort: the submission itself is already committed either way. On
    success it becomes 'ai_graded' with a Grade row per criterion; on any
    failure it becomes 'grading_failed' so a teacher can grade it by hand."""
    try:
        result = grade_submission(rubric, submission.content)
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
        db.session.add(Grade(
            submission_id=submission.id,
            criterion_id=criterion.id,
            ai_score=result["scores"][criterion.id],
            ai_feedback=result["feedback"][criterion.id],
        ))
    submission.ai_summary = result["summary"]
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
        d["ai_accepted"] = grade.ai_accepted
    return d


def _revision_to_dict(rev):
    return {
        "id": rev.id,
        "criterion_id": rev.criterion_id,
        "criterion_name": rev.criterion.name if rev.criterion else None,
        "old_final_score": float(rev.old_final_score) if rev.old_final_score is not None else None,
        "new_final_score": float(rev.new_final_score),
        "old_final_feedback": rev.old_final_feedback,
        "new_final_feedback": rev.new_final_feedback,
        "revised_by": rev.revised_by,
        "revised_by_name": rev.reviser.name if rev.reviser else None,
        "revised_at": rev.revised_at.isoformat() if rev.revised_at else None,
    }


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
        "final_summary": sub.final_summary,
        "grades": [_grade_to_dict(gr, include_ai=for_teacher) for gr in sub.grades],
    }
    if for_teacher:
        base["ai_total"] = float(sub.ai_total) if sub.ai_total is not None else None
        base["ai_summary"] = sub.ai_summary
        # Without these, two submissions to the same rubric are
        # indistinguishable in the review queue. Email alongside the name
        # because names aren't unique.
        base["student_name"] = sub.student.name
        base["student_email"] = sub.student.email
        # Teacher-only. A student is shown the grade that stands, not a list of
        # the teacher's corrections to it.
        base["revisions"] = [_revision_to_dict(rev) for rev in sub.revisions]
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

    # The server is the authority on both of these; the form mirrors them so a
    # student sees the problem before pressing the button, not after.
    closed = check_deadline(rubric)
    if closed:
        return jsonify(detail=closed), 422

    # The server is the authority on the word range; the form mirrors this
    # check so a student sees it before pressing the button, not after.
    too_long_or_short = check_submission_length(content, rubric)
    if too_long_or_short:
        return jsonify(detail=too_long_or_short), 422

    student_id = int(g.current_user["sub"])
    existing = (db.session.query(Submission)
                .filter_by(rubric_id=rubric_id, student_id=student_id).first())

    if existing:
        # Resubmission overwrites in place rather than keeping every attempt:
        # one row per student per rubric, which is what the unique constraint
        # already says. The deadline was checked above, so the only thing that
        # can still close this off is a teacher having released a grade.
        if existing.status == "approved":
            return jsonify(detail="This has already been graded by your teacher"), 409

        existing.content = content
        existing.status = "submitted"
        # Everything derived from the old text is now wrong. Clearing it before
        # re-grading means a failure leaves no stale score attached to text that
        # no longer exists.
        existing.ai_summary = None
        existing.final_summary = None
        for grade in list(existing.grades):
            db.session.delete(grade)
        db.session.commit()

        _auto_grade(existing, rubric)
        return jsonify(_submission_to_dict(existing, for_teacher=False)), 200

    submission = Submission(
        rubric_id=rubric_id,
        student_id=student_id,
        content=content,
        status="submitted",
    )
    db.session.add(submission)
    try:
        db.session.commit()
    except IntegrityError:
        # Two submissions racing: the loser re-reads and takes the overwrite
        # path next time rather than losing the student's work to a 409.
        db.session.rollback()
        return jsonify(detail="That submission collided with another; try again"), 409

    _auto_grade(submission, rubric)

    return jsonify(_submission_to_dict(submission, for_teacher=False)), 201


@submissions_bp.get("")
@login_required
def list_submissions():
    role = g.current_user.get("role")
    user_id = int(g.current_user["sub"])
    rubric_id = request.args.get("rubric_id", type=int)

    query = db.session.query(Submission).options(joinedload(Submission.student))
    if role == "student":
        query = query.filter(Submission.student_id == user_id)
    else:
        # Teachers only see submissions for rubrics they created
        query = query.join(Rubric).filter(Rubric.created_by == user_id)
    if rubric_id is not None:
        query = query.filter(Submission.rubric_id == rubric_id)

    submissions = query.order_by(Submission.created_at.desc()).all()
    return jsonify([_submission_to_dict(s, for_teacher=(role == "teacher")) for s in submissions]), 200


@submissions_bp.get("/pending")
@require_role("teacher")
def list_pending_review():
    """The teacher's review queue: auto-graded work awaiting sign-off, plus work
    the engine couldn't grade that needs scoring by hand. Registered before the
    /<int:submission_id> rule -- the int converter won't match "pending" anyway,
    but keeping them adjacent makes that obvious to the next reader."""
    submissions = (
        db.session.query(Submission).options(joinedload(Submission.student))
        .join(Rubric)
        .filter(Rubric.created_by == int(g.current_user["sub"]))
        .filter(Submission.status.in_(("ai_graded", "grading_failed")))
        .order_by(Submission.created_at.desc())
        .all()
    )
    return jsonify([_submission_to_dict(s, for_teacher=True) for s in submissions]), 200


def _parse_review_scores(raw, criteria):
    """Validates a teacher's override payload.
    Returns ({criterion_id: (score, feedback)}, error_message)."""
    if not isinstance(raw, list) or not raw:
        return None, "criterion_scores must be a non-empty list"

    by_id = {c.id: c for c in criteria}
    parsed = {}
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            return None, f"criterion_scores[{i}] must be an object"

        criterion_id = item.get("criterion_id")
        if criterion_id not in by_id:
            return None, f"criterion_scores[{i}].criterion_id {criterion_id!r} is not a criterion on this rubric"
        if criterion_id in parsed:
            return None, f"criterion_scores[{i}].criterion_id {criterion_id} appears more than once"

        try:
            score = float(item.get("final_score"))
        except (TypeError, ValueError):
            return None, f"criterion_scores[{i}].final_score must be a number"

        max_points = float(by_id[criterion_id].max_points)
        if not (0 <= score <= max_points):
            return None, (f"criterion_scores[{i}].final_score {score:g} is outside "
                          f"0..{max_points:g} for {by_id[criterion_id].name!r}")

        feedback = item.get("final_feedback")
        if feedback is not None and not isinstance(feedback, str):
            return None, f"criterion_scores[{i}].final_feedback must be a string"

        parsed[criterion_id] = (score, feedback)

    # Partial approval would leave some criteria unscored, so final_total could
    # never be computed and the student would see an incomplete grade.
    missing = [c.name for c in criteria if c.id not in parsed]
    if missing:
        return None, f"Every criterion needs a score — missing: {', '.join(missing)}"
    return parsed, None


@submissions_bp.put("/<int:submission_id>/review")
@require_role("teacher")
def review_submission(submission_id):
    """Approve the AI's suggestion as-is (send no body), or override it by
    supplying criterion_scores. Either way the submission ends up 'approved'
    and the grade becomes visible to the student."""
    submission = db.session.get(Submission, submission_id)
    if not submission:
        return jsonify(detail="Submission not found"), 404

    teacher_id = int(g.current_user["sub"])
    if submission.rubric.created_by != teacher_id:
        return jsonify(detail="You can only review submissions for rubrics you created"), 403
    if submission.status == "approved":
        # Terminal on purpose: the grade has been released to the student, so it
        # shouldn't change silently underneath them.
        return jsonify(detail="This submission has already been approved"), 409

    criteria = list(submission.rubric.criteria)
    grades_by_criterion = {gr.criterion_id: gr for gr in submission.grades}

    body = request.get_json(silent=True) or {}
    raw_scores = body.get("criterion_scores")

    if raw_scores is None:
        # Approve as-is -- only possible where the engine actually produced a score.
        ungraded = [
            c.name for c in criteria
            if c.id not in grades_by_criterion or grades_by_criterion[c.id].ai_score is None
        ]
        if ungraded:
            return jsonify(detail=(f"No AI score to approve for: {', '.join(ungraded)}. "
                                   "Supply criterion_scores to grade by hand.")), 422
        # Scores are unchanged, so the AI's wording still matches them exactly.
        decisions = {
            c.id: (float(grades_by_criterion[c.id].ai_score), grades_by_criterion[c.id].ai_feedback, True)
            for c in criteria
        }
        summary = submission.ai_summary
    else:
        parsed, error = _parse_review_scores(raw_scores, criteria)
        if error:
            return jsonify(detail=error), 422

        criteria_by_id = {c.id: c for c in criteria}
        decisions = {}
        for criterion_id, (score, teacher_feedback) in parsed.items():
            existing = grades_by_criterion.get(criterion_id)
            ai_score = float(existing.ai_score) if existing is not None and existing.ai_score is not None else None
            accepted = None if ai_score is None else (score == ai_score)
            # Fall back to freshly generated wording rather than copying the AI's:
            # if the teacher moved the score, the AI's sentence now describes a
            # number that isn't on the page. Where the score is unchanged this
            # regenerates the identical text anyway.
            text = teacher_feedback if teacher_feedback is not None else \
                feedback_templates.for_criterion(criteria_by_id[criterion_id], score)
            decisions[criterion_id] = (score, text, accepted)

        teacher_summary = body.get("summary")
        if teacher_summary is not None and not isinstance(teacher_summary, str):
            return jsonify(detail="summary must be a string"), 422
        summary = teacher_summary if teacher_summary is not None else \
            feedback_templates.summary(criteria, {cid: score for cid, (score, _) in parsed.items()})

    approved_at = datetime.now(timezone.utc)
    for criterion in criteria:
        score, text, accepted = decisions[criterion.id]
        grade = grades_by_criterion.get(criterion.id)
        if grade is None:
            # A grading_failed submission never got Grade rows -- create them now.
            grade = Grade(submission_id=submission.id, criterion_id=criterion.id)
            db.session.add(grade)
        grade.final_score = score
        grade.final_feedback = text
        grade.ai_accepted = accepted
        grade.approved_by = teacher_id
        grade.approved_at = approved_at

    submission.final_summary = summary
    submission.status = "approved"
    db.session.commit()
    return jsonify(_submission_to_dict(submission, for_teacher=True)), 200


@submissions_bp.put("/<int:submission_id>/revise")
@require_role("teacher")
def revise_submission(submission_id):
    """Correct a grade that has already been released to the student.

    Approval used to be terminal, so a mistake stood for good. This changes
    the live scores and writes a GradeRevision row per criterion that actually
    moved, recording what it was, what it became, who changed it and when.

    Validation is the override path's, unchanged: every criterion scored, each
    within its own maximum. Only an approved submission can be revised -- one
    that has not been released yet is still the review endpoint's business.
    """
    submission = db.session.get(Submission, submission_id)
    if not submission:
        return jsonify(detail="Submission not found"), 404

    teacher_id = int(g.current_user["sub"])
    if submission.rubric.created_by != teacher_id:
        return jsonify(detail="You can only revise submissions for rubrics you created"), 403
    if submission.status != "approved":
        return jsonify(detail="Only a released grade can be revised; review this submission instead"), 409

    criteria = list(submission.rubric.criteria)
    body = request.get_json(silent=True) or {}
    parsed, error = _parse_review_scores(body.get("criterion_scores"), criteria)
    if error:
        return jsonify(detail=error), 422

    criteria_by_id = {c.id: c for c in criteria}
    grades_by_criterion = {gr.criterion_id: gr for gr in submission.grades}
    revised_at = datetime.now(timezone.utc)
    changed = 0

    for criterion in criteria:
        score, teacher_feedback = parsed[criterion.id]
        grade = grades_by_criterion.get(criterion.id)
        if grade is None:  # defensive: an approved submission always has grades
            grade = Grade(submission_id=submission.id, criterion_id=criterion.id)
            db.session.add(grade)
            db.session.flush()

        text = teacher_feedback if teacher_feedback is not None else \
            feedback_templates.for_criterion(criteria_by_id[criterion.id], score)

        old_score = float(grade.final_score) if grade.final_score is not None else None
        if old_score == score and grade.final_feedback == text:
            continue  # nothing moved, so nothing to record

        # Written before the live grade changes: a failure part-way leaves a
        # record that something was attempted rather than a silently altered
        # score with no trace.
        db.session.add(GradeRevision(
            submission_id=submission.id,
            criterion_id=criterion.id,
            old_final_score=grade.final_score,
            old_final_feedback=grade.final_feedback,
            new_final_score=score,
            new_final_feedback=text,
            revised_by=teacher_id,
            revised_at=revised_at,
        ))
        grade.final_score = score
        grade.final_feedback = text
        grade.approved_by = teacher_id
        grade.approved_at = revised_at
        # The AI's suggestion was accepted or not at approval time; a later
        # correction by hand is the teacher's own, so that flag no longer
        # describes this score.
        grade.ai_accepted = None
        changed += 1

    teacher_summary = body.get("summary")
    if teacher_summary is not None and not isinstance(teacher_summary, str):
        return jsonify(detail="summary must be a string"), 422
    submission.final_summary = teacher_summary if teacher_summary is not None else \
        feedback_templates.summary(criteria, {cid: score for cid, (score, _) in parsed.items()})

    db.session.commit()
    body = _submission_to_dict(submission, for_teacher=True)
    body["revised_criteria"] = changed
    return jsonify(body), 200


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
