"""Rubric deadlines: parsing what a teacher typed, and deciding whether a
submission is too late.

A rubric with no due date never blocks anything -- that is the default and
what every rubric created before deadlines were enforced keeps.

The cutoff is the instant stored on the rubric. A submission arriving at
exactly the due time is accepted; one arriving after it is refused. The
frontend mirrors the same rule so the student sees it before pressing the
button, but the server is the authority: a clock that is wrong, or a page left
open across the deadline, must not let late work through.
"""

from datetime import datetime, timezone


def parse_due_date(value):
    """Reads a due date from a request body.

    Returns (datetime_or_None, error). Blank and null both mean "no deadline",
    which is how a teacher clears one. Naive datetimes are read as UTC rather
    than rejected: the browser sends a local-time string from <input
    type="datetime-local">, and guessing UTC is better than a 500.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, None
    if isinstance(value, datetime):
        parsed = value
    else:
        if not isinstance(value, str):
            return None, "due_date must be an ISO 8601 date-time, or blank for no deadline"
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None, "due_date must be an ISO 8601 date-time, or blank for no deadline"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed, None


def is_past_due(rubric, now=None):
    """True when this rubric's deadline has gone by. False when it has no
    deadline at all."""
    if rubric.due_date is None:
        return False
    due = rubric.due_date
    if due.tzinfo is None:  # a row written before the column was tz-aware
        due = due.replace(tzinfo=timezone.utc)
    return (now or datetime.now(timezone.utc)) > due


def check_deadline(rubric, now=None):
    """Returns a message if the rubric is closed to new work, else None."""
    if not is_past_due(rubric, now):
        return None
    return ("The deadline for this assignment has passed, so it is no longer accepting "
            "submissions. Speak to your teacher if you think this is wrong.")
