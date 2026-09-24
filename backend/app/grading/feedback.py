"""Template-based feedback: turns a score into plain-English commentary.

Deliberately not model-generated. The scores come from a small regression
trained on a few dozen examples, and writing fluent prose on top of an
uncertain number would make the grade sound far more authoritative than it is.
These templates say exactly what the score means and nothing more.

The same generator is used for AI-suggested scores and for scores a teacher
types by hand, so the wording can never contradict the number sitting next to
it -- a teacher who raises Evidence from 2 to 9 gets feedback describing a 9.
"""

# (minimum score/max_points ratio, label, guidance) -- highest band first.
BANDS = (
    (0.90, "excellent", "This criterion is handled thoroughly and specifically."),
    (0.75, "solid", "This criterion is clearly addressed, with room to sharpen the weaker parts."),
    (0.50, "partially meets expectations",
     "Some of what this criterion asks for is present, but it is uneven and needs development."),
    (0.00, "needs significant improvement",
     "The response does not yet address what this criterion asks for; start your revision here."),
)

# Below this, the feedback also quotes what the criterion asks for -- above it,
# that reads as nagging.
QUOTE_CRITERION_BELOW = 0.75


def _ratio(score, max_points):
    return (float(score) / float(max_points)) if max_points else 0.0


def band_for(score, max_points):
    """Returns (label, guidance) for a score on a 0..max_points scale."""
    ratio = _ratio(score, max_points)
    for minimum, label, guidance in BANDS:
        if ratio >= minimum:
            return label, guidance
    return BANDS[-1][1], BANDS[-1][2]


def for_criterion(criterion, score):
    """One line of commentary naming the criterion, its score, and what to do next."""
    score = float(score)
    max_points = float(criterion.max_points)
    label, guidance = band_for(score, max_points)

    text = f"{criterion.name}: {score:g}/{max_points:g} -- {label}. {guidance}"

    description = (criterion.description or "").strip()
    if description and _ratio(score, max_points) < QUOTE_CRITERION_BELOW:
        text += f" What this criterion asks for: {description}"
    return text


def summary(criteria, scores):
    """Overall line across every criterion, calling out the strongest and weakest.

    criteria: RubricCriterion rows. scores: {criterion_id: score}.
    """
    scored = [(c, float(scores[c.id]), float(c.max_points)) for c in criteria if c.id in scores]
    if not scored:
        return ""

    total = sum(score for _, score, _ in scored)
    possible = sum(max_points for _, _, max_points in scored)
    label, _ = band_for(total, possible)
    parts = [f"Overall {total:g}/{possible:g} -- {label}."]

    if len(scored) > 1:
        by_ratio = sorted(scored, key=lambda row: _ratio(row[1], row[2]))
        weakest, strongest = by_ratio[0], by_ratio[-1]
        # Only worth naming when they actually differ -- otherwise "strongest"
        # and "weakest" are the same thing and the line reads as nonsense.
        if _ratio(weakest[1], weakest[2]) < _ratio(strongest[1], strongest[2]):
            parts.append(f"Strongest: {strongest[0].name} ({strongest[1]:g}/{strongest[2]:g}).")
            parts.append(f"Weakest: {weakest[0].name} ({weakest[1]:g}/{weakest[2]:g})"
                         " -- focus your revision there first.")

    return " ".join(parts)


# Enough of each output to see what went wrong, without pasting a whole file
# into a grade.
OUTPUT_EXCERPT = 200


def _excerpt(text):
    text = (text or "").strip()
    if not text:
        return "(nothing)"
    if len(text) > OUTPUT_EXCERPT:
        text = f"{text[:OUTPUT_EXCERPT]}…"
    return repr(text) if "\n" not in text else f"\n      {text.replace(chr(10), chr(10) + '      ')}"


def for_code_criterion(criterion, results):
    """Which of this criterion's tests passed, and what went wrong with the
    rest. Wrong answers show expected against actual; a crash or a timeout
    says so instead, because there is no meaningful output to compare."""
    passed = sum(1 for result in results if result["passed"])
    total = len(results)

    lines = [f"{criterion.name}: passed {passed} of {total} test{'' if total == 1 else 's'}."]
    for number, result in enumerate(results, start=1):
        if result["passed"]:
            continue
        lines.append(f"  Test {number}: {result['detail']}")
        if result["outcome"] == "ok":  # ran fine, printed the wrong thing
            lines.append(f"    expected: {_excerpt(result['expected'])}")
            lines.append(f"    actual:   {_excerpt(result['actual'])}")
    return "\n".join(lines)
