"""Word counting and word-range checks for text rubrics.

One definition of "a word", used by the rubric route that validates a range,
the submission route that enforces it, and the message a student is shown. The
frontend mirrors `count_words` exactly (frontend/src/grading/wordCount.js); the
server stays the authority, the way RubricForm already double-validates.

Why there is a ceiling at all: the embedder reads at most 256 tokens, roughly
226 words, and silently drops the rest, so a longer essay is graded on a
partial read with no sign that anything was missed. A max_words at or below
GRADER_WORD_LIMIT keeps the grader reading the whole submission. Teachers may
set a higher limit deliberately -- it is their assignment -- but the UI warns
them what it costs.
"""

# all-MiniLM-L6-v2 reads 256 tokens; measured at ~1.13 tokens/word on this
# corpus, which is ~226 words. See the README's limitations section.
GRADER_WORD_LIMIT = 226

# What a new text rubric gets if the teacher doesn't choose: comfortably under
# the limit above, and a realistic length for a short argumentative essay.
DEFAULT_MAX_WORDS = 200


def count_words(text):
    """Whitespace-separated tokens. Deliberately the crudest possible rule:
    a student counting by eye, the box's live counter and the server all have
    to agree, and anything cleverer (hyphens, contractions, numerals) would
    disagree with at least one of them."""
    return len((text or "").split())


def describe_range(min_words, max_words):
    """The range in words, for a message or a form hint. None when unlimited."""
    if min_words is not None and max_words is not None:
        return f"between {min_words} and {max_words} words"
    if max_words is not None:
        return f"at most {max_words} words"
    if min_words is not None:
        return f"at least {min_words} words"
    return None


def check_submission_length(text, rubric):
    """Returns an error message if `text` falls outside the rubric's word
    range, or None if it is acceptable.

    A rubric with no limits, and any code rubric, accepts anything: word counts
    say nothing about a program, and rubrics that predate this feature keep
    their unrestricted behaviour.
    """
    if rubric.type != "text":
        return None
    if rubric.min_words is None and rubric.max_words is None:
        return None

    words = count_words(text)
    if rubric.min_words is not None and words < rubric.min_words:
        return (f"This essay must be {describe_range(rubric.min_words, rubric.max_words)}. "
                f"Yours is {words}, which is too short.")
    if rubric.max_words is not None and words > rubric.max_words:
        return (f"This essay must be {describe_range(rubric.min_words, rubric.max_words)}. "
                f"Yours is {words}, which is too long.")
    return None


def validate_limits(min_words, max_words):
    """Validates the pair a teacher submitted. Returns an error message, or
    None. Mirrors the database's CHECK constraints so the route answers 422
    rather than letting the insert fail with a 500."""
    for label, value in (("min_words", min_words), ("max_words", max_words)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{label} must be a whole number or blank"
        if value < 0:
            return f"{label} cannot be negative"
    if max_words is not None and max_words < 1:
        return "max_words must be at least 1"
    if min_words is not None and max_words is not None and min_words > max_words:
        return f"min_words ({min_words}) cannot be greater than max_words ({max_words})"
    return None
