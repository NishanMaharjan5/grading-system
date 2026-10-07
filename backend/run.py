import os

from app import create_app

app = create_app()


def _warm_up_text_grader():
    """Load the text grader before serving rather than on the first
    submission: cold it costs ~3.6s, warm ~140ms, and without this it is the
    first student after every restart who waits.

    Skipped in the reloader's parent process -- with debug=True this module
    runs twice, and only the child actually serves requests, so warming both
    would load 269 MB for nothing.

    Absent weights are not an error: grading falls back to grading_failed and
    the app still runs.
    """
    from app.grading import bert_scorer

    reloader_parent = app.debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true"
    if reloader_parent:
        return
    if not bert_scorer.is_available():
        print(f"No text grader at {bert_scorer.MODEL_DIR}; text submissions will need manual "
              "grading (see ml_models/bert_rubric_scorer/PROVENANCE.md).")
        return
    print("Loading the text grader...")
    bert_scorer.warm_up()
    print("Text grader ready.")


if __name__ == "__main__":
    # Schema is managed by Alembic, not create_all(): run `flask db upgrade`
    # (see migrations/README) before starting for the first time or after
    # pulling a change that touches the models.
    app.debug = True
    _warm_up_text_grader()
    app.run(host="0.0.0.0", port=8000, debug=True)
