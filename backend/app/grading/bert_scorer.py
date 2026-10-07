"""The fine-tuned DistilBERT scorer that grades text submissions in
production, replacing the per-criterion Ridge models for the criteria it
covers.

One shared model for every criterion, reading a sentence pair:

    [CLS] the student's essay [SEP] Thesis: Does the essay take a clear... [SEP]

It predicts a single number on 0-1, which is multiplied by that criterion's
own max_points, rounded and clamped. Provenance, the measurements behind the
swap, and the honest limits are in ml_models/bert_rubric_scorer/PROVENANCE.md.

Two things about this module are deliberate and worth reading before changing
it.

**Criteria are looked up by name, not by database row id.** The Ridge models
were keyed by `criterion_<row id>.joblib`, which meant re-seeding the dev
database silently orphaned every trained model -- new rows, new ids, no
matching file, everything falls to grading_failed. That was survivable when a
retrain took seconds. This model needs a GPU and ~15-20 minutes on Colab, so
the same accident is now expensive, and the fix is to key on something that
survives a reseed: the criterion's name.

**CRITERION_DESCRIPTIONS is not an extension point.** It looks like one. It
is not. The model was measured, on these weights, not to read the criterion
segment at all: Thesis and Evidence predictions correlate at 0.9997, and a
nonsense criterion correlates at 0.9986. It computes general essay quality and
rescales it. That works for these two criteria because their human labels are
themselves correlated at ~0.74; it would not work for a criterion that
diverges from general quality, and rubric_conditioning_results.md has the
worked example -- a confident one-sided essay with no counterargument scored
4/5 on "Counterargument" by this same architecture.

A criterion that is not in this table scores nothing and the submission falls
to grading_failed for a teacher to grade by hand. That is the correct
behaviour. Adding an entry here means first validating it the way Thesis and
Evidence were validated: a locked holdout, scored once, interval reported.
"""

import os

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "ml_models", "bert_rubric_scorer",
)

# Must stay byte-identical to the descriptions the model was trained on
# (scripts/build_rubric_conditioning_data.py, pre-declared in
# training_data/results/own_data_finetune_results.md). A reworded description
# is a different input than the one that was measured.
#
# Keyed by criterion NAME so a database reseed can't orphan the model. See the
# warning in the module docstring before adding to this.
CRITERION_DESCRIPTIONS = {
    "Thesis": "Does the essay take a clear, specific, arguable position?",
    "Evidence": "Does the essay support its position with specific, relevant "
                "evidence tied to the argument?",
}

# The window the model was trained with. Changing it changes the input.
MAX_LENGTH = 384

_model = None
_tokenizer = None


def describes(criterion_name):
    """The description this model was trained to read for a criterion, or None
    if it has never been validated on one by that name."""
    return CRITERION_DESCRIPTIONS.get(criterion_name)


def is_available():
    """Whether the weights are actually on disk. They are not in git -- a fresh
    clone has to place them by hand (see PROVENANCE.md), and until it does,
    text grading falls back to grading_failed rather than guessing."""
    return os.path.exists(os.path.join(MODEL_DIR, "config.json"))


def load():
    """The model and tokenizer, loaded once per process and kept.

    Same shape as embedder.get_embedder(): lazy, so importing this module
    costs nothing and the test suite doesn't pay 269 MB it isn't using, then
    cached for the life of the process so a request never pays the load twice.
    """
    global _model, _tokenizer
    if _model is None:
        if not is_available():
            raise FileNotFoundError(
                f"No fine-tuned scorer at {MODEL_DIR}. The weights are not in git: "
                "place them from a Colab run's best_model/ (see PROVENANCE.md).")
        import torch  # noqa: F401  (imported for its side effect on thread config below)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()  # no dropout at inference; the scores must be deterministic
        _tokenizer, _model = tokenizer, model
    return _model, _tokenizer


def score(text, criteria):
    """Scores one submission against several criteria in a single batch.

    criteria: an iterable of objects with .name and .max_points (RubricCriterion
    rows in production). Every one of them must have a description here --
    callers check with describes() first and fall back to grading_failed
    otherwise, so this raises rather than inventing a score.

    Returns {criterion_name: float score}, rounded and clamped to each
    criterion's own 0..max_points range, exactly as the experiment scored the
    holdouts.
    """
    import torch

    criteria = list(criteria)
    unknown = [c.name for c in criteria if describes(c.name) is None]
    if unknown:
        raise KeyError(f"No validated description for: {', '.join(sorted(unknown))}")
    if not criteria:
        return {}

    model, tokenizer = load()
    # One forward pass for the whole rubric: the same essay against each
    # criterion's segment. Truncation is "only_first" so a long essay is cut
    # and the criterion never is.
    encoded = tokenizer(
        [text] * len(criteria),
        [f"{c.name}: {describes(c.name)}" for c in criteria],
        truncation="only_first", max_length=MAX_LENGTH, padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(input_ids=encoded["input_ids"],
                       attention_mask=encoded["attention_mask"]).logits
    # (n_criteria, 1) -> (n_criteria,); always a list, even for a single criterion
    units = logits.squeeze(-1).float().tolist()

    scores = {}
    for criterion, unit in zip(criteria, units):
        max_points = float(criterion.max_points)
        raw = unit * max_points
        scores[criterion.name] = float(min(max(round(raw), 0.0), max_points))
    return scores
