"""Loads/saves the per-criterion models produced by scripts/train_grader.py.

One fitted pipeline per rubric criterion, keyed by criterion id, cached in
memory after first load so a request doesn't hit the filesystem every time.
Not committed to git (see .gitignore) -- rerun the training script after
cloning, or every rubric will fall back to grading_failed.
"""

import os

import joblib

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ml_models")

_cache = {}


def model_path(criterion_id):
    return os.path.join(MODEL_DIR, f"criterion_{criterion_id}.joblib")


def save_model(criterion_id, payload):
    """payload: {"model": <fitted sklearn pipeline>, "max_points": float, ...}"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(payload, model_path(criterion_id))
    _cache.pop(criterion_id, None)  # drop any stale in-memory copy so the next load picks up the retrain


def load_model(criterion_id):
    """Returns the saved payload dict, or None if this criterion has never been trained."""
    if criterion_id in _cache:
        return _cache[criterion_id]
    path = model_path(criterion_id)
    if not os.path.exists(path):
        return None
    payload = joblib.load(path)
    _cache[criterion_id] = payload
    return payload
