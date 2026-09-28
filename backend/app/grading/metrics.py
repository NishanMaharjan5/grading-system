"""Scoring metrics shared by the training script and the holdout evaluation,
so both report the same statistic computed the same way.

Kept in the app package rather than in scripts/ for the same reason
features.py is: two copies of a metric drift, and then a "before" number and
an "after" number stop being comparable.
"""

import numpy as np


def quadratic_weighted_kappa(y_true, y_pred, max_points):
    """QWK over the integer score range 0..max_points.

    Chance-corrected agreement that punishes being far off more than being
    slightly off, which suits ordinal rubric scores. Returns NaN when the
    expected-disagreement term is zero -- that happens when predictions are
    constant, and there is then no agreement to measure rather than perfect
    agreement.

    Written in plain numpy (rather than calling sklearn each time) because the
    bootstrap evaluates it tens of thousands of times; evaluate_holdout.py
    asserts it matches sklearn's cohen_kappa_score on every real run.
    """
    k = int(max_points) + 1
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)

    observed = np.bincount(true * k + pred, minlength=k * k).reshape(k, k).astype(float)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()

    i, j = np.indices((k, k))
    weights = (i - j) ** 2

    denominator = (weights * expected).sum()
    if denominator == 0:
        return float("nan")
    return float(1.0 - (weights * observed).sum() / denominator)
