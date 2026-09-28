"""Scores a held-out, team-labeled evaluation set against the models currently
saved in ml_models/, without touching training at all.

Usage (from backend/, with the venv active):
    ./venv/bin/python scripts/evaluate_holdout.py [HOLDOUT.json]
        [--label TEXT] [--json-out PATH] [--compare BEFORE.json] [--bootstrap N]

HOLDOUT.json defaults to training_data/holdout_team_scored.json. It uses the
same {"rubric", "criterion", "text", "score"} schema as sample_answers.json,
plus an "essay_id" so rows group back into essays. This script only ever loads
models with model_store.load_model(); it never calls train_grader.py, never
calls save_model(), and never writes to sample_answers.json. Re-running it
gives identical numbers until the models are retrained.

Per criterion it reports MAE, within-1, exact match and quadratic weighted
kappa (QWK), the guess-the-mean baseline, and bootstrap 95% intervals: essays
are resampled with replacement (default 10,000 times, fixed seed) and the
2.5th/97.5th percentiles are taken. With 12 essays those intervals are wide,
and that width is the honest answer to "how much does this number mean".

--json-out saves the per-essay predictions and metrics. --compare BEFORE.json
then compares this run with a saved one on the same essays, using a *paired*
bootstrap (the same resampled essays for both runs), which is the right way to
put an interval on "did retraining help".

Leakage guard: before scoring, every holdout row is checked against every
training row for the same criterion (see leakage_guard.py). A duplicate stops
the run. Weaker overlaps, such as the same real statistic reused in a new
argument, are printed and saved as a disclosure.

Scoring mirrors app/grading/engine.py grade_text_submission exactly -- same
featurisation, same round-then-clip -- so this reports what the app would have
scored, not an idealised version of the model.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import cohen_kappa_score

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from app import create_app
from app.grading.engine import build_matrix
from app.grading.model_store import load_model, model_path
from app.models import Rubric

import leakage_guard  # sibling script; scripts/ is on sys.path when this is run directly

TRAIN_PATH = os.path.join(BACKEND_DIR, "training_data", "sample_answers.json")
DEFAULT_HOLDOUT_PATH = os.path.join(BACKEND_DIR, "training_data", "holdout_team_scored.json")

BOOTSTRAP_SEED = 0

# The essay-length grouping checked for the first holdout file, by essay_id.
# It is specific to that file's essays, so it only runs for that file.
LENGTH_ANALYSIS = {
    "holdout_team_scored.json": {
        "long": {1, 2, 6, 8, 10, 12},
        "short": {3, 4, 5, 7, 9, 11},
        "known_weakness": {4, 11},  # facts stated without being tied to the argument
    }
}


def load_json(path, purpose):
    if not os.path.exists(path):
        raise SystemExit(f"{path} does not exist -- nothing to {purpose}.")
    with open(path) as f:
        return json.load(f)


def file_md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BACKEND_DIR, text=True).strip()
    except Exception:
        return None


def resolve_criterion(rubric, name):
    criterion = next((c for c in rubric.criteria if c.name == name), None)
    if not criterion:
        raise SystemExit(f"Rubric {rubric.title!r} has no criterion named {name!r}.")
    return criterion


def predict_for(criterion, texts):
    payload = load_model(criterion.id)
    if payload is None:
        raise SystemExit(f"No trained model for criterion {criterion.name!r} -- run scripts/train_grader.py first.")
    raw = np.asarray(payload["model"].predict(build_matrix(texts)), dtype=float)
    max_points = payload["max_points"]
    pred = np.clip(np.round(raw), 0, max_points)
    pred = np.where(pred == 0, 0.0, pred)  # drop the sign bit on -0.0 so it doesn't print as "-0"
    return raw, pred, max_points, payload


# ---------------------------------------------------------------- metrics

def qwk(y_true, y_pred, max_points):
    """Quadratic weighted kappa in plain numpy, so it is cheap enough to
    bootstrap. NaN when there is no variation to measure agreement against."""
    k = int(max_points) + 1
    t, p = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    observed = np.bincount(t * k + p, minlength=k * k).reshape(k, k).astype(float)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    i, j = np.indices((k, k))
    weights = (i - j) ** 2
    denominator = (weights * expected).sum()
    if denominator == 0:
        return float("nan")
    return float(1.0 - (weights * observed).sum() / denominator)


def metrics_for(y_true, y_pred, max_points):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    labels = list(range(0, int(max_points) + 1))
    sk_qwk = float(cohen_kappa_score(y_true.astype(int), y_pred.astype(int), weights="quadratic", labels=labels))
    own_qwk = qwk(y_true, y_pred, max_points)
    # The bootstrap uses the numpy version; make sure it is the same statistic.
    assert np.isnan(own_qwk) or abs(own_qwk - sk_qwk) < 1e-9, (own_qwk, sk_qwk)
    return {
        "mae": float(np.abs(y_pred - y_true).mean()),
        "within_one": float(np.mean(np.abs(y_pred - y_true) <= 1)),
        "exact": float(np.mean(y_pred == y_true)),
        "qwk": sk_qwk,
    }


def baseline_for(y_true, max_points):
    y_true = np.asarray(y_true, dtype=float)
    constant = float(np.clip(round(y_true.mean()), 0, max_points))
    return {**metrics_for(y_true, np.full_like(y_true, constant), max_points), "constant": constant}


def bootstrap_indices(n, n_boot):
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(n_boot, n))


def metric_samples(y_true, y_pred, max_points, idx):
    """Each metric recomputed on every resample of essays (rows of idx)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_pred - y_true)
    return {
        "mae": abs_err[idx].mean(axis=1),
        "within_one": (abs_err <= 1)[idx].mean(axis=1),
        "exact": (y_pred == y_true)[idx].mean(axis=1),
        "qwk": np.array([qwk(y_true[r], y_pred[r], max_points) for r in idx]),
    }


def baseline_mae_samples(y_true, max_points, idx):
    """The baseline is a procedure (predict the rounded mean), so it is redone
    inside each resample rather than fixed from the full set."""
    y_true = np.asarray(y_true, dtype=float)
    constants = np.clip(np.round(y_true[idx].mean(axis=1)), 0, max_points)
    return np.abs(y_true[idx] - constants[:, None]).mean(axis=1)


def interval(samples):
    valid = samples[~np.isnan(samples)]
    if len(valid) == 0:
        return [None, None]
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return [float(lo), float(hi)]


# ---------------------------------------------------------------- printing

def pct(x):
    return f"{x:.0%}"


def fmt(value, ci, as_pct=False):
    show = pct if as_pct else (lambda v: f"{v:.2f}")
    if ci is None or ci[0] is None:
        return show(value)
    return f"{show(value)} [{show(ci[0])}, {show(ci[1])}]"


def signed(value, as_pct=False):
    return f"{value * 100:+.0f}pp" if as_pct else f"{value:+.2f}"


def print_essay_table(name, res):
    print(f"\n  {name} -- true vs. predicted, sorted by |error| (worst first):")
    print(f"    {'essay':<6}{'words':<7}{'true':<6}{'pred':<6}{'error':<7}")
    order = sorted(range(len(res["essay_ids"])), key=lambda i: -abs(res["pred"][i] - res["true"][i]))
    for i in order:
        error = res["pred"][i] - res["true"][i]
        sign = "+" if error > 0 else ""
        print(f"    {res['essay_ids'][i]:<6}{res['words'][i]:<7}{res['true'][i]:<6.0f}{res['pred'][i]:<6.0f}{sign}{error:<6.0f}")


def print_length_breakdown(name, res, groups):
    ids = np.array(res["essay_ids"])
    abs_err = np.abs(res["pred"] - res["true"])
    long_mask = np.isin(ids, list(groups["long"]))
    short_mask = np.isin(ids, list(groups["short"]))
    long_mae, short_mae = abs_err[long_mask].mean(), abs_err[short_mask].mean()

    print(f"\n  {name} -- error by essay length:")
    print(f"    paragraph-length essays {sorted(groups['long'])}: MAE {long_mae:.2f}")
    print(f"    short essays            {sorted(groups['short'])}: MAE {short_mae:.2f}")
    for essay_id in sorted(groups["known_weakness"]):
        if essay_id in ids:
            i = int(np.where(ids == essay_id)[0][0])
            print(f"    essay {essay_id} (known weakness: facts stated without being tied to the argument): "
                  f"true {res['true'][i]:.0f}, predicted {res['pred'][i]:.0f}, error {res['pred'][i] - res['true'][i]:+.0f}")
    gap = short_mae - long_mae
    if abs(gap) < 0.5:
        verdict = "no meaningful length gap -- error is roughly flat across short and paragraph-length essays"
    elif gap > 0:
        verdict = "short essays have the larger error here"
    else:
        verdict = "paragraph-length essays have the larger error here"
    print(f"    verdict: {verdict}")


# ---------------------------------------------------------------- comparison

def compare_runs(current, before, idx):
    """Paired comparison of two saved runs on the same essays. The same
    resampled essays (idx) are used for both, so the interval on the change
    reflects only what changed between the runs."""
    comparison = {}
    for name, res in current["results"].items():
        prior = before["criteria"].get(name)
        if prior is None:
            continue
        prior_by_essay = {e["essay_id"]: e for e in prior["essays"]}
        if set(prior_by_essay) != set(res["essay_ids"]):
            raise SystemExit(f"{name}: --compare file scores different essays than this run.")
        before_true = np.array([prior_by_essay[i]["true"] for i in res["essay_ids"]], dtype=float)
        if not np.array_equal(before_true, res["true"]):
            raise SystemExit(f"{name}: --compare file has different true labels for the same essay ids.")
        before_pred = np.array([prior_by_essay[i]["pred"] for i in res["essay_ids"]], dtype=float)

        max_points = res["max_points"]
        b_point, a_point = metrics_for(res["true"], before_pred, max_points), metrics_for(res["true"], res["pred"], max_points)
        b_samples = metric_samples(res["true"], before_pred, max_points, idx)
        a_samples = res["samples"]

        rows = {}
        for key in ("mae", "within_one", "exact", "qwk"):
            diff = a_samples[key] - b_samples[key]
            rows[key] = {"before": b_point[key], "after": a_point[key], "change": a_point[key] - b_point[key],
                         "change_ci": interval(diff), "before_ci": interval(b_samples[key])}

        per_essay = []
        for i, essay_id in enumerate(res["essay_ids"]):
            per_essay.append({
                "essay_id": essay_id, "true": float(res["true"][i]),
                "before_pred": float(before_pred[i]), "after_pred": float(res["pred"][i]),
                "before_abs_error": float(abs(before_pred[i] - res["true"][i])),
                "after_abs_error": float(abs(res["pred"][i] - res["true"][i])),
            })
        comparison[name] = {"metrics": rows, "per_essay": per_essay}
    return comparison


def print_comparison(comparison, before_label):
    for name, block in comparison.items():
        print(f"=== {name}: BEFORE -> AFTER (paired bootstrap over the same essays; before = {before_label}) ===")
        for key, title, as_pct in (("mae", "MAE", False), ("within_one", "within-1", True),
                                   ("exact", "exact", True), ("qwk", "QWK", False)):
            r = block["metrics"][key]
            show = pct if as_pct else (lambda v: f"{v:.2f}")
            lo, hi = r["change_ci"]
            span = "n/a" if lo is None else f"[{signed(lo, as_pct)}, {signed(hi, as_pct)}]"
            print(f"    {title:<9}{show(r['before'])} -> {show(r['after'])}   change {signed(r['change'], as_pct)}  95% CI {span}")
        print(f"\n  {name} -- per-essay change, largest movement first (|error| before -> after):")
        print(f"    {'essay':<6}{'true':<6}{'before':<8}{'after':<7}{'|err| before':<14}{'|err| after':<13}{'change'}")
        order = sorted(block["per_essay"], key=lambda e: -abs(e["after_abs_error"] - e["before_abs_error"]))
        for e in order:
            delta = e["after_abs_error"] - e["before_abs_error"]
            note = "better" if delta < 0 else ("worse" if delta > 0 else "same")
            print(f"    {e['essay_id']:<6}{e['true']:<6.0f}{e['before_pred']:<8.0f}{e['after_pred']:<7.0f}"
                  f"{e['before_abs_error']:<14.0f}{e['after_abs_error']:<13.0f}{delta:+.0f} {note}")
        print()


# ---------------------------------------------------------------- main

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved models on a held-out set.")
    parser.add_argument("holdout", nargs="?", default=DEFAULT_HOLDOUT_PATH, help="holdout JSON file")
    parser.add_argument("--label", default=None, help="banner text, saved in the JSON (e.g. 'development set')")
    parser.add_argument("--json-out", default=None, help="write per-essay predictions and metrics here")
    parser.add_argument("--compare", default=None, help="a saved --json-out file to compare against, paired")
    parser.add_argument("--bootstrap", type=int, default=10000, help="bootstrap resamples (default 10000)")
    return parser.parse_args()


def main():
    args = parse_args()
    holdout_path = os.path.abspath(args.holdout)
    holdout_name = os.path.basename(holdout_path)

    app = create_app()
    with app.app_context():
        train_rows = load_json(TRAIN_PATH, "compare against")
        holdout_rows = load_json(holdout_path, "evaluate")

        print("=" * 78)
        print(f"Holdout: {holdout_name}   ({len(holdout_rows)} rows)")
        if args.label:
            print(f"Label:   {args.label}")
        print(f"Training data: {os.path.basename(TRAIN_PATH)}   ({len(train_rows)} rows)")
        print("=" * 78 + "\n")

        duplicates, notable = leakage_guard.find_matches(holdout_rows, train_rows, same_criterion=True)
        if duplicates:
            print("LEAKAGE GUARD: STOPPING -- these holdout rows look like duplicates of training data:\n")
            leakage_guard.print_matches(duplicates)
            raise SystemExit("Fix or remove these rows before evaluating: a holdout essay that duplicates a "
                             "training example isn't held out.")
        if notable:
            print("Leakage guard: no duplicates. These holdout rows share a specific real-world fact or")
            print("wording with a training example (not disqualifying, but worth knowing about):\n")
            leakage_guard.print_matches(notable)
        else:
            print("Leakage guard: no duplicate or near-duplicate rows found.\n")

        rubric = Rubric.query.filter_by(title=holdout_rows[0]["rubric"]).first()
        if not rubric:
            raise SystemExit(f"No rubric titled {holdout_rows[0]['rubric']!r} exists.")

        by_criterion = defaultdict(list)
        for row in holdout_rows:
            by_criterion[row["criterion"]].append(row)

        train_counts = defaultdict(int)
        for row in train_rows:
            train_counts[row["criterion"]] += 1

        idx = bootstrap_indices(len(next(iter(by_criterion.values()))), args.bootstrap)
        results, model_info = {}, {}

        for name, rows in by_criterion.items():
            criterion = resolve_criterion(rubric, name)
            raw, pred, max_points, payload = predict_for(criterion, [r["text"] for r in rows])
            true = np.array([r["score"] for r in rows], dtype=float)

            trained_on = payload.get("n_examples")
            model_info[name] = {"n_examples": trained_on, "md5": file_md5(model_path(criterion.id))}
            if trained_on != train_counts[name]:
                print(f"WARNING: the {name} model was trained on {trained_on} examples but "
                      f"{os.path.basename(TRAIN_PATH)} now has {train_counts[name]} -- retrain before trusting this.\n")

            model = metrics_for(true, pred, max_points)
            baseline = baseline_for(true, max_points)
            samples = metric_samples(true, pred, max_points, idx)
            base_samples = baseline_mae_samples(true, max_points, idx)
            ci = {
                "model": {k: interval(v) for k, v in samples.items()},
                "baseline_mae": interval(base_samples),
                "mae_improvement_over_baseline": interval(base_samples - samples["mae"]),
            }
            results[name] = {
                "max_points": max_points, "essay_ids": [r["essay_id"] for r in rows], "true": true, "raw": raw,
                "pred": pred, "words": [len(r["text"].split()) for r in rows],
                "model": model, "baseline": baseline, "ci": ci, "samples": samples,
            }

        print(f"Models were trained on: " + ", ".join(f"{n} n={i['n_examples']}" for n, i in model_info.items()))
        print(f"Bootstrap: {args.bootstrap:,} resamples of essays, seed {BOOTSTRAP_SEED}, percentile 95% intervals\n")

        for name, res in results.items():
            m, b, ci = res["model"], res["baseline"], res["ci"]
            n = len(res["true"])
            print(f"=== {name} (0-{res['max_points']:g}), n={n} ===")
            print(f"    model              MAE {fmt(m['mae'], ci['model']['mae'])}   "
                  f"within-1 {fmt(m['within_one'], ci['model']['within_one'], True)}   "
                  f"exact {fmt(m['exact'], ci['model']['exact'], True)}   QWK {fmt(m['qwk'], ci['model']['qwk'])}")
            baseline_label = f"baseline ({b['constant']:g})"
            print(f"    {baseline_label:<19}"
                  f"MAE {fmt(b['mae'], ci['baseline_mae'])}   within-1 {pct(b['within_one'])}   "
                  f"exact {pct(b['exact'])}   QWK {b['qwk']:.2f} (constant predictions carry no agreement)")
            print(f"    MAE improvement over baseline: {fmt(b['mae'] - m['mae'], ci['mae_improvement_over_baseline'])}")
            print_essay_table(name, res)
            if holdout_name in LENGTH_ANALYSIS:
                print_length_breakdown(name, res, LENGTH_ANALYSIS[holdout_name])
            print()

        output = {
            "label": args.label,
            "holdout_file": holdout_name,
            "training_file": os.path.basename(TRAIN_PATH),
            "training_rows": len(train_rows),
            "models": model_info,
            "git_head": git_head(),
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bootstrap": {"resamples": args.bootstrap, "seed": BOOTSTRAP_SEED},
            "leakage": {
                "duplicates": 0,
                "notable_overlaps": [
                    {"essay_id": m["candidate"].get("essay_id"), "criterion": m["candidate"]["criterion"],
                     "cosine": round(m["cosine"], 3), "closest_training_text": m["reference"]["text"][:140]}
                    for m in sorted(notable, key=lambda m: -m["cosine"])
                ],
            },
            "criteria": {
                name: {
                    "max_points": res["max_points"], "n": len(res["true"]),
                    "model": res["model"], "baseline": res["baseline"], "ci": res["ci"],
                    "essays": [
                        {"essay_id": res["essay_ids"][i], "true": float(res["true"][i]), "raw": round(float(res["raw"][i]), 3),
                         "pred": float(res["pred"][i]), "word_count": res["words"][i]}
                        for i in range(len(res["true"]))
                    ],
                }
                for name, res in results.items()
            },
        }

        if args.compare:
            before = load_json(os.path.abspath(args.compare), "compare against")
            comparison = compare_runs({"results": results}, before, idx)
            print_comparison(comparison, before.get("label") or before.get("holdout_file"))
            output["compare"] = {"against": os.path.basename(args.compare), "criteria": comparison}

        if args.json_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
            with open(args.json_out, "w") as f:
                json.dump(output, f, indent=2)
                f.write("\n")
            print(f"Saved {args.json_out}\n")

        print("Note: with about a dozen essays the intervals are wide and QWK is indicative at best -- read the "
              "direction and rough size, not the second decimal.")


if __name__ == "__main__":
    main()
