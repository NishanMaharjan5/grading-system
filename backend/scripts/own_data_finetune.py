"""Phase 1: does fine-tuning a BERT-family model on our OWN rubric data beat
the shipped frozen-embedding pipeline, on the shipped pipeline's own locked
test sets?

Every other fine-tuning experiment in this project (asap_finetune.py,
rubric_conditioning.py) trains on public or zero-shot data and asks a
methods question. This one asks the practical question: if we fine-tune on
exactly the ~130 labeled examples we actually have, does it beat what's
running in production right now? A loss or a null result is a legitimate,
expected-possible outcome at this sample size -- not a failure of execution.

Architecture is the rubric-conditioning experiment's rubric-aware variant,
reused as-is: one shared DistilBERT regression head reading a sentence pair
`[CLS] essay [SEP] criterion name: description [SEP]`, target normalised to
score / max_points. That let one model see Thesis and Evidence as related
rather than unrelated labels, which is the only lever a fine-tuned model has
that the shipped per-criterion Ridge models don't.

Runs on a GPU (Colab's free T4), not a laptop: fine-tuning froze an 8 GB
Apple M1 once already. scripts/own_data_finetune_colab.ipynb runs it.

    python scripts/own_data_finetune.py --data-dir DIR --out-dir DIR
    python scripts/own_data_finetune.py --data-dir DIR --out-dir DIR --dry-run
    python scripts/own_data_finetune.py --data-dir DIR --out-dir DIR --smoke

--data-dir is one flat folder (on Google Drive, for Colab) holding:
    rubric_conditioning_train.json   128 Thesis/Evidence triples (Essay 1 + 2)
    holdout2_team_scored.json        clean locked Essay 1 test set (12 essays)
    holdout_essay2_team_scored.json  clean locked Essay 2 test set (8 essays)
    phase1_baseline_holdout2.json    shipped pipeline's fresh predictions on the above
    phase1_baseline_essay2.json      shipped pipeline's fresh predictions on the above
Without it, the repo layout is used.

The design is pre-declared in
training_data/results/own_data_finetune_results.md, committed before this
runs, including the two locked file hashes checked below. Nothing here may
change in response to the test numbers. The order of work protects that:

  1. Pre-flight: both clean holdouts must be the locked files, the shipped
     baseline's recorded essays must match them exactly (same ids, same
     labels), and holdout_team_scored.json (the burned dev set from an
     earlier round) is never loaded by this script at all.
  2. A fixed learning-rate/epoch grid, chosen on a dev slice carved from the
     training data -- never from either locked holdout.
  3. Both locked holdouts scored once, at the end, against the shipped
     pipeline's own fresh predictions (paired bootstrap, same essays).
  4. Overfitting watched explicitly: every epoch's train loss and dev MAE are
     logged, and a model whose dev MAE keeps rising after its best epoch
     while train loss keeps falling is flagged in the output.

Isolation: writes only to --out-dir. No database, no Flask app, and it never
reads or writes ml_models/. The shipped graders are untouched, and this holds
no opinion on whether to integrate -- that's a separate decision after
seeing these results.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BACKEND_DIR, "ml_experiments", "own_data_finetune")

# ---- the pre-declared design; tests pin every value ------------------------
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 384          # essay + criterion description, as a sentence pair
BATCH_SIZE = 8
LEARNING_RATES = (2e-5, 3e-5)
MAX_EPOCHS = 4
SEED = 20261007
DEV_FRACTION = 0.2        # of the training data, stratified by criterion

BOOTSTRAP_N = 10000        # matches evaluate_holdout.py, so CIs are comparable
BOOTSTRAP_SEED = 0         # matches evaluate_holdout.py's fixed seed

CPU_LATENCY_ESSAYS = 16

# Canonical descriptions, identical to build_rubric_conditioning_data.py --
# the conditioning signal a reader of that file already recognises.
CRITERIA = {
    "Thesis": "Does the essay take a clear, specific, arguable position?",
    "Evidence": "Does the essay support its position with specific, relevant "
               "evidence tied to the argument?",
}
MAX_POINTS = {"Thesis": 5, "Evidence": 10}

# Locked at pre-declaration time (before this script ran). A pre-flight
# refuses to run if any of these have since changed.
LOCKED_SHA = {
    "rubric_conditioning_train.json": "c6dccc50f8223d1c",
    "holdout2_team_scored.json": "911274b40afb4518",
    "holdout_essay2_team_scored.json": "8c204d8a45cf184b",
}

# holdout_team_scored.json ("holdout1") was the development set in an earlier
# round -- studied in detail before training data was chosen to fix what it
# exposed, so any improvement on it is not evidence. It is deliberately not
# in LOCKED_SHA and never loaded below.


def log_to(path):
    def log(message=""):
        print(message, flush=True)
        with open(path, "a") as f:
            f.write(message + "\n")
    return log


# ------------------------------------------------------------------ inputs

def resolve_paths(data_dir=None):
    names = {
        "train": "rubric_conditioning_train.json",
        "holdout2": "holdout2_team_scored.json",
        "essay2_holdout": "holdout_essay2_team_scored.json",
        "baseline_holdout2": "phase1_baseline_holdout2.json",
        "baseline_essay2": "phase1_baseline_essay2.json",
    }
    base = data_dir or os.path.join(BACKEND_DIR, "training_data")
    results = os.path.join(BACKEND_DIR, "training_data", "results")
    return {
        "train": os.path.join(base, names["train"]),
        "holdout2": os.path.join(base, names["holdout2"]),
        "essay2_holdout": os.path.join(base, names["essay2_holdout"]),
        "baseline_holdout2": os.path.join(base if data_dir else results, names["baseline_holdout2"]),
        "baseline_essay2": os.path.join(base if data_dir else results, names["baseline_essay2"]),
    }


def load_json(path, what):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- nothing to {what}.")
    with open(path) as f:
        return json.load(f)


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def normalise(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def enrich(rows, source_label):
    """The locked holdout files use sample_answers.json's schema -- rubric,
    criterion, text, score, essay_id -- with no max_points or description.
    This adds the two the model needs, from the same canonical table
    build_rubric_conditioning_data.py used to build the training file, and
    checks every score actually fits its criterion's range."""
    out = []
    for row in rows:
        criterion = row["criterion"]
        if criterion not in CRITERIA:
            raise SystemExit(f"{source_label}: criterion {criterion!r} is outside this experiment "
                             f"(only {sorted(CRITERIA)} are covered).")
        max_points = MAX_POINTS[criterion]
        if not 0 <= row["score"] <= max_points:
            raise SystemExit(f"{source_label}: essay {row.get('essay_id')} scores {row['score']} on "
                             f"{criterion}, outside 0..{max_points}.")
        out.append({**row, "criterion_description": CRITERIA[criterion], "max_points": max_points})
    return out


# ------------------------------------------------------------------ hardware

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def require_gpu(device, allow_no_gpu=False, light=False):
    if device.type == "cuda" or light or allow_no_gpu:
        return
    raise SystemExit(
        f"No CUDA GPU found (device: {device.type}). Fine-tuning froze an 8 GB Apple M1 once "
        "already. Run this on Colab with a T4, or pass --allow-no-gpu to insist.")


def environment(device):
    import transformers
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "device": device.type,
        "cpu_count": os.cpu_count(),
    }
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(0)
        env["cuda"] = torch.version.cuda
    env["script_sha256"] = file_sha256(os.path.abspath(__file__))
    return env


# ------------------------------------------------------------------ metrics

def qwk(y_true, y_pred, labels):
    index = {v: i for i, v in enumerate(labels)}
    k = len(labels)
    t = np.array([index[int(v)] for v in y_true])
    p = np.array([index[int(v)] for v in y_pred])
    observed = np.bincount(t * k + p, minlength=k * k).reshape(k, k).astype(float)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    i, j = np.indices((k, k))
    weights = (i - j) ** 2
    denominator = (weights * expected).sum()
    if denominator == 0:
        return float("nan")
    return float(1.0 - (weights * observed).sum() / denominator)


def metrics(y_true, y_pred, labels):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "mae": float(np.abs(y_pred - y_true).mean()),
        "exact": float(np.mean(y_pred == y_true)),
        "within_one": float(np.mean(np.abs(y_pred - y_true) <= 1)),
        "qwk": qwk(y_true, y_pred, labels),
    }


def bootstrap_samples(y_true, y_pred, idx, labels):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = np.abs(y_pred - y_true)
    return {
        "mae": err[idx].mean(axis=1),
        "exact": (y_pred == y_true)[idx].mean(axis=1),
        "within_one": (err <= 1)[idx].mean(axis=1),
        "qwk": np.array([qwk(y_true[r], y_pred[r], labels) for r in idx]),
    }


def interval(samples):
    valid = np.asarray(samples)[~np.isnan(samples)]
    if valid.size == 0:
        return [None, None]
    return [float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))]


def fmt(value, ci, pct=False):
    show = (lambda v: f"{v:.0%}") if pct else (lambda v: f"{v:.2f}")
    if ci is None or ci[0] is None:
        return show(value)
    return f"{show(value)} [{show(ci[0])}, {show(ci[1])}]"


# ------------------------------------------------------------------ data prep

def criterion_segment(row):
    return f"{row['criterion']}: {row['criterion_description']}"


def encode(tokenizer, rows):
    encoded = tokenizer(
        [r["text"] for r in rows],
        [criterion_segment(r) for r in rows],
        truncation="only_first", max_length=MAX_LENGTH, padding="max_length",
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def split_train_dev(rows, rng):
    """Dev slice of the training data, stratified by criterion so Thesis and
    Evidence are both represented in both halves. Neither locked holdout is
    involved -- this exists only to choose lr/epoch."""
    train, dev = [], []
    by_criterion = {}
    for row in rows:
        by_criterion.setdefault(row["criterion"], []).append(row)
    for criterion, group in sorted(by_criterion.items()):
        order = rng.permutation(len(group))
        cut = max(1, int(round(len(group) * DEV_FRACTION)))
        dev += [group[i] for i in order[:cut]]
        train += [group[i] for i in order[cut:]]
    return train, dev


# ------------------------------------------------------------------ training

def new_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, problem_type="regression")


def predict_unit(model, ids, mask, device, batch_size=BATCH_SIZE):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            logits = model(input_ids=ids[start:start + batch_size].to(device),
                           attention_mask=mask[start:start + batch_size].to(device)).logits
            out.append(logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


def to_points(unit, max_points):
    raw = np.asarray(unit, float) * np.asarray(max_points, float)
    return np.clip(np.round(raw), 0, max_points)


def train_one(lr, max_epochs, tensors, y_train_unit, dev_rows, dev_tensors, device, log):
    """Trains at one learning rate, scoring the training-data dev slice after
    each epoch. Returns the best epoch by dev MAE in points, the full history,
    and whether dev MAE rose after its best point while train loss kept
    falling -- the overfitting signal the pre-declaration asked for."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = new_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    ids, mask = tensors
    dataset = TensorDataset(ids, mask, torch.tensor(y_train_unit, dtype=torch.float32))
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)

    dev_true = np.array([r["score"] for r in dev_rows], float)
    dev_max = np.array([r["max_points"] for r in dev_rows], float)

    best = {"mae": np.inf, "epoch": None, "state": None}
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, batches = 0.0, 0
        for batch_ids, batch_mask, batch_y in loader:
            out = model(input_ids=batch_ids.to(device), attention_mask=batch_mask.to(device),
                        labels=batch_y.unsqueeze(-1).to(device))
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += float(out.loss.detach().cpu())
            batches += 1

        unit = predict_unit(model, *dev_tensors, device=device)
        points = to_points(unit, dev_max)
        mae = float(np.abs(points - dev_true).mean())
        train_mse = total / batches
        history.append({"epoch": epoch, "train_mse": train_mse, "dev_mae_points": mae})
        log(f"      epoch {epoch}: train MSE {train_mse:.4f}  dev MAE {mae:.3f} points")

        if mae < best["mae"]:
            best = {"mae": mae, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Overfitting signal: after the best epoch, did train loss keep dropping
    # while dev MAE rose above its best value by a non-trivial margin?
    after_best = [h for h in history if h["epoch"] > best["epoch"]]
    overfit_signal = bool(after_best) and (
        after_best[-1]["train_mse"] < history[best["epoch"] - 1]["train_mse"] and
        max(h["dev_mae_points"] for h in after_best) > best["mae"] + 0.1
    )
    return best, history, overfit_signal


def fit(seen_train, seen_dev, tokenizer, device, learning_rates, max_epochs, log):
    log("\n=== training (rubric-aware: essay + criterion name/description) ===")
    train_tensors = encode(tokenizer, seen_train)
    dev_tensors = encode(tokenizer, seen_dev)
    y_train_unit = np.array([r["normalised_score"] for r in seen_train], float)

    best_overall, runs, any_overfit = None, {}, False
    for lr in learning_rates:
        log(f"  lr {lr:g}:")
        best, history, overfit_signal = train_one(lr, max_epochs, train_tensors, y_train_unit,
                                                   seen_dev, dev_tensors, device, log)
        runs[f"{lr:g}"] = {"history": history, "best_epoch": best["epoch"],
                           "best_dev_mae_points": best["mae"], "overfit_signal": overfit_signal}
        if overfit_signal:
            any_overfit = True
            log(f"      note: dev MAE rose after epoch {best['epoch']} while train loss kept "
                f"falling -- possible overfitting past that point")
        if best_overall is None or best["mae"] < best_overall["mae"]:
            best_overall = {**best, "lr": lr}

    log(f"\n  chosen on dev: lr {best_overall['lr']:g}, epoch {best_overall['epoch']}, "
        f"dev MAE {best_overall['mae']:.3f} points")
    model = new_model()
    model.load_state_dict(best_overall["state"])
    info = {"grid": runs, "any_overfit_signal": any_overfit,
            "chosen": {"lr": best_overall["lr"], "epoch": best_overall["epoch"],
                       "dev_mae_points": best_overall["mae"]}}
    return model.to(device), info


# ------------------------------------------------------------------ serving cost

def measure_cpu_cost(model, tokenizer, dev_tensors, out_dir, keep_weights, log):
    """Same measurement as asap_finetune.py: a cold CPU load, then essays timed
    one at a time the way the live app grades a submission. The training-data
    dev slice is used, so neither locked holdout is touched for this."""
    cost = {}
    tmp = tempfile.mkdtemp(prefix="own_data_finetune_best_model_")
    try:
        model.save_pretrained(tmp)
        tokenizer.save_pretrained(tmp)
        cost["saved_model_mb"] = sum(os.path.getsize(os.path.join(tmp, f)) for f in os.listdir(tmp)) / 1e6

        started = time.time()
        cpu_model = AutoModelForSequenceClassification.from_pretrained(tmp)
        cpu_model.eval()
        cost["cpu_cold_load_seconds"] = time.time() - started

        ids, mask = dev_tensors
        n = min(CPU_LATENCY_ESSAYS, len(ids))
        timings = []
        with torch.no_grad():
            for i in range(min(2, len(ids))):  # warm-up
                cpu_model(input_ids=ids[i:i + 1], attention_mask=mask[i:i + 1])
            for i in range(n):
                started = time.time()
                cpu_model(input_ids=ids[i:i + 1], attention_mask=mask[i:i + 1])
                timings.append(time.time() - started)
        cost["cpu_threads"] = torch.get_num_threads()
        cost["cpu_essays_timed"] = n
        cost["cpu_ms_per_submission_mean"] = float(np.mean(timings)) * 1000
        cost["cpu_ms_per_submission_max"] = float(np.max(timings)) * 1000
        log(f"  CPU ({cost['cpu_threads']} threads, {os.cpu_count()} cores): cold load "
            f"{cost['cpu_cold_load_seconds']:.1f}s, then {cost['cpu_ms_per_submission_mean']:.0f} ms/submission "
            f"(max {cost['cpu_ms_per_submission_max']:.0f}) one at a time, padded to {MAX_LENGTH} tokens")
        log(f"  saved model {cost['saved_model_mb']:.0f} MB")

        if keep_weights:
            dest = os.path.join(out_dir, "best_model")
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(tmp, dest)
            cost["weights_saved_to"] = dest
            log(f"  weights saved to {dest}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return cost


# ------------------------------------------------------------------ comparison vs shipped

def load_baseline(path, holdout_rows, criterion, require_exact=True):
    """The shipped pipeline's own fresh predictions (from evaluate_holdout.py,
    run before this script existed), for one criterion of one holdout file.
    Returns (true, pred) arrays ordered to match holdout_rows exactly, and
    raises if the baseline doesn't cover the same essays with the same
    labels -- the paired comparison is only valid if it does.

    require_exact=False (only for --smoke, which scores a slice of the real
    file) accepts the baseline covering a superset -- the real run always
    requires the sets to match exactly, so nothing is silently dropped."""
    baseline = load_json(path, "compare against")
    block = baseline["criteria"].get(criterion)
    if block is None:
        raise SystemExit(f"{path}: no {criterion!r} block in the shipped baseline.")
    by_id = {e["essay_id"]: e for e in block["essays"]}
    ids = [r["essay_id"] for r in holdout_rows]
    covers = set(by_id) == set(ids) if require_exact else set(ids) <= set(by_id)
    if not covers:
        raise SystemExit(f"{path} ({criterion}): covers different essays than the holdout file itself.")
    true = np.array([r["score"] for r in holdout_rows], float)
    baseline_true = np.array([by_id[i]["true"] for i in ids], float)
    if not np.array_equal(true, baseline_true):
        raise SystemExit(f"{path} ({criterion}): has different true labels than the holdout file itself.")
    pred = np.array([by_id[i]["pred"] for i in ids], float)
    return true, pred, baseline.get("run_at"), baseline.get("git_head")


def evaluate_holdout_file(name, rows, model, tokenizer, device, baseline_path, log, require_exact=True):
    """Scores one locked holdout, criterion by criterion, and pairs the
    fine-tuned model's predictions against the shipped pipeline's own fresh
    predictions on the identical essays."""
    by_criterion = {}
    for row in rows:
        by_criterion.setdefault(row["criterion"], []).append(row)

    log(f"\n=== {name} (scored once) ===")
    results = {}
    for criterion, crit_rows in sorted(by_criterion.items()):
        max_points = crit_rows[0]["max_points"]
        labels = list(range(0, int(max_points) + 1))
        ids, mask = encode(tokenizer, crit_rows)
        unit = predict_unit(model, ids, mask, device)
        pred = to_points(unit, max_points)
        true = np.array([r["score"] for r in crit_rows], float)

        shipped_true, shipped_pred, baseline_run_at, baseline_git_head = \
            load_baseline(baseline_path, crit_rows, criterion, require_exact=require_exact)
        assert np.array_equal(true, shipped_true)  # load_baseline already checked this

        rng = np.random.default_rng(BOOTSTRAP_SEED)
        idx = rng.integers(0, len(true), size=(BOOTSTRAP_N, len(true)))

        finetuned_m = metrics(true, pred, labels)
        shipped_m = metrics(true, shipped_pred, labels)
        finetuned_s = bootstrap_samples(true, pred, idx, labels)
        shipped_s = bootstrap_samples(true, shipped_pred, idx, labels)

        comparisons = {}
        for metric, better_is_higher in (("mae", False), ("qwk", True), ("within_one", True), ("exact", True)):
            diff = (finetuned_s[metric] - shipped_s[metric]) if better_is_higher else \
                   (shipped_s[metric] - finetuned_s[metric])
            point = finetuned_m[metric] - shipped_m[metric]
            if not better_is_higher:
                point = -point
            ci = interval(diff)
            comparisons[metric] = {"finetuned_minus_shipped": float(point), "ci": ci,
                                   "distinguishable": bool(ci[0] is not None and (ci[0] > 0 or ci[1] < 0))}

        log(f"  {criterion} (0-{max_points:g}), n={len(true)}:")
        log(f"    fine-tuned   MAE {fmt(finetuned_m['mae'], interval(finetuned_s['mae']))}  "
            f"QWK {fmt(finetuned_m['qwk'], interval(finetuned_s['qwk']))}  "
            f"within-1 {fmt(finetuned_m['within_one'], interval(finetuned_s['within_one']), True)}  "
            f"exact {fmt(finetuned_m['exact'], interval(finetuned_s['exact']), True)}")
        log(f"    shipped      MAE {fmt(shipped_m['mae'], interval(shipped_s['mae']))}  "
            f"QWK {fmt(shipped_m['qwk'], interval(shipped_s['qwk']))}  "
            f"within-1 {fmt(shipped_m['within_one'], interval(shipped_s['within_one']), True)}  "
            f"exact {fmt(shipped_m['exact'], interval(shipped_s['exact']), True)}")
        for metric, c in comparisons.items():
            mark = "distinguishable" if c["distinguishable"] else "NOT distinguishable"
            log(f"    fine-tuned - shipped, {metric:11} {c['finetuned_minus_shipped']:+.3f} "
                f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] -- {mark}")

        results[criterion] = {
            "n": len(true), "max_points": max_points,
            "fine_tuned": finetuned_m, "shipped": shipped_m, "comparisons": comparisons,
            "shipped_baseline_run_at": baseline_run_at, "shipped_baseline_git_head": baseline_git_head,
            "per_essay": [
                {"essay_id": r["essay_id"], "true": float(true[i]),
                 "fine_tuned_pred": float(pred[i]), "shipped_pred": float(shipped_pred[i]),
                 "words": len(r["text"].split())}
                for i, r in enumerate(crit_rows)
            ],
        }
    return results


# ------------------------------------------------------------------ main

def build_parser():
    p = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on our own data, compare with the shipped pipeline (pre-declared design).")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--dry-run", action="store_true", help="pre-flight only; train nothing")
    p.add_argument("--smoke", action="store_true",
                   help="one lr, one epoch, a slice of the data; writes to smoke.json, never touches a locked holdout")
    p.add_argument("--allow-no-gpu", action="store_true")
    p.add_argument("--no-save-weights", action="store_true")
    p.add_argument("--rescore-test", action="store_true",
                   help="run although results already exist, i.e. score the locked holdouts a second time")
    return p


def main():
    args = build_parser().parse_args()
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir or OUT_DIR))
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "smoke.json" if args.smoke else "own_data_finetune.json")

    if not (args.smoke or args.dry_run or args.rescore_test) and os.path.exists(results_path):
        raise SystemExit(
            f"{results_path} already exists, so both locked holdouts have been scored. Running again "
            "would score them twice, which the pre-declaration rules out. Pass --rescore-test only for "
            "a deliberate re-run that will be disclosed.")

    log_path = os.path.join(out_dir, "own_data_finetune_log.txt")
    open(log_path, "w").close()
    log = log_to(log_path)

    device = pick_device()
    require_gpu(device, args.allow_no_gpu, light=args.smoke or args.dry_run)
    env = environment(device)
    log("environment: " + ", ".join(f"{k} {v}" for k, v in env.items() if k != "script_sha256"))
    log(f"script sha256 {env['script_sha256']}")

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir)) if args.data_dir else None
    paths = resolve_paths(data_dir)

    # ---- 1. pre-flight -----------------------------------------------------
    log("\n=== pre-flight ===")
    for key, filename in (("train", "rubric_conditioning_train.json"),
                          ("holdout2", "holdout2_team_scored.json"),
                          ("essay2_holdout", "holdout_essay2_team_scored.json")):
        if not os.path.exists(paths[key]):
            raise SystemExit(f"PRE-FLIGHT FAILED: {paths[key]} not found.")
        actual = file_sha256(paths[key])
        locked = LOCKED_SHA[filename]
        if actual != locked:
            raise SystemExit(f"PRE-FLIGHT FAILED: {filename} hashes to {actual}, not the locked {locked}. "
                             "It has been edited since the design was pre-declared.")
        log(f"  {filename}: matches the locked file ({actual})")

    train_rows = load_json(paths["train"], "train on")
    holdout2_rows = enrich(load_json(paths["holdout2"], "evaluate"), "holdout2_team_scored.json")
    essay2_rows = enrich(load_json(paths["essay2_holdout"], "evaluate"), "holdout_essay2_team_scored.json")
    log(f"  training triples: {len(train_rows)}; holdout2: {len(holdout2_rows)} rows; "
        f"essay2 holdout: {len(essay2_rows)} rows")
    log("  holdout_team_scored.json (\"holdout1\", burned as a dev set in an earlier round) "
        "is not loaded by this script")

    train_texts = {normalise(r["text"]) for r in train_rows}
    for name, rows in (("holdout2", holdout2_rows), ("essay2 holdout", essay2_rows)):
        hits = [r["essay_id"] for r in rows if normalise(r["text"]) in train_texts]
        if hits:
            raise SystemExit(f"PRE-FLIGHT FAILED: {name} essays {hits} appear verbatim in the training text.")
    log("  no holdout essay appears verbatim in the training text "
        "(the fuller cosine-similarity leakage guard was already run by evaluate_holdout.py "
        "when the fresh baseline was produced: no duplicates on either clean holdout)")

    # The shipped baseline files must cover exactly these essays with exactly
    # these labels -- otherwise the paired comparison below would be comparing
    # the new model against a stale or mismatched run.
    for crit_rows, baseline_path, label in (
        (holdout2_rows, paths["baseline_holdout2"], "holdout2"),
        (essay2_rows, paths["baseline_essay2"], "essay2 holdout"),
    ):
        by_criterion = {}
        for row in crit_rows:
            by_criterion.setdefault(row["criterion"], []).append(row)
        for criterion, rows in by_criterion.items():
            load_baseline(baseline_path, rows, criterion)  # raises on any mismatch
        log(f"  {label}: the shipped baseline file covers the same essays, same labels")
    log("PRE-FLIGHT PASSED")
    if args.dry_run:
        log("--dry-run: stopping before training")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    rng = np.random.default_rng(SEED)
    seen_train, seen_dev = split_train_dev(train_rows, rng)
    learning_rates, max_epochs = LEARNING_RATES, MAX_EPOCHS
    if args.smoke:
        seen_train, seen_dev = seen_train[:24], seen_dev[:8]
        holdout2_rows, essay2_rows = holdout2_rows[:4], essay2_rows[:4]
        learning_rates, max_epochs = LEARNING_RATES[:1], 1
    log(f"\ntrain/dev split: {len(seen_train)} train, {len(seen_dev)} dev "
        f"(dev is for choosing lr/epoch only; neither locked holdout is used for a choice)")

    # ---- 2. train -----------------------------------------------------------
    model, fit_info = fit(seen_train, seen_dev, tokenizer, device, learning_rates, max_epochs, log)
    if fit_info["any_overfit_signal"]:
        log("\n  OVERFITTING SIGNAL: at least one learning rate's dev MAE rose after its best epoch "
            "while train loss kept falling. See the per-epoch history below.")
    else:
        log("\n  no overfitting signal: dev MAE did not rise after the best epoch at any learning rate "
            "while train loss kept falling")

    # ---- 3. both locked holdouts, scored once -------------------------------
    if args.smoke:
        log("\n--smoke: scoring a 4-essay slice of each holdout file, not the real one")
    holdout2_results = evaluate_holdout_file("holdout2 (clean, locked, Essay 1)", holdout2_rows,
                                             model, tokenizer, device, paths["baseline_holdout2"], log,
                                             require_exact=not args.smoke)
    essay2_results = evaluate_holdout_file("essay2 holdout (clean, locked, Essay 2)", essay2_rows,
                                           model, tokenizer, device, paths["baseline_essay2"], log,
                                           require_exact=not args.smoke)

    # ---- cost ----------------------------------------------------------------
    parameters = sum(p.numel() for p in model.parameters())
    log(f"\n=== cost ===")
    log(f"  {parameters / 1e6:.1f}M parameters, device {env.get('gpu', device.type)}")
    cost = {"parameters": int(parameters), "device": device.type}
    try:
        dev_tensors = encode(tokenizer, seen_dev)
        cost.update(measure_cpu_cost(model, tokenizer, dev_tensors, out_dir,
                                     keep_weights=not (args.smoke or args.no_save_weights), log=log))
    except Exception as exc:  # the holdout results are already computed
        log(f"  CPU cost measurement failed ({type(exc).__name__}: {exc}); holdout results are unaffected")

    payload = {
        "pre_declared_in": "training_data/results/own_data_finetune_results.md",
        "smoke": bool(args.smoke),
        "model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "environment": env,
        "data": {"train_triples": len(train_rows), "seen_train": len(seen_train), "seen_dev": len(seen_dev),
                 "holdout2": len(holdout2_rows), "essay2_holdout": len(essay2_rows),
                 "locked_sha": LOCKED_SHA},
        "fit": fit_info,
        "holdout2": holdout2_results,
        "essay2_holdout": essay2_results,
        "cost": cost,
        "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED},
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v))
        f.write("\n")
    log(f"\nSaved {results_path}")
    log("\nReminder: 12 and 8 essays respectively. A paired difference whose interval contains zero "
        "is not a result, whichever direction the point estimate points.")


if __name__ == "__main__":
    main()
