"""Can a grader score a criterion it has never trained on, by reading that
criterion's description?

Every grader this project ships is keyed to one criterion row id: it learns
"Thesis" as an opaque label and cannot score anything else. This asks whether
conditioning on the criterion's *description* instead gives a model something
transferable -- so that a teacher adding a new criterion gets a usable score
without collecting labels for it first.

Runs on a GPU (Google Colab's free T4), not a laptop: the earlier fine-tuning
attempt froze an 8 GB Apple M1. scripts/rubric_conditioning_colab.ipynb runs it.

    python scripts/rubric_conditioning.py --data-dir DIR --out-dir DIR
    python scripts/rubric_conditioning.py --data-dir DIR --out-dir DIR --dry-run
    python scripts/rubric_conditioning.py --data-dir DIR --out-dir DIR --smoke

--data-dir is one flat folder (on Google Drive, for Colab) holding:
    rubric_conditioning_train.json            128 Thesis/Evidence triples
    holdout_counterargument_team_scored.json  the 8 locked zero-shot essays
Without it, the repo layout is used.

The design is pre-declared in training_data/results/rubric_conditioning_results.md
and committed before this runs. Nothing here may change in response to the test
numbers. The order of work protects that:

  1. Pre-flight: the holdout must be the locked one, and no holdout essay may
     appear in training. Checked before a weight moves.
  2. A fixed learning-rate grid, chosen on a dev slice of the SEEN criteria.
     The Counterargument essays are never used for any choice.
  3. Two models on identical data and architecture -- (A) with the criterion
     description, (B) with it replaced by a placeholder -- so the difference
     isolates whether the description does anything.
  4. Test scored once. A second run into the same --out-dir is refused.

Isolation: writes only to --out-dir. No database, no Flask app, and it never
reads or writes ml_models/. The shipped graders are untouched.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BACKEND_DIR, "ml_experiments")

# ---- the pre-declared design; tests pin every value ------------------------
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 384          # essay + criterion description, as a sentence pair
BATCH_SIZE = 8
LEARNING_RATES = (2e-5, 3e-5)
MAX_EPOCHS = 4
SEED = 20261005
DEV_FRACTION = 0.2        # of the seen-criteria data, stratified by criterion

# The ablation's stand-in for a description. Constant, so model B can never
# tell one criterion from another.
PLACEHOLDER_CRITERION = "criterion: unspecified"

COUNTERARGUMENT_MAX = 5
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 0

# The holdout as committed in 54cc4e1, before any of this existed. Pinned so a
# pre-flight can prove the test set was not edited after the fact.
LOCKED_HOLDOUT_SHA = "ca94cac65049f18b"


def log_to(path):
    def log(message=""):
        print(message, flush=True)
        with open(path, "a") as f:
            f.write(message + "\n")
    return log


# ------------------------------------------------------------------ inputs

def resolve_paths(data_dir=None):
    if data_dir is None:
        return {
            "train": os.path.join(BACKEND_DIR, "training_data", "rubric_conditioning_train.json"),
            "holdout": os.path.join(BACKEND_DIR, "training_data",
                                    "holdout_counterargument_team_scored.json"),
        }
    return {
        "train": os.path.join(data_dir, "rubric_conditioning_train.json"),
        "holdout": os.path.join(data_dir, "holdout_counterargument_team_scored.json"),
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
    import re
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


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

def criterion_segment(row, with_description):
    """The second half of the sentence pair.

    Model A reads "Name: description". Model B reads a constant placeholder, so
    it cannot tell one criterion from another and must rely on the essay alone.
    """
    if not with_description:
        return PLACEHOLDER_CRITERION
    return f"{row['criterion']}: {row['criterion_description']}"


def encode(tokenizer, rows, with_description):
    encoded = tokenizer(
        [r["text"] for r in rows],
        [criterion_segment(r, with_description) for r in rows],
        truncation="only_first", max_length=MAX_LENGTH, padding="max_length",
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def split_seen(rows, rng):
    """Dev slice of the seen criteria, stratified by criterion so both are
    represented. The Counterargument essays are never involved."""
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


def predict_unit(model, ids, mask, device):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(ids), BATCH_SIZE):
            logits = model(input_ids=ids[start:start + BATCH_SIZE].to(device),
                           attention_mask=mask[start:start + BATCH_SIZE].to(device)).logits
            out.append(logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


def to_points(unit, max_points):
    """0-1 back to whole marks on this criterion's own scale."""
    raw = np.asarray(unit, float) * max_points
    return np.clip(np.round(raw), 0, max_points)


def train_one(lr, max_epochs, tensors, y_train_unit, dev_rows, dev_tensors, device, log):
    """Trains at one learning rate, scoring the seen-criteria dev slice after
    each epoch. Returns the best epoch by dev MAE in points."""
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
        points = np.clip(np.round(unit * dev_max), 0, dev_max)
        mae = float(np.abs(points - dev_true).mean())
        history.append({"epoch": epoch, "train_loss": total / batches, "dev_mae_points": mae})
        log(f"      epoch {epoch}: train MSE {total / batches:.4f}  dev MAE {mae:.3f} points")

        if mae < best["mae"]:
            best = {"mae": mae, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best, history


def fit_variant(name, with_description, seen_train, seen_dev, tokenizer, device,
                learning_rates, max_epochs, log):
    """Runs the grid for one variant and returns its best model."""
    log(f"\n=== {name} ===")
    train_tensors = encode(tokenizer, seen_train, with_description)
    dev_tensors = encode(tokenizer, seen_dev, with_description)
    y_train_unit = np.array([r["normalised_score"] for r in seen_train], float)

    best_overall, runs = None, {}
    for lr in learning_rates:
        log(f"  lr {lr:g}:")
        best, history = train_one(lr, max_epochs, train_tensors, y_train_unit,
                                  seen_dev, dev_tensors, device, log)
        runs[f"{lr:g}"] = {"history": history, "best_epoch": best["epoch"],
                           "best_dev_mae_points": best["mae"]}
        if best_overall is None or best["mae"] < best_overall["mae"]:
            best_overall = {**best, "lr": lr}

    log(f"  chosen on dev: lr {best_overall['lr']:g}, epoch {best_overall['epoch']}, "
        f"dev MAE {best_overall['mae']:.3f} points")
    model = new_model()
    model.load_state_dict(best_overall["state"])
    return model.to(device), {"grid": runs, "chosen": {"lr": best_overall["lr"],
                                                       "epoch": best_overall["epoch"],
                                                       "dev_mae_points": best_overall["mae"]}}


# ------------------------------------------------------------------ main

def build_parser():
    p = argparse.ArgumentParser(description="Does reading a criterion's description transfer?")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--dry-run", action="store_true", help="pre-flight only; train nothing")
    p.add_argument("--smoke", action="store_true",
                   help="one lr, one epoch, a slice of the data; writes to smoke.json")
    p.add_argument("--allow-no-gpu", action="store_true")
    p.add_argument("--rescore-test", action="store_true",
                   help="run although results already exist, i.e. score the holdout a second time")
    return p


def main():
    args = build_parser().parse_args()
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir or OUT_DIR))
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "smoke.json" if args.smoke else "rubric_conditioning.json")

    if not (args.smoke or args.dry_run or args.rescore_test) and os.path.exists(results_path):
        raise SystemExit(
            f"{results_path} already exists, so the holdout has been scored. Running again "
            "would score it twice, which the pre-declaration rules out. Pass --rescore-test "
            "only for a deliberate re-run that will be disclosed.")

    log_path = os.path.join(out_dir, "rubric_conditioning_log.txt")
    open(log_path, "w").close()
    log = log_to(log_path)

    device = pick_device()
    require_gpu(device, args.allow_no_gpu, light=args.smoke or args.dry_run)
    env = environment(device)
    log("environment: " + ", ".join(f"{k} {v}" for k, v in env.items() if k != "script_sha256"))
    log(f"script sha256 {env['script_sha256']}")

    paths = resolve_paths(os.path.abspath(os.path.expanduser(args.data_dir)) if args.data_dir else None)
    seen_rows = load_json(paths["train"], "train on")
    holdout = load_json(paths["holdout"], "evaluate")

    log(f"\ntraining triples: {len(seen_rows)} (seen criteria)")
    log(f"zero-shot holdout: {len(holdout)} essays, criterion "
        f"{holdout[0]['criterion']!r} (0-{holdout[0]['max_points']})")

    # ---- 1. pre-flight ----------------------------------------------------
    log("\n=== pre-flight ===")
    unseen = {r["criterion"] for r in holdout}
    trained_on = {r["criterion"] for r in seen_rows}
    overlap = unseen & trained_on
    if overlap:
        raise SystemExit(f"PRE-FLIGHT FAILED: {overlap} appears in training; this is not zero-shot.")
    log(f"  training criteria {sorted(trained_on)}; holdout criterion {sorted(unseen)} -- disjoint")

    train_texts = {normalise(r["text"]) for r in seen_rows}
    exact_hits = [r["essay_id"] for r in holdout if normalise(r["text"]) in train_texts]
    if exact_hits:
        raise SystemExit(f"PRE-FLIGHT FAILED: holdout essays {exact_hits} appear verbatim in training.")
    log("  no holdout essay appears verbatim in the training text")

    actual_sha = file_sha256(paths["holdout"])
    if actual_sha != LOCKED_HOLDOUT_SHA:
        raise SystemExit(
            f"PRE-FLIGHT FAILED: the holdout file hashes to {actual_sha}, not the locked "
            f"{LOCKED_HOLDOUT_SHA}. It has been edited since it was committed, so it is no "
            "longer a held-out test set.")
    log(f"  holdout matches the locked file committed before any training ({actual_sha})")
    log("  (essay 5 is a near-duplicate of a training row at cosine 0.95 -- disclosed in the")
    log("   pre-declaration, and reported again below as a sensitivity pass)")
    log("PRE-FLIGHT PASSED")
    if args.dry_run:
        log("--dry-run: stopping before training")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    rng = np.random.default_rng(SEED)
    seen_train, seen_dev = split_seen(seen_rows, rng)
    learning_rates, max_epochs = LEARNING_RATES, MAX_EPOCHS
    if args.smoke:
        seen_train, seen_dev = seen_train[:24], seen_dev[:8]
        learning_rates, max_epochs = LEARNING_RATES[:1], 1
    log(f"\nseen-criteria split: {len(seen_train)} train, {len(seen_dev)} dev "
        f"(dev is for choosing only; the holdout is never used for a choice)")

    # ---- 2+3. the two variants -------------------------------------------
    variants = {}
    model_a, info_a = fit_variant("A: rubric-aware (reads the criterion description)", True,
                                  seen_train, seen_dev, tokenizer, device,
                                  learning_rates, max_epochs, log)
    variants["rubric-aware"] = (model_a, info_a, True)
    model_b, info_b = fit_variant("B: ablation (criterion description replaced by a placeholder)", False,
                                  seen_train, seen_dev, tokenizer, device,
                                  learning_rates, max_epochs, log)
    variants["no-description"] = (model_b, info_b, False)

    # ---- 4. scored once ---------------------------------------------------
    labels = list(range(0, COUNTERARGUMENT_MAX + 1))
    y_true = np.array([r["score"] for r in holdout], float)
    rng_boot = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng_boot.integers(0, len(holdout), size=(BOOTSTRAP_N, len(holdout)))

    predictions = {
        # Pre-declared baseline: the midpoint of the range, for a criterion with
        # no training-derived mean to fall back on.
        "midpoint baseline": np.full(len(holdout), float(np.round(COUNTERARGUMENT_MAX / 2))),
    }
    for name, (model, _, with_description) in variants.items():
        ids, mask = encode(tokenizer, holdout, with_description)
        predictions[name] = to_points(predict_unit(model, ids, mask, device), COUNTERARGUMENT_MAX)

    log("\n=== ZERO-SHOT TEST: Counterargument, scored once ===")
    results, samples = {}, {}
    for name, pred in predictions.items():
        results[name] = metrics(y_true, pred, labels)
        samples[name] = bootstrap_samples(y_true, pred, idx, labels)
        results[name]["ci"] = {k: interval(v) for k, v in samples[name].items()}
        r = results[name]
        log(f"  {name:20} MAE {fmt(r['mae'], r['ci']['mae'])}  "
            f"QWK {fmt(r['qwk'], r['ci']['qwk'])}  "
            f"exact {fmt(r['exact'], r['ci']['exact'], True)}  "
            f"within-1 {fmt(r['within_one'], r['ci']['within_one'], True)}")

    log("\n  per-essay (true -> rubric-aware / ablation):")
    for i, row in enumerate(holdout):
        log(f"    essay {row['essay_id']}: {y_true[i]:.0f} -> "
            f"{predictions['rubric-aware'][i]:.0f} / {predictions['no-description'][i]:.0f}")

    # ---- the question the experiment exists to answer ---------------------
    log("\n=== A vs B: does reading the description help? ===")
    comparisons = {}
    for metric, better_is_higher in (("mae", False), ("qwk", True), ("within_one", True), ("exact", True)):
        a, b = samples["rubric-aware"][metric], samples["no-description"][metric]
        diff = (a - b) if better_is_higher else (b - a)   # positive = A better
        point = (results["rubric-aware"][metric] - results["no-description"][metric])
        if not better_is_higher:
            point = -point
        ci = interval(diff)
        comparisons[metric] = {"a_minus_b": float(point), "ci": ci,
                               "distinguishable": bool(ci[0] is not None and (ci[0] > 0 or ci[1] < 0))}
        verdict = "distinguishable" if comparisons[metric]["distinguishable"] else "NOT distinguishable"
        log(f"  {metric:11} A better by {point:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] -- {verdict}")

    # sensitivity: the contaminated essay removed
    keep = [i for i, r in enumerate(holdout) if r["essay_id"] != 5]
    idx_small = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(keep), size=(BOOTSTRAP_N, len(keep)))
    log("\n  without essay 5 (near-duplicate of a training row, cosine 0.95):")
    sensitivity = {}
    for name in ("rubric-aware", "no-description", "midpoint baseline"):
        sub = metrics(y_true[keep], predictions[name][keep], labels)
        sensitivity[name] = sub
        log(f"    {name:20} MAE {sub['mae']:.2f}  QWK {sub['qwk']:.2f}")

    # ---- 5. did A get worse at what it was trained on? --------------------
    log("\n=== sanity check on the SEEN criteria (dev slice) ===")
    seen_check = {}
    dev_true = np.array([r["score"] for r in seen_dev], float)
    dev_max = np.array([r["max_points"] for r in seen_dev], float)
    for name, (model, _, with_description) in variants.items():
        ids, mask = encode(tokenizer, seen_dev, with_description)
        unit = predict_unit(model, ids, mask, device)
        points = np.clip(np.round(unit * dev_max), 0, dev_max)
        seen_check[name] = {
            "mae_points": float(np.abs(points - dev_true).mean()),
            "within_one": float(np.mean(np.abs(points - dev_true) <= 1)),
        }
        log(f"  {name:20} MAE {seen_check[name]['mae_points']:.3f} points, "
            f"within-1 {seen_check[name]['within_one']:.0%}")

    payload = {
        "pre_declared_in": "training_data/results/rubric_conditioning_results.md",
        "smoke": bool(args.smoke),
        "environment": env,
        "config": {"model": MODEL_NAME, "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
                   "learning_rates": list(learning_rates), "max_epochs": max_epochs,
                   "seed": SEED, "dev_fraction": DEV_FRACTION,
                   "placeholder": PLACEHOLDER_CRITERION,
                   "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED}},
        "data": {"train_triples": len(seen_rows), "seen_train": len(seen_train),
                 "seen_dev": len(seen_dev), "holdout": len(holdout),
                 "train_sha256": file_sha256(paths["train"]),
                 "holdout_sha256": file_sha256(paths["holdout"])},
        "variants": {name: info for name, (_, info, _) in variants.items()},
        "zero_shot": results,
        "per_essay": [
            {"essay_id": r["essay_id"], "true": float(y_true[i]),
             "rubric_aware": float(predictions["rubric-aware"][i]),
             "no_description": float(predictions["no-description"][i]),
             "words": len(r["text"].split())}
            for i, r in enumerate(holdout)
        ],
        "a_vs_b": comparisons,
        "sensitivity_without_essay_5": sensitivity,
        "seen_criteria_dev": seen_check,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
        f.write("\n")
    log(f"\nSaved {results_path}")
    log("\nReminder: 8 essays. Every interval here is wide, and a point estimate "
        "whose interval contains zero is not a result.")


if __name__ == "__main__":
    main()
