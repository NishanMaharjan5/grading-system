"""Fine-tunes DistilBERT end to end on ASAP-AES set 1 and compares it with the
five systems already benchmarked in asap_results.md.

Runs on a GPU -- Google Colab's free T4 -- not on a laptop: the full grid froze
an 8 GB Apple M1. scripts/asap_finetune_colab.ipynb runs it cell by cell.

Usage:
    python scripts/asap_finetune.py --data-dir DIR --out-dir DIR             # the run
    python scripts/asap_finetune.py --data-dir DIR --out-dir DIR --dry-run   # pre-flight only
    python scripts/asap_finetune.py --data-dir DIR --out-dir DIR --smoke     # every code path, tiny

--data-dir is one flat folder (on Google Drive, for Colab) holding:
    essays.xlsx              ASAP-AES training set -- licensed, never committed
    asap_split.json          the split locked in f3c5d17, before anything was fit
    asap.json                the published benchmark results (training_data/results/)
    embedding_cache/*.npy    the benchmark's cached frozen embeddings
Without --data-dir the repo layout is used (backend/data/asap/, training_data/).

--out-dir (default ml_experiments/, gitignored) receives asap_finetune.json,
finetune_log.txt -- written line by line, so a dropped Colab session still
leaves it -- and best_model/, the chosen weights in Hugging Face format.

Why this experiment: the benchmark found the *frozen* MiniLM embedding adds
almost nothing -- word count alone matches the shipped pipeline. That is a
claim about frozen features, not about transformers. A model fine-tuned on
these essays learns its own representation, so this separates "the approach
was the limit" from "the frozen features were the limit".

The design is pre-declared in training_data/asap_results.md (committed in
494bd77, before the first attempt). Nothing here may change in response to
test numbers. The order of work protects that:

  1. Pre-flight, before any training: the split must be the locked one, and
     the two published reference systems must reproduce their benchmark
     numbers exactly from the cached embeddings. Anything off stops the run
     here, before a new candidate has seen test. (Re-deriving those two
     systems reads test labels, but only to reproduce numbers already
     published.)
  2. The learning-rate / epoch grid, chosen on dev.
  3. Candidate B's blend weight, chosen on dev.
  4. Test, scored once. A second run into the same --out-dir is refused.

Isolation: writes only to --out-dir. It touches no database, builds no Flask
app, and never reads or writes ml_models/. The production grader is
unaffected by anything in this file.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import tempfile
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BACKEND_DIR, "ml_experiments")

# ---- the pre-declared design (asap_results.md); tests pin every value -------
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATES = (2e-5, 3e-5)
MAX_EPOCHS = 4
SEED = 20260928
BLEND_WEIGHTS = (0.25, 0.5, 0.75)

LOCKED_SPLIT_FINGERPRINT = "828dbd347ea2cf81"  # committed in f3c5d17, before anything was fit
CPU_LATENCY_ESSAYS = 20

SCORE_MIN, SCORE_MAX = 2, 12
SCORE_RANGE = SCORE_MAX - SCORE_MIN


def _load_benchmark():
    """Reuse the benchmark's cleaning, split loading, metrics and bootstrap so
    the two experiments are measured identically."""
    path = os.path.join(BACKEND_DIR, "scripts", "asap_benchmark.py")
    spec = importlib.util.spec_from_file_location("asap_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark()

# Captured before anything can repoint them, so resolve_paths(None) always
# means the repo layout.
_REPO_PATHS = {
    "essays": bench.DATA_PATH,
    "split": bench.SPLIT_PATH,
    "published": os.path.join(BACKEND_DIR, "training_data", "results", "asap.json"),
    "cache": bench.CACHE_DIR,
}


# --------------------------------------------------------------- target scaling

def to_unit(scores):
    """Rubric score -> 0-1, the regression target."""
    return (np.asarray(scores, dtype=float) - SCORE_MIN) / SCORE_RANGE


def rescale(unit):
    """0-1 -> the continuous 2-12 scale, unrounded. Blending happens here, on
    continuous values, so candidate B rounds once rather than twice."""
    return np.asarray(unit, dtype=float) * SCORE_RANGE + SCORE_MIN


def from_unit(unit):
    """0-1 -> a whole rubric score, rounded and clamped exactly as every other
    system in this comparison does it."""
    return np.clip(np.round(rescale(unit)), SCORE_MIN, SCORE_MAX)


# --------------------------------------------------------------- inputs

def resolve_paths(data_dir=None):
    """Where the inputs live: the repo layout by default, or one flat folder
    (Google Drive, on Colab) with --data-dir."""
    if data_dir is None:
        return dict(_REPO_PATHS)
    return {
        "essays": os.path.join(data_dir, "essays.xlsx"),
        "split": os.path.join(data_dir, "asap_split.json"),
        "published": os.path.join(data_dir, "asap.json"),
        "cache": os.path.join(data_dir, "embedding_cache"),
    }


def apply_paths(paths):
    """Point the benchmark's loaders at these files. Its functions read these
    module-level names when they run, so reassigning them is enough and
    asap_benchmark.py itself stays untouched."""
    bench.DATA_PATH = paths["essays"]
    bench.SPLIT_PATH = paths["split"]
    bench.CACHE_DIR = paths["cache"]


def expected_cache_path(name, texts):
    """The file bench.cached_embeddings reads for these texts. Mirrors its
    naming, so a missing cache is caught before training instead of being
    silently recomputed on a different device (a test pins the two together)."""
    key = hashlib.sha256(("||".join(texts) + name).encode()).hexdigest()[:20]
    return os.path.join(bench.CACHE_DIR, f"{name}_{key}.npy")


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# --------------------------------------------------------------- hardware

def pick_device():
    """CUDA (Colab's GPU) first; MPS and CPU only as fallbacks."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def require_gpu(device, allow_no_gpu=False, light=False):
    """The full grid is only safe on a real GPU: on an 8 GB Apple M1 it froze
    the machine. Light runs (--smoke, --dry-run) are exempt; otherwise running
    without CUDA takes an explicit --allow-no-gpu."""
    if device.type == "cuda" or light or allow_no_gpu:
        return
    raise SystemExit(
        f"No CUDA GPU found (device: {device.type}). The full grid froze an 8 GB Apple M1. "
        "Run it on Colab with a T4 runtime, or pass --allow-no-gpu to insist.")


def free_device_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def environment(device):
    """Recorded rather than pinned: Colab's library versions move, so the
    output says exactly what this run used."""
    import pandas
    import sklearn
    import transformers

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "device": device.type,
        "cpu_count": os.cpu_count(),
    }
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(0)
        env["cuda"] = torch.version.cuda
    env["script_sha256"] = {
        name: file_sha256(os.path.join(BACKEND_DIR, "scripts", name))
        for name in ("asap_finetune.py", "asap_benchmark.py")
    }
    return env


# --------------------------------------------------------------- training

def seed_everything():
    torch.manual_seed(SEED)  # seeds the CPU, CUDA and MPS generators
    np.random.seed(SEED)


def tokenize(tokenizer, texts):
    encoded = tokenizer(list(texts), truncation=True, max_length=MAX_LENGTH,
                        padding="max_length", return_tensors="pt")
    return encoded["input_ids"], encoded["attention_mask"]


def predict(model, ids, mask, device, batch_size=BATCH_SIZE):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size].to(device)
            batch_mask = mask[start:start + batch_size].to(device)
            logits = model(input_ids=batch_ids, attention_mask=batch_mask).logits
            out.append(logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


def new_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, problem_type="regression")


def train_one(lr, max_epochs, tensors, y_train, y_dev, device, log):
    """Trains at one learning rate, scoring dev after every epoch. Returns the
    best epoch by dev QWK (its weights copied to CPU) and the full history."""
    seed_everything()
    model = new_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_ids, train_mask = tensors["train"]
    dataset = TensorDataset(train_ids, train_mask, torch.tensor(to_unit(y_train), dtype=torch.float32))
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)

    best = {"qwk": -np.inf, "epoch": None, "state": None}
    history = []
    for epoch in range(1, max_epochs + 1):
        started = time.time()
        model.train()
        epoch_loss, batches = 0.0, 0
        for batch_ids, batch_mask, batch_y in loader:
            outputs = model(input_ids=batch_ids.to(device), attention_mask=batch_mask.to(device),
                            labels=batch_y.unsqueeze(-1).to(device))
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += float(outputs.loss.detach().cpu())
            batches += 1
        seconds = time.time() - started

        dev_pred = from_unit(predict(model, *tensors["dev"], device=device))
        scores = bench.metrics(y_dev, dev_pred)
        history.append({"epoch": epoch, "train_loss": epoch_loss / batches, "train_seconds": seconds, **scores})
        log(f"      epoch {epoch}: train MSE {epoch_loss / batches:.4f}  dev QWK {scores['qwk']:.3f}  "
            f"dev MAE {scores['mae']:.3f}  ({seconds:.0f}s)")

        if scores["qwk"] > best["qwk"]:  # strict: ties keep the earlier epoch
            best = {"qwk": scores["qwk"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    del model, optimizer
    free_device_memory(device)
    return best, history


def reference_systems(parts, y, hand):
    """Refits the two published systems the fine-tuned model is compared with
    -- handcrafted only, and the shipped embeddings + handcrafted -- on the
    same split and the same cached frozen embeddings as the benchmark."""
    embeddings = {}
    for name in ("train", "test"):
        texts = list(parts[name].text)
        embeddings[name] = bench.cached_embeddings(f"single_{name}", texts, lambda texts=texts: bench.embed(texts))

    ridge = bench.make_model().fit(hand["train"], y["train"])
    shipped = bench.make_model().fit(np.hstack([embeddings["train"], hand["train"]]), y["train"])
    predictions = {
        "handcrafted only": bench.to_scores(ridge.predict(hand["test"])),
        "embeddings + handcrafted": bench.to_scores(
            shipped.predict(np.hstack([embeddings["test"], hand["test"]]))),
    }
    return ridge, predictions


# --------------------------------------------------------------- serving cost

def measure_cpu_cost(model, tokenizer, dev_tensors, out_dir, keep_weights, log):
    """What serving this would cost without a GPU: save the chosen weights,
    time a cold load on CPU, then time essays one at a time -- the way the
    live app grades a submission. Dev essays are used, so test is still
    scored exactly once. Every essay is padded to the full 512 tokens, so
    these are worst-case timings."""
    cost = {}
    tmp = tempfile.mkdtemp(prefix="asap_best_model_")
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
        cost["cpu_ms_per_essay_mean"] = float(np.mean(timings)) * 1000
        cost["cpu_ms_per_essay_max"] = float(np.max(timings)) * 1000
        log(f"  CPU ({cost['cpu_threads']} threads, {os.cpu_count()} cores): cold load "
            f"{cost['cpu_cold_load_seconds']:.1f}s, then {cost['cpu_ms_per_essay_mean']:.0f} ms/essay "
            f"(max {cost['cpu_ms_per_essay_max']:.0f}) one at a time, padded to {MAX_LENGTH} tokens")
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


def write_json(path, payload):
    def default(value):
        return value.item() if hasattr(value, "item") else str(value)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=default)
        f.write("\n")


# --------------------------------------------------------------- main

def build_parser():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on ASAP set 1 (pre-declared design).")
    parser.add_argument("--data-dir", default=None,
                        help="flat folder with essays.xlsx, asap_split.json, asap.json and embedding_cache/ "
                             "(e.g. on Google Drive); default: the repo layout")
    parser.add_argument("--out-dir", default=None,
                        help="where results, the log and best_model/ go; default: ml_experiments/ (gitignored)")
    parser.add_argument("--dry-run", action="store_true",
                        help="pre-flight only: load the data, verify the locked split and the reference systems, stop")
    parser.add_argument("--smoke", action="store_true",
                        help="exercise every code path on a few train/dev essays, one epoch, one learning rate; "
                             "a dev slice stands in for test, so the locked test split is never read")
    parser.add_argument("--allow-no-gpu", action="store_true",
                        help="run the full grid without CUDA anyway (it froze an 8 GB Apple M1)")
    parser.add_argument("--no-save-weights", action="store_true",
                        help="don't copy the chosen weights into --out-dir/best_model")
    parser.add_argument("--rescore-test", action="store_true",
                        help="run although --out-dir already holds a finished run, i.e. score test a second "
                             "time; breaks the one-look rule and must be disclosed")
    return parser


def main():
    args = build_parser().parse_args()
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir or OUT_DIR))
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "smoke.json" if args.smoke else "asap_finetune.json")

    # One look at test: a finished run in this folder means test was scored.
    if not (args.smoke or args.dry_run or args.rescore_test) and os.path.exists(results_path):
        raise SystemExit(
            f"{results_path} already exists, so the test split has already been scored. Running again "
            "would score it a second time, which the pre-declaration rules out. Pass --rescore-test only "
            "for a deliberate re-run that will be disclosed.")

    log_path = os.path.join(out_dir, "finetune_log_smoke.txt" if args.smoke else "finetune_log.txt")
    open(log_path, "w").close()

    def log(message=""):
        print(message, flush=True)
        with open(log_path, "a") as f:
            f.write(message + "\n")

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir)) if args.data_dir else None
    paths = resolve_paths(data_dir)
    apply_paths(paths)

    device = pick_device()
    require_gpu(device, allow_no_gpu=args.allow_no_gpu, light=args.smoke or args.dry_run)
    env = environment(device)
    log("environment: " + ", ".join(f"{k} {v}" for k, v in env.items() if k != "script_sha256"))
    log("scripts: " + ", ".join(f"{k} {v}" for k, v in env["script_sha256"].items()))

    needed = ("essays", "split") if args.smoke else ("essays", "split", "published")
    missing = [f"{label}: {paths[label]}" for label in needed if not os.path.exists(paths[label])]
    if missing:
        raise SystemExit("PRE-FLIGHT FAILED: input files not found:\n  " + "\n  ".join(missing))

    df = bench.load_set1()
    split = bench.load_split()  # also refuses a file whose ids no longer match its fingerprint
    if split["fingerprint"] != LOCKED_SPLIT_FINGERPRINT:
        raise SystemExit(f"PRE-FLIGHT FAILED: split fingerprint {split['fingerprint']} is not the locked "
                         f"{LOCKED_SPLIT_FINGERPRINT}. Upload the asap_split.json committed in f3c5d17.")

    parts = {name: df[df.essay_id.isin(split[name])].reset_index(drop=True)
             for name in ("train", "dev", "test")}
    learning_rates, max_epochs = LEARNING_RATES, MAX_EPOCHS
    if args.smoke:
        # Never touch the real test split, even to smoke-test: a dev slice stands in.
        dev = parts["dev"]
        parts = {"train": parts["train"].iloc[:32].reset_index(drop=True),
                 "dev": dev.iloc[:24].reset_index(drop=True),
                 "test": dev.iloc[24:48].reset_index(drop=True)}
        learning_rates, max_epochs = LEARNING_RATES[:1], 1

    log(f"ASAP set 1, locked split {split['fingerprint']}: train {len(parts['train'])}, "
        f"dev {len(parts['dev'])}, test {len(parts['test'])}"
        + ("   [SMOKE: a dev slice stands in for test]" if args.smoke else ""))
    log(f"model {MODEL_NAME}, max_length {MAX_LENGTH}, batch {BATCH_SIZE}, seed {SEED}, device {device}")
    log(f"grid: lr {', '.join(f'{lr:g}' for lr in learning_rates)} x up to {max_epochs} epochs, "
        f"best by dev QWK; blend weights {BLEND_WEIGHTS} chosen on dev")

    y = {name: part.score.to_numpy(dtype=float) for name, part in parts.items()}
    hand = {name: bench.extract_features(list(part.text)) for name, part in parts.items()}

    # ---- 1. pre-flight ---------------------------------------------------
    log("\n=== pre-flight ===")
    if not args.smoke:
        for name in ("train", "test"):
            path = expected_cache_path(f"single_{name}", list(parts[name].text))
            if not os.path.exists(path):
                raise SystemExit(
                    f"PRE-FLIGHT FAILED: embedding cache file not found: {path}\n"
                    f"Upload {os.path.basename(path)} from backend/data/asap/.cache/ on the machine that ran "
                    "the benchmark. Recomputing it here, on a different device, would not reproduce the "
                    "published reference numbers.")
    ridge, reference_preds = reference_systems(parts, y, hand)
    if args.smoke:
        log("  smoke: reference systems refit on the slice, not checked against the published numbers")
    else:
        with open(paths["published"]) as f:
            published = json.load(f)
        for name, pred in reference_preds.items():
            got, want = bench.metrics(y["test"], pred)["qwk"], published["test"][name]["qwk"]
            if abs(got - want) > 1e-9:
                raise SystemExit(
                    f"PRE-FLIGHT FAILED: refit {name!r} gives test QWK {got:.6f}; the benchmark published "
                    f"{want:.6f}. No new candidate has been scored, so this experiment has not used the "
                    "test split. Check that embedding_cache/ holds the benchmark's files.")
            log(f"  {name}: test QWK {got:.3f}, identical to the published benchmark")
    log(f"  split is the locked one ({LOCKED_SPLIT_FINGERPRINT}); no new candidate has touched test")
    log("PRE-FLIGHT PASSED")
    if args.dry_run:
        log("--dry-run: stopping before training")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tensors = {name: tokenize(tokenizer, part.text) for name, part in parts.items()}
    total = sum(len(part) for part in parts.values())
    at_limit = sum(int((tensors[name][1].sum(dim=1) == MAX_LENGTH).sum()) for name in tensors)
    log(f"tokenised: {at_limit}/{total} essays fill the {MAX_LENGTH}-token window (anything beyond is cut)")

    # ---- 2. the grid, chosen on dev ---------------------------------------
    log("\n=== training grid (chosen on dev) ===")
    started = time.time()
    runs, best_run = {}, None
    for lr in learning_rates:
        log(f"  lr {lr:g}:")
        best, history = train_one(lr, max_epochs, tensors, y["train"], y["dev"], device, log)
        runs[lr] = {"history": history, "best_epoch": best["epoch"], "best_dev_qwk": best["qwk"]}
        if best_run is None or best["qwk"] > best_run["qwk"]:  # strict: ties keep the lower lr
            best_run = {**best, "lr": lr}
    grid_minutes = (time.time() - started) / 60
    log(f"\nbest on dev: lr {best_run['lr']:g}, epoch {best_run['epoch']}, dev QWK {best_run['qwk']:.3f}")
    log(f"grid time: {grid_minutes:.1f} min")

    model = new_model()
    model.load_state_dict(best_run["state"])
    model.to(device)

    # ---- 3. candidate B's blend weight, chosen on dev ----------------------
    ridge_dev_raw = ridge.predict(hand["dev"])
    finetuned_dev_raw = rescale(predict(model, *tensors["dev"], device=device))
    log("\n=== candidate B blend weight (dev only) ===")
    blend_scores = {}
    for weight in BLEND_WEIGHTS:
        blended = bench.to_scores(weight * finetuned_dev_raw + (1 - weight) * ridge_dev_raw)
        blend_scores[weight] = bench.metrics(y["dev"], blended)
        log(f"  w={weight}: dev QWK {blend_scores[weight]['qwk']:.3f}  dev MAE {blend_scores[weight]['mae']:.3f}")
    best_weight = max(blend_scores, key=lambda w: blend_scores[w]["qwk"])
    log(f"  -> chosen: w={best_weight}")

    # ---- 4. test, scored once ---------------------------------------------
    log("\n=== TEST (scored once) ===")
    started = time.time()
    finetuned_test_raw = rescale(predict(model, *tensors["test"], device=device))
    device_seconds = time.time() - started
    ridge_test_raw = ridge.predict(hand["test"])

    predictions = {
        "fine-tuned DistilBERT": bench.to_scores(finetuned_test_raw),
        "fine-tuned + handcrafted (blend)": bench.to_scores(
            best_weight * finetuned_test_raw + (1 - best_weight) * ridge_test_raw),
    }

    # Same resamples as the published benchmark, so the intervals line up.
    rng = np.random.default_rng(bench.BOOTSTRAP_SEED)
    idx = rng.integers(0, len(parts["test"]), size=(bench.BOOTSTRAP_N, len(parts["test"])))

    results, samples = {}, {}
    for name, pred in {**predictions, **reference_preds}.items():
        results[name] = bench.metrics(y["test"], pred)
        samples[name] = bench.bootstrap_metrics(y["test"], pred, idx)
        results[name]["ci"] = {k: bench.interval(v) for k, v in samples[name].items()}
    for name in predictions:
        r = results[name]
        log(f"  {name:34} MAE {bench.fmt(r['mae'], r['ci']['mae'])}  "
            f"QWK {bench.fmt(r['qwk'], r['ci']['qwk'])}  "
            f"exact {bench.fmt(r['exact'], r['ci']['exact'], True)}  "
            f"within-1 {bench.fmt(r['within_one'], r['ci']['within_one'], True)}")

    log("\n  paired differences (positive = fine-tuned better):")
    comparisons = {}
    for other in ("handcrafted only", "embeddings + handcrafted"):
        diff = samples["fine-tuned DistilBERT"]["qwk"] - samples[other]["qwk"]
        ci = bench.interval(diff)
        point = results["fine-tuned DistilBERT"]["qwk"] - results[other]["qwk"]
        distinguishable = bool(ci[0] is not None and (ci[0] > 0 or ci[1] < 0))
        comparisons[f"fine-tuned vs {other}"] = {"qwk_diff": float(point), "qwk_ci": ci,
                                                 "distinguishable": distinguishable}
        log(f"    fine-tuned vs {other}: QWK {point:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] -- "
            f"{'distinguishable' if distinguishable else 'NOT distinguishable'}")

    best_candidate = max(predictions, key=lambda n: results[n]["qwk"])
    fair = bench.single_rater_comparison(parts["test"], predictions[best_candidate], idx)
    log(f"\n  like-for-like, {best_candidate} vs a human, both predicting rater 1 (1-6):")
    log(f"    model:   QWK {bench.fmt(fair['model_vs_rater1']['qwk'], fair['model_vs_rater1']['ci'])}  "
        f"MAE {fair['model_vs_rater1']['mae']:.3f}")
    log(f"    rater 2: QWK {bench.fmt(fair['rater2_vs_rater1']['qwk'], fair['rater2_vs_rater1']['ci'])}  "
        f"MAE {fair['rater2_vs_rater1']['mae']:.3f}")
    lo, hi = fair["human_minus_model_ci"]
    log(f"    human advantage: {fair['human_minus_model_qwk']:+.3f} [{lo:+.3f}, {hi:+.3f}] -- "
        f"{'distinguishable' if fair['distinguishable'] else 'NOT distinguishable'}")

    parameters = sum(p.numel() for p in model.parameters())
    cost = {
        "grid_minutes": grid_minutes,
        "device": device.type,
        "device_ms_per_essay": device_seconds / len(parts["test"]) * 1000,
        "essays_at_token_limit": at_limit,
    }
    if device.type == "cuda":
        cost["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30

    payload = {
        "pre_declared_in": "training_data/asap_results.md (494bd77), amended for Colab before any test number existed",
        "smoke": bool(args.smoke),
        "model": MODEL_NAME,
        "parameters": int(parameters),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "split_fingerprint": split["fingerprint"],
        "environment": env,
        "grid": {f"{lr:g}": run for lr, run in runs.items()},
        "chosen": {"lr": best_run["lr"], "epoch": best_run["epoch"], "dev_qwk": best_run["qwk"],
                   "blend_weight": best_weight},
        "blend_dev_scores": {str(w): s for w, s in blend_scores.items()},
        "test": {name: results[name] for name in predictions},
        "reference_test": {name: results[name] for name in reference_preds},
        "comparisons": comparisons,
        "single_rater_comparison": {"candidate": best_candidate, **fair},
        "cost": cost,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(results_path, payload)  # saved now, so nothing below can lose the test results

    log("\n=== cost ===")
    log(f"  {parameters / 1e6:.1f}M parameters; grid {grid_minutes:.1f} min on {env.get('gpu', device.type)}; "
        f"inference {cost['device_ms_per_essay']:.0f} ms/essay on {device.type} in batches of {BATCH_SIZE}")
    if "peak_gpu_memory_gb" in cost:
        log(f"  peak GPU memory {cost['peak_gpu_memory_gb']:.1f} GB")
    try:
        cost.update(measure_cpu_cost(model, tokenizer, tensors["dev"], out_dir,
                                     keep_weights=not (args.smoke or args.no_save_weights), log=log))
    except Exception as exc:  # the test results are already on disk
        log(f"  CPU cost measurement failed ({type(exc).__name__}: {exc}); test results are unaffected")
    payload["cost"] = cost
    write_json(results_path, payload)
    log(f"\nSaved {results_path}")


if __name__ == "__main__":
    main()
