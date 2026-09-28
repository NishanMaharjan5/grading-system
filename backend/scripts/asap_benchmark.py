"""Benchmarks this project's featurisation + Ridge pipeline on ASAP-AES set 1,
a public corpus of 1,783 persuasive essays scored 2-12 by two human raters.

Usage (from backend/, with the venv active):
    ./venv/bin/python scripts/asap_benchmark.py --write-split    # step 1: lock the split
    ./venv/bin/python scripts/asap_benchmark.py                  # step 2: run the experiment

Why this exists: every number this project has reported so far comes from
essays the project team wrote and scored itself, on one prompt. This measures
the same featurisation against scores real teachers gave, on text nobody here
wrote. It validates *the method*; it says nothing about the Essay 1 rubric.

The dataset is licensed and is never committed (see .gitignore). Only the
split file and the results are.

Isolation
---------
This script touches no database, creates no Flask app, and does not read or
write ml_models/. It cannot: `import app.grading.features` would execute
`app/__init__.py`, which imports Flask and SQLAlchemy, so the two featurisation
modules are loaded directly from their files instead. They are the same source
files the live grader uses, and tests/test_asap_benchmark.py asserts this
script's feature matrix is identical to engine.build_matrix's, so the two
cannot drift apart.

Discipline
----------
The split is written and committed before any model is fit. Every choice --
notably single-pass vs. chunked embeddings -- is made on dev. Test is scored
once, at the end. Ridge alpha stays at the production value (1.0) rather than
being tuned here, because the question is how the shipped pipeline does, not
how well it could be made to do on ASAP.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BACKEND_DIR, "data", "asap", "essays.xlsx")
CACHE_DIR = os.path.join(BACKEND_DIR, "data", "asap", ".cache")
SPLIT_PATH = os.path.join(BACKEND_DIR, "training_data", "asap_split.json")
RESULTS_PATH = os.path.join(BACKEND_DIR, "training_data", "asap_results.md")

ESSAY_SET = 1
SCORE_MIN, SCORE_MAX = 2, 12
SPLIT_SEED = 20260928
BOOTSTRAP_SEED = 0
BOOTSTRAP_N = 2000
RIDGE_ALPHA = 1.0  # the production value; deliberately not tuned for ASAP
RARE_STRATUM_MIN = 20  # scores rarer than this share one stratification bucket


def _load_module(name, relative_path):
    """Load a module straight from its file, bypassing the `app` package
    __init__ (which pulls in Flask and SQLAlchemy)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(BACKEND_DIR, relative_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_embedder = _load_module("_asap_embedder", "app/grading/embedder.py")
_features = _load_module("_asap_features", "app/grading/features.py")

embed = _embedder.embed
extract_features = _features.extract
FEATURE_NAMES = _features.FEATURE_NAMES


def build_matrix(texts, embeddings=None):
    """Identical to app.grading.engine.build_matrix. Accepts precomputed
    embeddings so the chunked variant can be swapped in without re-embedding."""
    vectors = embed(texts) if embeddings is None else embeddings
    return np.hstack([vectors, extract_features(texts)])


# ------------------------------------------------------------------ cleaning

# ASAP redacts named entities as @TYPE<n>. Each becomes a neutral lowercase
# phrase: dropping them outright breaks sentences, and leaving them in feeds
# the hand-crafted features junk -- "@NUM1" contains a digit, so digit_count
# would reward an essay for having been anonymised.
PLACEHOLDER_REPLACEMENTS = {
    "PERSON": "someone",
    "ORGANIZATION": "an organization",
    "LOCATION": "a place",
    "CITY": "a city",
    "STATE": "a state",
    "DATE": "a date",
    "MONTH": "a month",
    "TIME": "a time",
    "NUM": "a number",
    "MONEY": "an amount",
    "PERCENT": "a percentage",
    "EMAIL": "an email address",
    "DR": "a doctor",
    "CAPS": "",  # a redacted capitalised word; no neutral noun fits, so it goes
}
PLACEHOLDER_RE = re.compile(r"@([A-Z]+)(\d*)")

MOJIBAKE = {
    "â€™": "'", "â€œ": '"', "â€\u009d": '"',
    "â€˜": "'", "â€“": "-", "â€”": "-",
    "â€¦": "...",
}
SMART_CHARS = {"’": "'", "‘": "'", "“": '"', "”": '"',
               "–": "-", "—": "-", "…": "..."}


def clean_text(text):
    """Returns (cleaned_text, n_placeholders_replaced)."""
    for bad, good in MOJIBAKE.items():
        text = text.replace(bad, good)
    for bad, good in SMART_CHARS.items():
        text = text.replace(bad, good)

    count = 0

    def replace(match):
        nonlocal count
        count += 1
        return PLACEHOLDER_REPLACEMENTS.get(match.group(1), "")

    text = PLACEHOLDER_RE.sub(replace, text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)  # space before punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text, count


# ------------------------------------------------------------------ data

def load_set1():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            f"{DATA_PATH} not found. This dataset is licensed and is not in the repo; "
            "download ASAP-AES and place essays.xlsx there."
        )
    df = pd.read_excel(DATA_PATH, sheet_name=0)
    df = df[df.essay_set == ESSAY_SET].copy()
    df = df[df.domain1_score.notna()]

    cleaned, counts = zip(*(clean_text(t) for t in df.essay))
    df["text"] = cleaned
    df["n_placeholders"] = counts
    df["score"] = df.domain1_score.astype(int)
    df["rater1"] = df.rater1_domain1.astype(int)
    df["rater2"] = df.rater2_domain1.astype(int)
    return df[["essay_id", "text", "score", "rater1", "rater2", "n_placeholders"]].reset_index(drop=True)


MIN_STRATUM = 4  # a stratum must survive two successive splits (70/30, then 50/50)


def stratification_key(scores):
    """Scores with very few essays share one bucket, so a 3-way stratified
    split is possible at all (set 1 has a score with a single essay).

    If the pooled rare bucket is itself too small to survive both splits, it
    is folded into the most common score rather than crashing: stratifying
    "where possible" beats refusing to split.
    """
    counts = scores.value_counts()
    key = scores.map(lambda s: "rare" if counts[s] < RARE_STRATUM_MIN else str(s))
    key_counts = key.value_counts()
    if key_counts.get("rare", 0) and key_counts["rare"] < MIN_STRATUM:
        biggest = str(counts.idxmax())
        key = key.replace("rare", biggest)
    return key


def make_split(essay_ids, scores, seed=SPLIT_SEED):
    """Deterministic 70/15/15 split, stratified where the data allows."""
    key = stratification_key(scores)
    train_ids, temp_ids, _, temp_key = train_test_split(
        essay_ids, key, train_size=0.70, random_state=seed, stratify=key, shuffle=True
    )
    dev_ids, test_ids = train_test_split(
        temp_ids, train_size=0.50, random_state=seed, stratify=temp_key, shuffle=True
    )
    return {k: sorted(int(i) for i in v) for k, v in
            (("train", train_ids), ("dev", dev_ids), ("test", test_ids))}


def split_fingerprint(split):
    payload = json.dumps({k: split[k] for k in ("train", "dev", "test")}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write_split():
    df = load_set1()
    split = make_split(df.essay_id, df.score)
    assert not (set(split["train"]) & set(split["dev"]) & set(split["test"]))
    assert len(split["train"]) + len(split["dev"]) + len(split["test"]) == len(df)

    by_id = dict(zip(df.essay_id, df.score))
    payload = {
        "dataset": "ASAP-AES (Hewlett Foundation, Kaggle 2012), essay_set 1",
        "n_essays": len(df),
        "seed": SPLIT_SEED,
        "proportions": {"train": 0.70, "dev": 0.15, "test": 0.15},
        "stratified_by": f"domain1_score; scores with fewer than {RARE_STRATUM_MIN} essays share a 'rare' bucket",
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Committed before any model was fit. Test is scored once, at the end.",
        "fingerprint": split_fingerprint(split),
        "counts": {k: len(v) for k, v in split.items()},
        "score_distribution": {
            k: {str(s): int(sum(1 for i in v if by_id[i] == s)) for s in sorted(set(by_id.values()))}
            for k, v in split.items()
        },
        **split,
    }
    with open(SPLIT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote {SPLIT_PATH}")
    print(f"  counts: {payload['counts']}   fingerprint {payload['fingerprint']}")
    return payload


def load_split():
    if not os.path.exists(SPLIT_PATH):
        raise SystemExit(f"{SPLIT_PATH} not found -- run with --write-split and commit it first.")
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    if split_fingerprint(split) != split["fingerprint"]:
        raise SystemExit("The split file's ids do not match its fingerprint; it has been edited.")
    return split


# ------------------------------------------------------------------ embedding

def token_lengths(texts):
    tokenizer = _embedder.get_embedder().tokenizer
    return np.array([len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts])


def chunk_words(text, words_per_chunk):
    words = text.split()
    if not words:
        return [text]
    return [" ".join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]


def embed_chunked(texts, words_per_chunk=170):
    """Split each essay into chunks that fit the encoder, embed every chunk,
    mean-pool per essay. 170 words is conservative: set 1 runs about 1.15
    tokens per word, so a chunk lands near 195 of the 256 available."""
    chunks, owner = [], []
    for i, text in enumerate(texts):
        pieces = chunk_words(text, words_per_chunk)
        chunks.extend(pieces)
        owner.extend([i] * len(pieces))

    vectors = embed(chunks)
    owner = np.array(owner)
    pooled = np.vstack([vectors[owner == i].mean(axis=0) for i in range(len(texts))])
    return pooled, len(chunks)


def cached_embeddings(name, texts, compute):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.sha256(("||".join(texts) + name).encode()).hexdigest()[:20]
    path = os.path.join(CACHE_DIR, f"{name}_{key}.npy")
    if os.path.exists(path):
        return np.load(path)
    vectors = compute()
    np.save(path, vectors)
    return vectors


# ------------------------------------------------------------------ metrics

def qwk(y_true, y_pred, labels):
    """Quadratic weighted kappa over an explicit label range, so a system that
    never predicts an extreme score is still measured against the full scale."""
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


LABELS = list(range(SCORE_MIN, SCORE_MAX + 1))


def to_scores(raw):
    return np.clip(np.round(np.asarray(raw, dtype=float)), SCORE_MIN, SCORE_MAX)


def metrics(y_true, y_pred, labels=LABELS):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return {
        "mae": float(np.abs(y_pred - y_true).mean()),
        "exact": float(np.mean(y_pred == y_true)),
        "within_one": float(np.mean(np.abs(y_pred - y_true) <= 1)),
        "qwk": qwk(y_true, y_pred, labels),
    }


def bootstrap_metrics(y_true, y_pred, idx, labels=LABELS):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_pred - y_true)
    return {
        "mae": abs_err[idx].mean(axis=1),
        "exact": (y_pred == y_true)[idx].mean(axis=1),
        "within_one": (abs_err <= 1)[idx].mean(axis=1),
        "qwk": np.array([qwk(y_true[r], y_pred[r], labels) for r in idx]),
    }


def interval(samples):
    valid = np.asarray(samples)[~np.isnan(samples)]
    if valid.size == 0:
        return [None, None]
    return [float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))]


def make_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))


def fmt(value, ci, as_pct=False):
    show = (lambda v: f"{v:.0%}") if as_pct else (lambda v: f"{v:.3f}")
    if ci is None or ci[0] is None:
        return show(value)
    return f"{show(value)} [{show(ci[0])}, {show(ci[1])}]"


# ------------------------------------------------------------------ systems

def feature_sets(texts, embeddings):
    """The five systems, from trivial to the full shipped pipeline."""
    hand = extract_features(texts)
    word_count = hand[:, [FEATURE_NAMES.index("word_count")]]
    return {
        "guess-the-mean": None,
        "word count only": word_count,
        "handcrafted only": hand,
        "embeddings only": embeddings,
        "embeddings + handcrafted": np.hstack([embeddings, hand]),
    }


def fit_predict(X_train, y_train, X_eval):
    if X_train is None:  # guess-the-mean
        return np.full(len(X_eval), float(np.mean(y_train)))
    model = make_model().fit(X_train, y_train)
    return model.predict(X_eval)


def run_systems(train, evaluate, train_emb, eval_emb):
    """Fits every system on train, predicts on evaluate. Returns name ->
    rounded, clamped predictions."""
    train_features = feature_sets(list(train.text), train_emb)
    eval_features = feature_sets(list(evaluate.text), eval_emb)
    y_train = train.score.to_numpy(dtype=float)

    predictions = {}
    for name in train_features:
        X_train, X_eval = train_features[name], eval_features[name]
        if name == "guess-the-mean":
            raw = np.full(len(evaluate), float(y_train.mean()))
        else:
            raw = fit_predict(X_train, y_train, X_eval)
        predictions[name] = to_scores(raw)
    return predictions


def human_ceiling(df, idx):
    """Two readings of rater-vs-rater agreement.

    Each rater scores 1-6 and the final score is their sum, so the raters'
    native scale is not the model's. The 1-6 figure is the one the ASAP
    literature quotes; the doubled figure puts both raters on the model's 2-12
    range so the numbers sit on one axis, at the cost of only ever landing on
    even scores. Neither is a like-for-like ceiling, and both are stated
    rather than quietly blended.
    """
    r1, r2 = df.rater1.to_numpy(float), df.rater2.to_numpy(float)
    native_labels = list(range(1, 7))
    out = {
        "native": {
            "point": qwk(r1, r2, native_labels),
            "samples": np.array([qwk(r1[r], r2[r], native_labels) for r in idx]),
            "exact": float(np.mean(r1 == r2)),
            "within_one": float(np.mean(np.abs(r1 - r2) <= 1)),
        },
        "doubled": {
            "point": qwk(r1 * 2, r2 * 2, LABELS),
            "samples": np.array([qwk(r1[r] * 2, r2[r] * 2, LABELS) for r in idx]),
            "mae": float(np.abs(r1 * 2 - r2 * 2).mean()),
        },
    }
    for block in out.values():
        block["ci"] = interval(block["samples"])
    return out


def single_rater_comparison(df, model_pred, idx):
    """A like-for-like version of the ceiling.

    The headline model number is scored against domain1_score, which is two
    raters added together and so is smoother than either rater alone -- an
    easier target than what a single human faces. Comparing that with
    rater-vs-rater agreement would flatter the model.

    This puts both on the same task: predict what rater 1 said. The model's
    2-12 prediction is halved onto the raters' 1-6 scale, and rater 2 is the
    human doing the same job. Same target, same scale, same essays.
    """
    r1, r2 = df.rater1.to_numpy(float), df.rater2.to_numpy(float)
    labels = list(range(1, 7))
    model_1to6 = np.clip(np.round(np.asarray(model_pred, dtype=float) / 2.0), 1, 6)

    model_samples = np.array([qwk(model_1to6[r], r1[r], labels) for r in idx])
    human_samples = np.array([qwk(r2[r], r1[r], labels) for r in idx])
    diff = human_samples - model_samples
    ci = interval(diff)
    return {
        "model_vs_rater1": {"qwk": qwk(model_1to6, r1, labels), "ci": interval(model_samples),
                            "mae": float(np.abs(model_1to6 - r1).mean())},
        "rater2_vs_rater1": {"qwk": qwk(r2, r1, labels), "ci": interval(human_samples),
                             "mae": float(np.abs(r2 - r1).mean())},
        "human_minus_model_qwk": float(qwk(r2, r1, labels) - qwk(model_1to6, r1, labels)),
        "human_minus_model_ci": ci,
        "distinguishable": bool(ci[0] is not None and (ci[0] > 0 or ci[1] < 0)),
    }


def alpha_sensitivity_on_dev(train, dev, train_emb, dev_emb):
    """Dev-only diagnostic: is 'embeddings are weak' an artefact of using the
    production Ridge alpha? Sweeps alpha for the embeddings-only system. Never
    touches test and never changes what is reported there."""
    y_train, y_dev = train.score.to_numpy(float), dev.score.to_numpy(float)
    out = {}
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(train_emb, y_train)
        pred = to_scores(model.predict(dev_emb))
        out[alpha] = metrics(y_dev, pred)
    return out


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description="Benchmark the shipped featurisation on ASAP-AES set 1.")
    parser.add_argument("--write-split", action="store_true",
                        help="write the train/dev/test split and exit, before anything is fit")
    parser.add_argument("--json-out", default=os.path.join(BACKEND_DIR, "training_data", "results", "asap.json"))
    args = parser.parse_args()

    if args.write_split:
        write_split()
        return

    df = load_set1()
    split = load_split()
    print(f"ASAP-AES set {ESSAY_SET}: {len(df)} essays, scores {SCORE_MIN}-{SCORE_MAX}")
    print(f"Split fingerprint {split['fingerprint']} (train {split['counts']['train']}, "
          f"dev {split['counts']['dev']}, test {split['counts']['test']})\n")

    affected = int((df.n_placeholders > 0).sum())
    print(f"Cleaning: {affected}/{len(df)} essays ({affected / len(df):.1%}) contained anonymisation "
          f"placeholders; {int(df.n_placeholders.sum()):,} occurrences replaced.\n")

    parts = {name: df[df.essay_id.isin(split[name])].reset_index(drop=True)
             for name in ("train", "dev", "test")}

    # ---- truncation check -------------------------------------------------
    encoder = _embedder.get_embedder()
    limit = encoder.max_seq_length
    lengths = token_lengths(list(df.text))
    over = int((lengths > limit).sum())
    print(f"Truncation: encoder max_seq_length is {limit} tokens. "
          f"{over}/{len(df)} essays ({over / len(df):.1%}) exceed it.")
    print(f"  token length: median {np.median(lengths):.0f}, mean {lengths.mean():.0f}, "
          f"p90 {np.percentile(lengths, 90):.0f}, max {lengths.max()}")
    print(f"  the median essay loses about {max(0, np.median(lengths) - limit):.0f} tokens "
          f"({max(0, 1 - limit / np.median(lengths)):.0%} of itself) to silent truncation.\n")

    texts = {name: list(part.text) for name, part in parts.items()}
    single = {name: cached_embeddings(f"single_{name}", t, lambda t=t: embed(t)) for name, t in texts.items()}
    chunked, chunk_counts = {}, {}

    for name, t in texts.items():
        chunked[name] = cached_embeddings(f"chunked_{name}", t, lambda t=t: embed_chunked(t)[0])
        # Counted from the texts, not from the embedding call, so the figure is
        # right whether or not the cache was used.
        chunk_counts[name] = sum(len(chunk_words(x, 170)) for x in t)

    # ---- dev: choose the embedding variant --------------------------------
    print("=== DEV: single-pass vs chunked embeddings (this is the only choice made) ===")
    dev_choice = {}
    for variant, vectors in (("single-pass", single), ("chunked", chunked)):
        predictions = run_systems(parts["train"], parts["dev"], vectors["train"], vectors["dev"])
        full = predictions["embeddings + handcrafted"]
        m = metrics(parts["dev"].score, full)
        dev_choice[variant] = m
        print(f"  {variant:12} embeddings+handcrafted on dev: "
              f"MAE {m['mae']:.3f}  QWK {m['qwk']:.3f}  exact {m['exact']:.0%}  within-1 {m['within_one']:.0%}")

    chosen = "chunked" if dev_choice["chunked"]["qwk"] > dev_choice["single-pass"]["qwk"] else "single-pass"
    chosen_vectors = chunked if chosen == "chunked" else single
    print(f"  -> chosen on dev QWK: {chosen}\n")

    # ---- test: scored once ------------------------------------------------
    print(f"=== TEST (scored once, {chosen} embeddings) ===")
    test = parts["test"]
    predictions = run_systems(parts["train"], test, chosen_vectors["train"], chosen_vectors["test"])
    y_true = test.score.to_numpy(dtype=float)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(test), size=(BOOTSTRAP_N, len(test)))

    results, samples = {}, {}
    for name, pred in predictions.items():
        results[name] = metrics(y_true, pred)
        samples[name] = bootstrap_metrics(y_true, pred, idx)
        results[name]["ci"] = {k: interval(v) for k, v in samples[name].items()}
        r = results[name]
        print(f"  {name:26} MAE {fmt(r['mae'], r['ci']['mae'])}  "
              f"QWK {fmt(r['qwk'], r['ci']['qwk'])}  "
              f"exact {fmt(r['exact'], r['ci']['exact'], True)}  "
              f"within-1 {fmt(r['within_one'], r['ci']['within_one'], True)}")

    # ---- paired differences ----------------------------------------------
    print("\n  paired differences (positive = first system better):")
    pairs = [
        ("embeddings + handcrafted", "embeddings only"),
        ("embeddings + handcrafted", "handcrafted only"),
        ("embeddings only", "handcrafted only"),
        ("handcrafted only", "word count only"),
        ("word count only", "guess-the-mean"),
    ]
    comparisons = {}
    for a, b in pairs:
        qwk_diff = samples[a]["qwk"] - samples[b]["qwk"]
        mae_diff = samples[b]["mae"] - samples[a]["mae"]  # lower MAE is better
        ci_q, ci_m = interval(qwk_diff), interval(mae_diff)
        distinguishable = ci_q[0] is not None and (ci_q[0] > 0 or ci_q[1] < 0)
        comparisons[f"{a} vs {b}"] = {
            "qwk_diff": float(results[a]["qwk"] - results[b]["qwk"]), "qwk_ci": ci_q,
            "mae_diff": float(results[b]["mae"] - results[a]["mae"]), "mae_ci": ci_m,
            "qwk_distinguishable": bool(distinguishable),
        }
        verdict = "distinguishable" if distinguishable else "NOT distinguishable"
        print(f"    {a} vs {b}:")
        print(f"      QWK {results[a]['qwk'] - results[b]['qwk']:+.3f} [{ci_q[0]:+.3f}, {ci_q[1]:+.3f}] -- {verdict}")
        print(f"      MAE {results[b]['mae'] - results[a]['mae']:+.3f} [{ci_m[0]:+.3f}, {ci_m[1]:+.3f}]")

    # ---- human ceiling ----------------------------------------------------
    print("\n=== HUMAN CEILING (rater 1 vs rater 2) ===")
    ceiling_test = human_ceiling(test, idx)
    all_idx = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(df), size=(BOOTSTRAP_N, len(df)))
    ceiling_all = human_ceiling(df, all_idx)
    for label, block in (("test essays", ceiling_test), ("all set-1 essays", ceiling_all)):
        n = ceiling_native = block["native"]
        print(f"  {label}:")
        print(f"    raters on their native 1-6 scale: QWK {fmt(n['point'], n['ci'])}  "
              f"exact {n['exact']:.0%}  within-1 {n['within_one']:.0%}")
        d = block["doubled"]
        print(f"    both doubled onto the 2-12 scale:  QWK {fmt(d['point'], d['ci'])}  "
              f"MAE {d['mae']:.3f}   (QWK is scale-invariant, so this matches by construction)")

    print("\n  like-for-like -- both predicting what rater 1 said, on the raters' own 1-6 scale:")
    fair = single_rater_comparison(test, predictions["embeddings + handcrafted"], idx)
    m, h = fair["model_vs_rater1"], fair["rater2_vs_rater1"]
    print(f"    model (halved to 1-6) vs rater 1: QWK {fmt(m['qwk'], m['ci'])}  MAE {m['mae']:.3f}")
    print(f"    rater 2              vs rater 1: QWK {fmt(h['qwk'], h['ci'])}  MAE {h['mae']:.3f}")
    lo, hi = fair["human_minus_model_ci"]
    print(f"    human advantage: QWK {fair['human_minus_model_qwk']:+.3f} [{lo:+.3f}, {hi:+.3f}] -- "
          f"{'distinguishable' if fair['distinguishable'] else 'NOT distinguishable'}")

    print("\n=== DEV-ONLY DIAGNOSTIC: is the weak embeddings result just the alpha? ===")
    alpha_sweep = alpha_sensitivity_on_dev(parts["train"], parts["dev"],
                                           chosen_vectors["train"], chosen_vectors["dev"])
    for alpha, m in alpha_sweep.items():
        marker = "  <- production value" if alpha == RIDGE_ALPHA else ""
        print(f"    embeddings-only, alpha {alpha:>7}: dev MAE {m['mae']:.3f}  QWK {m['qwk']:.3f}{marker}")

    payload = {
        "dataset": f"ASAP-AES set {ESSAY_SET}",
        "n_essays": len(df),
        "split_fingerprint": split["fingerprint"],
        "counts": split["counts"],
        "cleaning": {"essays_affected": affected, "occurrences": int(df.n_placeholders.sum())},
        "truncation": {"max_seq_length": int(limit), "essays_over_limit": over,
                       "fraction_over": over / len(df), "median_tokens": float(np.median(lengths)),
                       "mean_tokens": float(lengths.mean()), "max_tokens": int(lengths.max())},
        "dev_variant_choice": {"scores": {k: {m: v for m, v in val.items()} for k, val in dev_choice.items()},
                               "chosen": chosen},
        "chunk_counts": chunk_counts,
        "test": {name: {k: v for k, v in r.items()} for name, r in results.items()},
        "comparisons": comparisons,
        "human_ceiling": {
            label: {"native_qwk": b["native"]["point"], "native_ci": b["native"]["ci"],
                    "native_exact": b["native"]["exact"], "native_within_one": b["native"]["within_one"],
                    "doubled_qwk": b["doubled"]["point"], "doubled_ci": b["doubled"]["ci"],
                    "doubled_mae": b["doubled"]["mae"]}
            for label, b in (("test", ceiling_test), ("all_set1", ceiling_all))
        },
        "single_rater_comparison": fair,
        "dev_alpha_sweep": {str(a): m for a, m in alpha_sweep.items()},
        "config": {"ridge_alpha": RIDGE_ALPHA, "split_seed": SPLIT_SEED,
                   "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED}},
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\nSaved {args.json_out}")


if __name__ == "__main__":
    main()
