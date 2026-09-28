"""Duplicate check between two sets of labeled rows, used in both directions:
held-out rows against the training data (evaluate_holdout.py), and candidate
new training rows against the held-out files before they are added.

Usage (from backend/, with the venv active):
    ./venv/bin/python scripts/leakage_guard.py CANDIDATE.json --against A.json [B.json ...] [--any-criterion]

Each candidate row is compared with every reference row two ways:

1. Text similarity (difflib.SequenceMatcher on lowercased, punctuation-
   stripped text). Catches a copy-paste or a lightly edited duplicate.
2. Embedding similarity (cosine between frozen SBERT vectors -- the feature
   space the models are actually fit on). Catches a paraphrase that reads
   differently but lands in nearly the same place for the model, which text
   similarity misses entirely.

A hard match on either (text ratio >= DUPLICATE_TEXT_THRESHOLD or cosine >=
DUPLICATE_EMBED_THRESHOLD) is a duplicate: the caller stops. A softer
embedding match (>= NOTABLE_EMBED_THRESHOLD) is only disclosed. Two essays on
the same prompt will reuse the same well-known facts (the same statistic, the
same law) without being duplicates of each other, so that is reported rather
than treated as disqualifying.

Exit status is 1 if any duplicate is found, so it can gate a pipeline.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.grading.embedder import embed

DUPLICATE_TEXT_THRESHOLD = 0.85
DUPLICATE_EMBED_THRESHOLD = 0.90
NOTABLE_EMBED_THRESHOLD = 0.60


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def _unit(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def find_matches(candidates, references, same_criterion=True):
    """Returns (duplicates, notable), each a list of
    {"candidate": row, "reference": row, "ratio": float, "cosine": float}.

    same_criterion=True only compares rows scored on the same criterion, which
    is what the evaluation wants (a Thesis holdout row against Thesis training
    rows). False compares every row with every row, which is what a text-level
    duplicate check wants.
    """
    groups = defaultdict(list)
    for row in references:
        groups[row["criterion"] if same_criterion else None].append(row)
    group_vectors = {key: _unit(embed([r["text"] for r in rows])) for key, rows in groups.items()}
    candidate_vectors = _unit(embed([c["text"] for c in candidates]))

    duplicates, notable = [], []
    for row, vector in zip(candidates, candidate_vectors):
        key = row["criterion"] if same_criterion else None
        pool = groups.get(key, [])
        if not pool:
            continue

        cosines = group_vectors[key] @ vector
        normalised = normalize(row["text"])
        ratios = np.array([SequenceMatcher(None, normalised, normalize(r["text"])).ratio() for r in pool])

        best_ratio, best_cosine = int(np.argmax(ratios)), int(np.argmax(cosines))

        def match(i):
            return {"candidate": row, "reference": pool[i], "ratio": float(ratios[i]), "cosine": float(cosines[i])}

        if ratios[best_ratio] >= DUPLICATE_TEXT_THRESHOLD:
            duplicates.append(match(best_ratio))
        elif cosines[best_cosine] >= DUPLICATE_EMBED_THRESHOLD:
            duplicates.append(match(best_cosine))
        elif cosines[best_cosine] >= NOTABLE_EMBED_THRESHOLD:
            notable.append(match(best_cosine))

    return duplicates, notable


def label(row):
    tag = row.get("essay_id")
    source = row.get("_source")
    parts = [f"{source}:" if source else "", f"essay {tag} /" if tag is not None else "", row["criterion"]]
    return " ".join(p for p in parts if p)


def print_matches(matches, indent="  "):
    for m in sorted(matches, key=lambda m: -m["cosine"]):
        print(f"{indent}{label(m['candidate'])}: cosine {m['cosine']:.2f}, text ratio {m['ratio']:.2f}")
        print(f"{indent}  candidate: {m['candidate']['text'][:110]}")
        print(f"{indent}  {label(m['reference'])}: {m['reference']['text'][:110]}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("candidate", help="JSON file of rows to check")
    parser.add_argument("--against", nargs="+", required=True, help="JSON file(s) of reference rows")
    parser.add_argument("--any-criterion", action="store_true",
                        help="compare across criteria (candidate rows are de-duplicated by text first)")
    args = parser.parse_args()

    with open(args.candidate) as f:
        candidates = json.load(f)
    references = []
    for path in args.against:
        with open(path) as f:
            for row in json.load(f):
                references.append({**row, "_source": os.path.basename(path)})

    if args.any_criterion:
        seen, unique = set(), []
        for row in candidates:
            if row["text"] not in seen:
                seen.add(row["text"])
                unique.append(row)
        candidates = unique
        # Same text under two criteria is the same essay; keep one reference per text too.
        seen, unique = set(), []
        for row in references:
            if row["text"] not in seen:
                seen.add(row["text"])
                unique.append(row)
        references = unique

    print(f"Checking {len(candidates)} candidate rows against {len(references)} reference rows "
          f"({', '.join(os.path.basename(p) for p in args.against)})")
    print(f"Duplicate = text ratio >= {DUPLICATE_TEXT_THRESHOLD} or cosine >= {DUPLICATE_EMBED_THRESHOLD}; "
          f"disclosed = cosine >= {NOTABLE_EMBED_THRESHOLD}\n")

    duplicates, notable = find_matches(candidates, references, same_criterion=not args.any_criterion)

    if duplicates:
        print("DUPLICATES FOUND -- stop:\n")
        print_matches(duplicates)
    else:
        print("No duplicates or near-duplicates.\n")
    if notable:
        print(f"{len(notable)} weaker overlaps (shared real-world facts or wording, not duplicates):\n")
        print_matches(notable)
    sys.exit(1 if duplicates else 0)


if __name__ == "__main__":
    main()
