"""Regression probe for the Evidence grader's known weakness: rewarding what
evidence *looks* like (names, numbers, citation words) rather than whether it
supports the claim.

Usage (from backend/, with the venv active):
    ./venv/bin/python scripts/padding_probe.py [--json-out PATH] [--compare BEFORE.json]

Two families of paired probes, each pair differing in one controlled way:

  padding  base vague claim  vs  the same claim with names/numbers/years
           bolted on that support nothing. The gain is how many points the
           grader pays for decoration alone. Lower is better; 0 is ideal.

  tying    a statistic stated and left hanging  vs  the same statistic used
           to support a claim. The gap is whether the grader notices the
           connection. Higher is better; a gap near 0 means it is scoring
           the statistic, not the argument.

Every probe is written fresh for this script and checked against
sample_answers.json by the leakage guard, because a probe that duplicates a
training example measures memorisation instead of behaviour. --json-out saves
the scores; --compare diffs against a saved run, which is how a retrain is
checked for regression.

This never trains or saves a model; it only reads what is in ml_models/.
"""

import argparse
import json
import os
import sys

import numpy as np

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from app import create_app
from app.grading.engine import build_matrix
from app.grading.model_store import load_model
from app.models import Rubric

import leakage_guard

TRAIN_PATH = os.path.join(BACKEND_DIR, "training_data", "sample_answers.json")

# (id, kind, weak_variant, strong_variant). "weak" is the one that should score
# lower: the unpadded claim, or the untied statistic.
PROBES = [
    ("pad-1", "padding",
     "Social media causes problems for society and that is why people are worried about it.",
     "Social media causes problems for society and that is why people are worried about it. In 2016, 2019 and "
     "2021, groups in Norway, Chile, Kenya and Vietnam published 47 reports, and roughly 38% of those surveyed "
     "agreed."),
    ("pad-2", "padding",
     "These companies ought to be watched more closely because of the things they do.",
     "These companies ought to be watched more closely because of the things they do. Analysts at four "
     "universities reviewed 1,200 accounts across 9 countries in 2020, and a survey that year put the figure "
     "at 61%."),
    ("pad-3", "padding",
     "Young people are affected by the things they see on their phones every day.",
     "Young people are affected by the things they see on their phones every day. A 2022 study of 3,400 "
     "students in Portugal, Malaysia and Peru recorded 17 separate measures, and 44% of the sample shifted "
     "over 11 weeks."),
    ("pad-4", "padding",
     "Rules are needed here because the current situation is not working properly.",
     "Rules are needed here because the current situation is not working properly. Between 2014 and 2023, "
     "regulators in 28 countries opened 156 proceedings, and spending reached 4.2 billion dollars according "
     "to figures published in 2023."),

    ("tie-1", "tying",
     "A 2022 review of 2,100 accounts in Denmark found that 29% had been created within a single week.",
     "A 2022 review of 2,100 accounts in Denmark found that 29% had been created within a single week, which "
     "shows that coordinated sign-ups can be detected in bulk and therefore that platforms could act on them "
     "before they spread."),
    ("tie-2", "tying",
     "Regulators in Sweden issued 73 formal warnings to platforms during 2021.",
     "Regulators in Sweden issued 73 formal warnings to platforms during 2021, and because almost none led to "
     "a penalty, the case for warnings without enforcement powers is weak."),
    ("tie-3", "tying",
     "A survey of 5,600 parents in Ireland in 2023 reported that 52% had set no limits on screen time.",
     "A survey of 5,600 parents in Ireland in 2023 reported that 52% had set no limits on screen time, which "
     "supports the argument that leaving this to households alone does not work and that platforms must "
     "share the duty."),
]


def score_all(rubric, texts):
    """Every criterion's model applied to every probe text, exactly as
    engine.grade_text_submission does it (one featurisation, reused)."""
    X = build_matrix(texts)
    out = {}
    for criterion in rubric.criteria:
        payload = load_model(criterion.id)
        if payload is None:
            continue
        raw = np.asarray(payload["model"].predict(X), dtype=float)
        pred = np.clip(np.round(raw), 0, payload["max_points"])
        out[criterion.name] = {
            "max_points": payload["max_points"],
            "n_examples": payload.get("n_examples"),
            "pred": np.where(pred == 0, 0.0, pred),
        }
    return out


def check_probes_are_fresh(rubric_title):
    with open(TRAIN_PATH) as f:
        train_rows = json.load(f)
    candidates = []
    for probe_id, kind, weak, strong in PROBES:
        for variant, text in (("weak", weak), ("strong", strong)):
            candidates.append({"rubric": rubric_title, "criterion": "Evidence", "text": text,
                               "essay_id": f"{probe_id}/{variant}"})
    duplicates, notable = leakage_guard.find_matches(candidates, train_rows, same_criterion=True)
    if duplicates:
        print("PROBE LEAKAGE: these probes duplicate training examples, so they measure memorisation:\n")
        leakage_guard.print_matches(duplicates)
        raise SystemExit("Rewrite the probes before trusting this result.")
    print(f"Probe freshness: no probe duplicates a training example "
          f"({len(notable)} share topic/wording only, max cosine "
          f"{max((m['cosine'] for m in notable), default=0):.2f}).\n")


def main():
    parser = argparse.ArgumentParser(description="Paired padding/tying probes for the Evidence grader.")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--compare", default=None, help="a saved --json-out file to diff against")
    parser.add_argument("--rubric", default="Essay 1")
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        rubric = Rubric.query.filter_by(title=args.rubric).first()
        if not rubric:
            raise SystemExit(f"No rubric titled {args.rubric!r} exists.")

        check_probes_are_fresh(args.rubric)

        texts, index = [], {}
        for probe_id, kind, weak, strong in PROBES:
            index[probe_id] = (kind, len(texts), len(texts) + 1)
            texts += [weak, strong]

        scored = score_all(rubric, texts)
        models = {name: block["n_examples"] for name, block in scored.items()}
        print(f"Models trained on: " + ", ".join(f"{n} n={v}" for n, v in models.items()) + "\n")

        results = {}
        for name, block in scored.items():
            pred = block["pred"]
            rows = []
            for probe_id, (kind, i_weak, i_strong) in index.items():
                rows.append({"probe": probe_id, "kind": kind,
                             "weak": float(pred[i_weak]), "strong": float(pred[i_strong]),
                             "gap": float(pred[i_strong] - pred[i_weak])})
            results[name] = {"max_points": block["max_points"], "n_examples": block["n_examples"], "probes": rows}

            padding = [r for r in rows if r["kind"] == "padding"]
            tying = [r for r in rows if r["kind"] == "tying"]
            pad_mean = float(np.mean([r["gap"] for r in padding]))
            tie_mean = float(np.mean([r["gap"] for r in tying]))
            results[name]["padding_gain_mean"] = pad_mean
            results[name]["tying_gap_mean"] = tie_mean

            print(f"=== {name} (0-{block['max_points']:g}) ===")
            print(f"  padding: plain claim -> padded with names/numbers (lower gain is better)")
            print(f"    {'probe':<8}{'plain':<8}{'padded':<8}{'gain'}")
            for r in padding:
                print(f"    {r['probe']:<8}{r['weak']:<8.0f}{r['strong']:<8.0f}{r['gap']:+.0f}")
            print(f"    mean padding gain: {pad_mean:+.2f} points")
            print(f"  tying: statistic left hanging -> same statistic tied to a claim (higher gap is better)")
            print(f"    {'probe':<8}{'untied':<8}{'tied':<8}{'gap'}")
            for r in tying:
                print(f"    {r['probe']:<8}{r['weak']:<8.0f}{r['strong']:<8.0f}{r['gap']:+.0f}")
            print(f"    mean tying gap: {tie_mean:+.2f} points")
            print(f"    untied statistics alone score: {[int(r['weak']) for r in tying]} "
                  f"out of {block['max_points']:g}\n")

        if args.compare:
            with open(args.compare) as f:
                before = json.load(f)
            print("=== BEFORE -> AFTER ===")
            for name, block in results.items():
                prior = before["criteria"].get(name)
                if not prior:
                    continue
                prior_probes = {r["probe"]: r for r in prior["probes"]}
                print(f"  {name}: mean padding gain {prior['padding_gain_mean']:+.2f} -> "
                      f"{block['padding_gain_mean']:+.2f}   |   mean tying gap "
                      f"{prior['tying_gap_mean']:+.2f} -> {block['tying_gap_mean']:+.2f}")
                print(f"    {'probe':<8}{'kind':<10}{'before':<20}{'after':<20}{'gap change'}")
                for r in block["probes"]:
                    p = prior_probes.get(r["probe"])
                    if not p:
                        continue
                    print(f"    {r['probe']:<8}{r['kind']:<10}"
                          f"{p['weak']:.0f} -> {p['strong']:.0f} ({p['gap']:+.0f}){' ' * 8}"
                          f"{r['weak']:.0f} -> {r['strong']:.0f} ({r['gap']:+.0f}){' ' * 8}"
                          f"{r['gap'] - p['gap']:+.0f}")
                print()

        if args.json_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
            # Provenance is the *models*, not the file on disk: the file can be
            # appended to before a run, which would label a before-run with the
            # after-run's count. n_examples per criterion comes from the model.
            payload = {"models_trained_on": {n: b["n_examples"] for n, b in results.items()},
                       "criteria": results}
            with open(args.json_out, "w") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
            print(f"Saved {args.json_out}")


if __name__ == "__main__":
    main()
