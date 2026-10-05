"""Reformats sample_answers.json into the (essay, criterion, normalised score)
triples the rubric-conditioning experiment trains on.

    ./venv/bin/python scripts/build_rubric_conditioning_data.py

Reads training_data/sample_answers.json and writes
training_data/rubric_conditioning_train.json. It never writes back to
sample_answers.json: that file feeds the shipped per-criterion graders, and
this reformatting exists only for one experiment.

Why normalise the score: a single model scores criteria with different ranges
(Thesis 0-5, Evidence 0-10, Counterargument 0-5). Training on raw points would
teach it that Evidence answers are worth roughly twice as much as Thesis ones,
which is an artefact of the scale rather than anything about the writing.
Dividing by max_points puts every criterion on 0-1; inference multiplies back
by the target criterion's own maximum.

The canonical descriptions below are the conditioning signal. They are written
as questions a marker would ask, in the same voice as the held-out
Counterargument description, so that a model which has learned to read the
description has something transferable to read.
"""

import json
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(BACKEND_DIR, "training_data", "sample_answers.json")
TARGET = os.path.join(BACKEND_DIR, "training_data", "rubric_conditioning_train.json")

# Pre-declared in results/rubric_conditioning_results.md. Changing these after
# a run would invalidate the comparison, so they live in one place.
CRITERIA = {
    "Thesis": {
        "description": "Does the essay take a clear, specific, arguable position?",
        "max_points": 5,
    },
    "Evidence": {
        "description": "Does the essay support its position with specific, relevant evidence "
                       "tied to the argument?",
        "max_points": 10,
    },
}


def build(rows):
    """Returns (triples, skipped) -- skipped lists rows for criteria this
    experiment does not cover, so nothing is dropped silently."""
    triples, skipped = [], []
    for row in rows:
        spec = CRITERIA.get(row["criterion"])
        if spec is None:
            skipped.append(row["criterion"])
            continue
        score = float(row["score"])
        if not 0 <= score <= spec["max_points"]:
            raise SystemExit(
                f"{row['criterion']} score {score:g} is outside 0..{spec['max_points']} "
                f"for text starting {row['text'][:60]!r}")
        triples.append({
            "source_rubric": row["rubric"],
            "criterion": row["criterion"],
            "criterion_description": spec["description"],
            "max_points": spec["max_points"],
            "text": row["text"],
            "score": score,
            "normalised_score": score / spec["max_points"],
        })
    return triples, skipped


def main():
    if not os.path.exists(SOURCE):
        raise SystemExit(f"{SOURCE} not found.")
    with open(SOURCE) as f:
        rows = json.load(f)

    triples, skipped = build(rows)
    if not triples:
        raise SystemExit("No Thesis or Evidence rows found -- nothing to build.")

    with open(TARGET, "w") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)
        f.write("\n")

    from collections import Counter
    counts = Counter((t["source_rubric"], t["criterion"]) for t in triples)
    print(f"Read {len(rows)} rows from {os.path.basename(SOURCE)}")
    print(f"Wrote {len(triples)} triples to {os.path.basename(TARGET)}")
    for (rubric, criterion), n in sorted(counts.items()):
        print(f"  {rubric:<10} {criterion:<10} {n:>4}")
    if skipped:
        print(f"  skipped {len(skipped)} row(s) for criteria outside this experiment: "
              f"{sorted(set(skipped))}")
    print(f"\n{os.path.basename(SOURCE)} is unchanged -- it feeds the shipped graders.")


if __name__ == "__main__":
    main()
