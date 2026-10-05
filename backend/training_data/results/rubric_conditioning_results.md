# Can a grader score a criterion it has never seen?

**Pre-declared. Written and committed before anything was trained and before
the test set was scored.** Nothing below was edited once results existed; the
results are appended underneath when the Colab run finishes.

## The question

Every grader this project ships is keyed to one criterion *row id*. It learns
"Thesis" as an opaque label, and a teacher who adds a new criterion gets
nothing until they have collected labels for it. This asks whether
conditioning on the criterion's **description** instead gives the model
something transferable: score the essay against *this question about it*,
rather than against criterion #1.

If it works, adding "Counterargument" to a rubric produces a usable first-pass
score with no labelled examples for it. If it does not, the per-criterion model
the project ships is the right shape and this line of work stops.

## Design, fixed in advance

| | |
|---|---|
| model | `distilbert-base-uncased`, regression head, one output |
| input | a **sentence pair**: `[CLS] essay [SEP] criterion name: description [SEP]` |
| max length | 384 tokens, truncating the essay only, so the criterion is never cut |
| target | score ÷ max_points, so every criterion sits on 0-1 |
| at inference | multiply by the target criterion's own maximum, round, clamp |
| grid | lr ∈ {2e-5, 3e-5}, up to 4 epochs, best by **dev MAE in points** |
| dev | a 20% slice of the *seen* criteria, stratified by criterion |
| seed | 20261005 |
| hardware | Colab T4. Not local: fine-tuning froze an 8 GB M1 once already |

Normalising the target matters. Trained on raw points a single model would
learn that Evidence answers are worth about twice Thesis ones, which is a fact
about the scale, not about the writing.

### Canonical descriptions

These are the conditioning signal, written as questions a marker would ask, in
the same voice as the held-out one so that a model which learns to read them
has something transferable to read.

- **Thesis** (0-5) — *Does the essay take a clear, specific, arguable position?*
- **Evidence** (0-10) — *Does the essay support its position with specific,
  relevant evidence tied to the argument?*
- **Counterargument** (0-5, held out) — *Does the essay acknowledge an opposing
  viewpoint and respond to it?*

### Training data

All 128 existing Thesis/Evidence rows from Essay 1 and Essay 2, reformatted by
`scripts/build_rubric_conditioning_data.py` into
`training_data/rubric_conditioning_train.json`. **`sample_answers.json` is not
modified** — it feeds the shipped per-criterion graders, and this reformatting
exists for one experiment.

### The three systems

- **(A) rubric-aware** — reads `name: description`.
- **(B) ablation** — identical architecture, identical data, identical grid,
  with the description replaced by the constant string `criterion:
  unspecified`. It cannot tell one criterion from another, so it must work from
  the essay alone.
- **(baseline) midpoint** — predicts 3 (the rounded midpoint of 0-5) for every
  essay. There is no training-derived mean for a criterion with no training
  data, so the midpoint is the fair naive guess.

**A − B is the number that answers the question.** A beating the baseline only
shows the essays are not uniformly scored; A beating B is what shows the
description is doing work.

## The test set

8 essays, criterion **Counterargument**, locked in commit `54cc4e1` before the
training file, the script, or any weights existed. The script refuses to run
unless the file still hashes to `ca94cac65049f18b`.

Scored **once**, at the end. A second run into the same output folder is
refused.

## Disclosed before running: one contaminated essay

The leakage guard **stops** on essay 5:

> holdout essay 5 (Counterargument 0/5): *"Social media is used by millions of
> people every day for communication and entertainment."*
> training row (Evidence 0/10), cosine **0.95**: *"Social media is used by
> billions of people around the world every single day."*

Essays 3, 4 and 7 sit at 0.80-0.89 against vague low-scoring training rows,
under the 0.90 duplicate threshold but close.

This is *text* overlap, not criterion overlap — no model has seen a
Counterargument label. The essays are kept as given, because the labels are the
team's, but the collision is recorded because it is exactly the sort of thing
that quietly flatters a result.

**It biases against the hypothesis, not toward it.** Both rows normalise to
0.0, so a model that ignores the criterion entirely and memorises the text
scores essay 5 correctly. That helps the ablation at least as much as the
rubric-aware model, narrowing the A − B difference the experiment is trying to
measure. A **sensitivity pass with essay 5 removed** is therefore pre-declared
alongside the headline numbers.

## What will be reported

MAE, exact, within-1 and QWK (labels 0-5) for all three systems, with
bootstrap 95% intervals; the paired A − B difference with its interval; the
per-essay table; the sensitivity pass without essay 5; and a check that A did
not get *worse* on the seen criteria it was actually trained on.

## What would count as a result, at n = 8

Stated now so the answer cannot be rationalised afterwards.

- **A believable positive**: A clearly ahead of B on the paired difference with
  an interval clear of zero, *and* the gap surviving the removal of essay 5,
  *and* A no worse than B on the seen criteria. With 8 essays the interval
  would have to be wide-but-clear — roughly a full point of MAE — for this.
- **A believable negative**: A and B indistinguishable, or A behind. That is
  the more likely outcome and a perfectly good answer: it would say the
  description is not being read, and that per-criterion models are the right
  shape for this project.
- **The expected outcome is inconclusive.** 8 essays and 128 training rows is a
  pilot. An interval containing zero means "this did not measure the effect",
  not "there is no effect" — and it will be reported that way rather than as a
  near-miss.

Whatever the point estimate, **a difference whose interval contains zero will
not be called a result.**

---

*Results appended below after the Colab run.*
