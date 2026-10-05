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

# Result: a clean null on the thing being tested

Run once on a Colab T4 on 2026-10-05, exactly as pre-declared above. The
pre-flight passed before any weight moved: the holdout still hashed to
`ca94cac65049f18b`, the file committed in `54cc4e1`, and no holdout essay
appeared verbatim in the training text. 102 training triples, 26 dev, 8 test.
Both variants independently chose lr 3e-5 at epoch 3.

## Headline: the description made no measurable difference

**A − B, paired, on the zero-shot Counterargument holdout:**

| metric | A − B (positive = rubric-aware better) | 95% CI | distinguishable? |
|---|---|---|---|
| QWK | **−0.056** | [−0.345, +0.047] | no |
| MAE | **−0.25** | [−0.75, +0.25] | no |
| within-1 | 0.000 | [0.000, 0.000] | no |
| exact | −0.25 | [−0.75, +0.25] | no |

**This is a null result, by the criteria written down before the run.** Not a
near-miss and not a positive in disguise. Every interval contains zero, and
every point estimate that moves at all moves *against* the hypothesis — the
ablation, which cannot see the criterion description, scored slightly better.
Reading the description did not help the model score a criterion it had never
seen.

The pre-declaration said a believable positive would need A ahead with an
interval clear of zero, surviving the removal of essay 5, and A no worse on the
seen criteria. None of those held.

## But something real did transfer

Both models beat the naive baseline comfortably on a criterion with no training
data at all:

| system | MAE | QWK | within-1 | exact |
|---|---|---|---|---|
| midpoint baseline (always 3) | 1.875 [1.125, 2.625] | 0.000 | 38% | 13% |
| **(A) rubric-aware** | 1.125 [0.625, 1.750] | 0.694 [0.066, 0.922] | 88% | 13% |
| **(B) no-description ablation** | 0.875 [0.375, 1.503] | 0.750 [0.293, 0.945] | 88% | 38% |

A QWK near 0.7 on an unseen criterion is not nothing. The models rank these
essays in roughly the right order, and both land within one point on seven of
the eight. **Zero-shot transfer happened. It just did not happen because of
the description** — which is a different claim from the one the experiment set
out to test, and a weaker one.

## Why the description did nothing: it did nothing on the seen criteria either

This is the number that explains the null:

| seen-criteria dev slice (26 held-back Thesis/Evidence rows) | MAE (points) | within-1 |
|---|---|---|
| rubric-aware | **1.038** | **76.9%** |
| no-description ablation | **1.038** | **76.9%** |

**Identical.** Not close — the same to three decimal places, on both metrics.
Both variants also picked the same learning rate and the same epoch, and both
bottomed out at a dev MAE of 1.0385. The description had no measurable effect
even on the two criteria the model was actually trained on.

That points at the experimental design rather than at the hypothesis. The
training data contains exactly **two distinct criterion descriptions**. A model
can drive the training loss down by judging general essay quality and ignoring
the second segment entirely, because with two criteria there is almost nothing
to gain from telling them apart — and a gradient that buys nothing does not get
followed. The conditioning signal was available but never became *necessary*.

So this run cannot separate two very different explanations:

1. Conditioning on a rubric description does not help a model generalise to a
   new criterion.
2. Two seen criteria is too little diversity for conditioning to be learned at
   all, so the experiment never put the hypothesis to the test.

**The second is at least as consistent with the evidence as the first.** This
is a limitation of how the experiment was built, not evidence against the idea.

## What actually transferred: a confidence proxy, not counterargument detection

Per-essay, true score → (A / B):

| essay | true | A | B | what it is |
|---|---|---|---|---|
| 1 | 5 | 4 | 5 | names an objection, answers it |
| 2 | 5 | 4 | 5 | names an objection, answers it |
| 3 | 3 | 2 | 2 | gestures at disagreement, no response |
| 4 | 2 | 1 | 1 | vague nod to other views |
| 5 | 0 | 0 | 1 | off-topic, no argument at all |
| 6 | 4 | 5 | 5 | concedes a point, redirects |
| **7** | **1** | **4** | **4** | **confident, one-sided, no counterargument** |
| 8 | 5 | 4 | 5 | names an objection, answers it |

Both models are within one point on every essay **except essay 7, where both
are three points out** — by far the largest error in the set, and the same
error from both.

> **Essay 7** (true 1): *"Governments should regulate social media because
> companies don't fix problems until forced to."*

It is a clean, confident, well-formed claim. It also does not acknowledge an
opposing viewpoint at all, which is the entire criterion. Both models scored
it **4 out of 5**.

That is the tell. In seven of these eight essays, counterargument quality
happens to track general argumentative quality: the essays that engage an
objection are also the better-written ones. A model that has learned "score
good argumentative writing highly" scores those seven about right while knowing
nothing about counterarguments. Essay 7 is the case built to pull the two
apart, and both models fail it identically.

**What transferred looks like a general essay-quality proxy, not detection of
counterargument structure.** It fails exactly where confidence and
non-engagement coincide — which, for this criterion, is the case that matters
most.

## Sensitivity: the contaminated essay did not cause the null

Essay 5 was disclosed before the run as a near-duplicate of a training row
(cosine 0.95). Removing it:

| without essay 5 (n = 7) | MAE | QWK | exact |
|---|---|---|---|
| midpoint baseline | 1.857 | 0.000 | 14% |
| rubric-aware | 1.286 | 0.456 | 0% |
| no-description | **0.857** | **0.638** | 43% |

The ablation is still ahead, and the gap **widens**. The leakage does not
explain the null.

**The pre-declared reasoning about that leakage was wrong, though, and it is
worth recording.** The write-up above argued the contamination would flatter
the ablation, because a model memorising the text would score essay 5 correctly
without reading the criterion. In the event the opposite happened: the
rubric-aware model got essay 5 exactly right (0) and the ablation did not
(predicted 1). Removing the essay therefore hurt A and left B essentially
unchanged. The argument was plausible and still wrong — which is the ordinary
reason to run the sensitivity pass rather than reason about it.

## What it would take to test this properly

The clean follow-up is specific and well-scoped: **train across five or more
genuinely distinct criteria**, not two.

With enough distinct descriptions, a model cannot drive the loss down by
scoring general quality — the same essay has to receive different scores
depending on which criterion is attached, and the only way to do that is to
read the second segment. Conditioning becomes *necessary* rather than optional,
and an ablation against it becomes informative. Candidate criteria already
within reach: Counterargument, Structure, Clarity, Use of sources, Register.

Two further things that run should fix:

- **A bigger zero-shot test.** Eight essays gives intervals wide enough to hide
  anything short of a large effect. Twenty-five to thirty per unseen criterion,
  across two or three unseen criteria, would make a modest effect visible.
- **A test set where the unseen criterion diverges from general quality.** Essay
  7 shows that most of this holdout can be scored by a quality proxy. A test
  built deliberately around confident-but-unengaged and hesitant-but-engaged
  essays would measure the criterion rather than the writing.

Until then, the per-criterion models the project ships remain the right shape:
not because conditioning was shown not to work, but because this run did not
show that it does.

## Caveats

- **8 test essays.** Every interval above is wide and all of them are
  indicative only. The null is a genuine null at this sample size, which is not
  the same as evidence that the effect is zero.
- **One seed, one training run per variant.** Two runs of the same variant
  would differ by some amount this experiment never measured, so a difference
  smaller than that noise would be invisible.
- **128 training triples, two criteria, one prompt family.** All of it written
  and scored by the project team.
- **The holdout labels are the team's own**, as with every other holdout here.

