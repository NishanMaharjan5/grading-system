# Essay 2: does the method work on a second topic?

Every Essay 1 result shares one subject: whether governments should regulate
social media. A grader can look like it understands argument quality when it
has only learned the vocabulary of one debate. Essay 2 is the check — a
different topic (**should students be allowed to use AI tools like ChatGPT
for schoolwork?**), the same pipeline, nothing about the code changed.

- **Rubric:** "Essay 2", criteria Thesis (0-5) and Evidence (0-10) — the same
  names and points as Essay 1, deliberately, so the feedback templates and the
  review UI needed no change. Seeded by `scripts/seed_demo_rubric.py` (38998e0).
- **Holdout:** 8 team-scored essays, committed in **9f6232e before a single
  Essay 2 training row was written** and before any Essay 2 model existed.
- **Training:** 30 rows (15 Thesis, 15 Evidence) appended to
  `sample_answers.json`, taking it from 98 to 128. Append-only: 180
  insertions, 0 deletions, the first 98 rows byte-identical.
- **Scored once**, at the end, with the same bootstrap (10,000 resamples,
  seed 0) as every other holdout here.

## The models really are separate

Worth stating because the training file matches rows to a rubric by its
*title*, which could suggest the two topics share a model. They do not. Only
`train_grader.py` uses the strings, and only to resolve a row to a criterion
database row. Everything after that is keyed by **criterion row id**:
`ml_models/criterion_<id>.joblib`, and `engine.py` loads `load_model(c.id)`.
Essay 1 holds criteria 1-2, Essay 2 holds 4-5, so they are four separate
files fit on disjoint rows.

Retraining on the combined 128-row file left Essay 1 **byte-identical**:

| | Thesis | Evidence |
|---|---|---|
| Essay 1 leave-one-out, before (98 rows) | MAE 1.11, QWK 0.53 | MAE 1.52, QWK 0.79 |
| Essay 1 leave-one-out, after (128 rows) | MAE 1.11, QWK 0.53 | MAE 1.52, QWK 0.79 |
| `criterion_1/2.joblib` md5 | unchanged | unchanged |

No cross-contamination, by construction and by measurement.

## Result on the locked holdout

| criterion | | MAE | within 1 | exact | QWK |
|---|---|---|---|---|---|
| Thesis (0-5) | model | 0.88 [0.50, 1.25] | 88% [62%, 100%] | 25% [0%, 62%] | 0.79 [0.35, 0.93] |
| | guess the mean (3) | 1.62 [0.62, 2.12] | 38% | 12% | 0.00 |
| | MAE gain over guessing | 0.75 [-0.25, 1.38] | | | |
| Evidence (0-10) | model | 2.50 [1.25, 3.88] | 38% [12%, 75%] | 12% [0%, 38%] | 0.55 [0.17, 0.85] |
| | guess the mean (4) | 2.62 [1.12, 3.38] | 38% | 0% | 0.00 |
| | MAE gain over guessing | 0.12 [-1.88, 1.62] | | | |

Intervals are 95% bootstrap intervals over **8 essays**, which is half the
size of the Essay 1 holdouts and very small. "MAE gain over guessing" is the
baseline's MAE minus the model's; a gain whose interval contains 0 cannot be
told apart from no gain.

**Thesis transferred. Evidence did not.**

- **Thesis: MAE 0.88 against a 1.62 baseline, QWK 0.79, seven of eight essays
  within one point.** On a topic the criterion had never seen, scored by
  models fit on 15 examples, that is a real result — though the gain's
  interval still touches zero at this sample size.
- **Evidence: MAE 2.50 against a 2.62 baseline — a gain of 0.12, interval
  [-1.88, +1.62].** It did not beat guessing the mean in any way this data can
  detect. QWK 0.55 says the ranking is not random, but the scores themselves
  are no better than a constant.

### Per-essay, worst first (error = predicted − true)

Thesis:

| essay | words | true | predicted | error |
|---|---|---|---|---|
| 3 | 26 | 1 | 3 | +2 |
| 1 | 47 | 5 | 4 | -1 |
| 2 | 37 | 5 | 4 | -1 |
| 5 | 45 | 4 | 3 | -1 |
| 7 | 24 | 1 | 2 | +1 |
| 8 | 32 | 4 | 5 | +1 |
| 4 | 30 | 0 | 0 | 0 |
| 6 | 22 | 3 | 3 | 0 |

Evidence:

| essay | words | true | predicted | error |
|---|---|---|---|---|
| 4 | 30 | 0 | 6 | +6 |
| 7 | 24 | 5 | 9 | +4 |
| 8 | 32 | 5 | 9 | +4 |
| 2 | 37 | 5 | 7 | +2 |
| 3 | 26 | 0 | 2 | +2 |
| 1 | 47 | 8 | 9 | +1 |
| 6 | 22 | 2 | 3 | +1 |
| 5 | 45 | 8 | 8 | 0 |

### The same failure mode, on a new topic

Every Evidence error is an **over-score**, and the three worst are the exact
weakness documented for Essay 1 in `holdout_results.md`, reproduced on a
subject the model was never trained on:

- **Essay 4** (true 0, predicted **6**): "I used ChatGPT to help me study for
  my chemistry test last week... I got a good grade." A personal anecdote with
  no evidence at all, scored as though it had some. The Thesis model got this
  one exactly right (0), so the failure is specific to Evidence.
- **Essay 7** (true 5, predicted **9**): "Sciences Po banned ChatGPT in 2023.
  Turnitin released detection data the same year." Three real facts, listed,
  tied to no argument — scored near the top of the scale. This is the
  "statistic stated without being used" weakness, and it is now confirmed to
  be a property of the method rather than of the social-media training set.
- **Essay 8** (true 5, predicted **9**): a reasonable claim with thin support,
  over-credited for naming "several reported cases".

## What this does and does not establish

**Does, for Thesis only:** the Thesis criterion generalised to a new topic.
A model fit on 15 fresh examples, on a subject with no overlap with the
original training data, reached **QWK 0.79** and MAE 0.88 against a 1.62
baseline on a holdout it had never seen. The feedback templates, review flow
and UI carried over with no code change.

This does **not** license the claim that "the method generalises". One of the
two criteria did; the other did not, on the same essays, in the same run.

**Does not:**

- **Evidence did not generalise.** Its gain over guessing the mean is 0.12
  with an interval of [-1.88, +1.62] — not distinguishable from zero. On this
  topic it is no better than predicting a constant, and reporting Essay 2 as a
  success for "the grading approach" without that qualifier would be wrong.
- **8 essays is very small.** Every interval here is wide, and one essay's
  label moving by a point would visibly shift the numbers.
- **15 training examples per criterion** is a quarter of what Essay 1 has (44
  and 54). Some of the Evidence gap is likely sample size rather than topic.
- **Team-scored, one prompt, same authors.** The essays and their scores were
  written by the project team, as with every other holdout here. It measures
  generalisation to a new *topic*, not agreement with real teachers.
- **Three holdout essays share a specific fact with a training row** (the
  Stanford follow-up study, the Sciences Po ban, Turnitin's detection data;
  top cosine 0.86, no duplicates). That overlap is higher than on the Essay 1
  holdouts, because 8 essays on one narrow prompt draw from a small pool of
  real facts. The leakage guard cleared all three against every holdout file,
  but it is a reason to read the Evidence numbers conservatively rather than
  as a clean topic transfer.

## Verdict

Two topics now, with the same honest finding in both: **the Thesis criterion
works, and the Evidence criterion rewards what evidence looks like rather than
whether it supports the claim.** Essay 2 is the stronger evidence for that
conclusion, because the weakness reappeared on a topic the model had never
been trained on.

Teacher review remains the backstop. No AI score reaches a student unapproved,
which is exactly the protection essays 4, 7 and 8 need.
