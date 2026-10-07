# Phase 1: does fine-tuning on our own data beat what's shipped?

**Pre-declared. Written and committed before anything was trained and before
either locked test set was scored by the new model.** Nothing below may be
edited once results exist; they are appended underneath when the Colab run
finishes.

## The question

Every result so far about fine-tuning a transformer in this project has
answered a methods question on data that isn't ours: ASAP-AES (public,
`asap_results.md`) or a zero-shot criterion with no training data at all
(`rubric_conditioning_results.md`). Neither says anything about the one
question that actually matters for this app: **if we fine-tune on the ~130
examples we've actually labeled, does it beat the frozen-embedding pipeline
that's running in production right now, on that pipeline's own locked test
sets?**

A loss or a null result is a legitimate, expected-possible outcome at this
sample size — not a failure of execution. This is Phase 1, an experiment
only. Nothing here touches `ml_models/`, `app/grading/engine.py`, or
production. Integration is a separate decision made after seeing these
results, not before.

## A correction to the brief this was written against

The request that started this said Essay 2 "just got 10 more essays added."
I checked before doing anything else: `sample_answers.json` is 128 rows,
unchanged since commit `75868b6`, and `git log` shows no later commit
touching it. `git status` is clean — nothing uncommitted either. Essay 2 is
still 15 Thesis + 15 Evidence rows, exactly as `essay2_results.md` describes.
I didn't find the 10 new rows anywhere in this repo. Everything below uses
the data that's actually here; if those rows exist somewhere else, say so and
this gets rerun before Colab, not after.

## Fresh baseline: the shipped pipeline, run again today

`scripts/evaluate_holdout.py` reruns the four saved models in `ml_models/`
against each locked holdout. It only loads weights and predicts — it never
retrains — so a number changing here would mean the models on disk don't
match what the training file says, which is worth catching before anything
else. All three were run fresh (git HEAD `6351d86`, 2026-10-07), not pulled
from memory:

| holdout | criterion | MAE | within-1 | exact | QWK |
|---|---|---|---|---|---|
| holdout1 (burned dev set) | Thesis | 0.75 [0.42, 1.08] | 92% [75%, 100%] | 33% [8%, 58%] | 0.86 [0.65, 0.94] |
| holdout1 (burned dev set) | Evidence | 1.42 [0.75, 2.25] | 58% [33%, 83%] | 25% [0%, 50%] | 0.81 [0.55, 0.94] |
| **holdout2 (clean)** | **Thesis** | **1.17 [0.67, 1.67]** | 50% [25%, 75%] | 33% [8%, 58%] | 0.63 [0.28, 0.78] |
| **holdout2 (clean)** | **Evidence** | **1.75 [1.08, 2.33]** | 25% [0%, 50%] | 25% [0%, 50%] | 0.63 [0.28, 0.78] |
| **essay2 holdout (clean)** | **Thesis** | **0.88 [0.50, 1.25]** | 88% [62%, 100%] | 25% [0%, 62%] | 0.79 [0.35, 0.93] |
| **essay2 holdout (clean)** | **Evidence** | **2.50 [1.25, 3.88]** | 38% [12%, 75%] | 12% [0%, 38%] | 0.55 [0.17, 0.85] |

**Every number above is identical to the previously published "AFTER" state**
in `holdout_results.md` and `essay2_results.md`, to two decimal places. That
is the expected result when nothing has changed — the shipped models, the
training file and the holdouts are exactly what they were — and it is
reassuring rather than redundant: it confirms the four `.joblib` files in
`ml_models/` really do match `sample_answers.json` as it stands today, with
no silent drift. Raw per-essay output: `results/phase1_baseline_holdout1.json`,
`phase1_baseline_holdout2.json`, `phase1_baseline_essay2.json`.

**The bold rows are the baseline to beat.** holdout1's numbers are reported
for completeness only — see below for why they don't count.

## Which holdouts are clean

Confirmed by rereading `holdout_results.md` and `essay2_results.md`, not from
memory:

| file | role | clean? |
|---|---|---|
| `holdout_team_scored.json` ("holdout1") | **Development set.** `holdout_results.md` names it that explicitly: "Studied in detail before the round-two changes, so any improvement on it is not evidence." Eleven new training essays in round 2 were written specifically to fix the two failure modes this set exposed. | **No — burned.** Excluded from the comparison below. |
| `holdout2_team_scored.json` | **Final test set.** Committed (`3623730`) before any result was computed on it; never used to tune, retrain or choose anything since. | **Yes.** |
| `holdout_essay2_team_scored.json` | Committed (`9f6232e`) before a single Essay 2 training row existed. Scored once, at the end, when Essay 2 training was added. | **Yes.** |
| `holdout_counterargument_team_scored.json` | Out of scope here — a different criterion (zero-shot), not Thesis/Evidence. Used only by `rubric_conditioning.py`. | n/a |

`own_data_finetune.py` enforces this structurally: it never loads
`holdout_team_scored.json` at all, and a pre-flight hash check refuses to run
if `holdout2_team_scored.json` or `holdout_essay2_team_scored.json` has
changed since this was written.

## Design, fixed in advance

| | |
|---|---|
| model | `distilbert-base-uncased`, regression head, one output — **identical architecture to the rubric-conditioning experiment's rubric-aware variant** |
| input | a **sentence pair**: `[CLS] essay [SEP] criterion name: description [SEP]` |
| max length | 384 tokens, truncating the essay only |
| target | score ÷ max_points, so Thesis (0–5) and Evidence (0–10) sit on the same 0–1 scale for one shared model |
| at inference | multiply by the target criterion's own maximum, round, clamp — identical to `engine.py`'s own rounding |
| grid | lr ∈ {2e-5, 3e-5}, up to 4 epochs, best by **dev MAE in points** |
| dev | a 20% slice of the **training data only**, stratified by criterion |
| seed | 20261007 |
| hardware | Colab T4. Not local: fine-tuning froze an 8 GB M1 once already |
| test sets | `holdout2_team_scored.json` (12 essays) and `holdout_essay2_team_scored.json` (8 essays) — **both**, scored once each |
| what's compared | the fine-tuned model's predictions vs. the shipped pipeline's own fresh predictions (above), paired bootstrap, same essays |

### Canonical descriptions

Identical to the rubric-conditioning experiment, reused rather than
rewritten:

- **Thesis** (0–5) — *Does the essay take a clear, specific, arguable position?*
- **Evidence** (0–10) — *Does the essay support its position with specific,
  relevant evidence tied to the argument?*

### Training data

All 128 existing Thesis/Evidence rows from Essay 1 and Essay 2 —
`training_data/rubric_conditioning_train.json`, already built by
`scripts/build_rubric_conditioning_data.py` for the earlier experiment and
reused here unchanged. `sample_answers.json` is not touched.

This is a single shared model across both criteria and both essay topics —
not four separate fine-tunes — because the whole point of fine-tuning on our
own data is seeing whether one model can use Thesis examples to help it score
Evidence, and Essay 1 examples to help it score Essay 2, the way the shipped
per-criterion Ridge models structurally cannot.

### One departure from the rubric-conditioning run, disclosed

That experiment trained two variants (with the description, and with it
replaced by a placeholder) to isolate whether conditioning helped. This
experiment trains **one** variant — the rubric-aware one only — because the
question here isn't whether conditioning helps; it's whether fine-tuning on
our own data beats the shipped pipeline at all. Running the ablation too
would answer a question nobody asked and cost another full grid for nothing.

### Overfitting, watched explicitly

~100 training rows for a 67M-parameter model is small enough that
memorization is a real risk, not a formality. Every epoch's train MSE and dev
MAE are logged for both learning rates (`own_data_finetune.json` → `fit.grid`).
The script also computes an automatic flag: for a given learning rate, if dev
MAE rises by more than 0.1 points above its best value *after* the best
epoch while train loss keeps falling, that run is flagged in the log and the
output as a possible overfitting signal. A flag doesn't invalidate the run —
early stopping on dev MAE already protects the chosen checkpoint — but it
would mean the grid's upper end (epoch 4) is not to be trusted, and that is
reported either way.

## What will be reported

For each of the four (holdout, criterion) pairs: MAE, within-1, exact and QWK
for both the fine-tuned model and the shipped pipeline, each with a bootstrap
95% interval (10,000 resamples, seed 0 — the same bootstrap `evaluate_holdout.py`
uses, so the two are read the same way); the paired difference
(fine-tuned − shipped) with its own interval; and the per-essay table. Also:
model size, CPU inference time per submission (mean and max, one essay at a
time, padded to 384 tokens — realistic for whether Phase 2 is practical
without a GPU in production), and the train/dev curves for both learning
rates with the overfitting flag.

## What would count as a result

Stated now so the answer cannot be rationalised afterwards.

- **A believable win for fine-tuning**: beats the shipped pipeline on MAE (or
  QWK) on **both** clean holdouts, for **at least one** criterion, with an
  interval clear of zero, and no overfitting flag on the learning rate that
  was chosen. One holdout improving and the other not would be a mixed
  result, not a win — Essay 2 vs. holdout2 already showed once (Evidence)
  that a result on one topic doesn't carry to another.
- **A believable loss**: the shipped pipeline stays ahead with an interval
  clear of zero, on one or both holdouts. A real, useful answer: it would
  mean ~130 examples is not enough to beat a much cheaper frozen-embedding
  model, and the current pipeline's shape is right for this amount of data.
- **The expected outcome is a null or a mixed result.** 12 and 8 essays give
  wide intervals, as every holdout here has. An interval containing zero
  means "this did not measure a difference," not "there is no difference,"
  and it will be written up that way — not as a near-miss in either
  direction.

Whatever the point estimates, **a difference whose interval contains zero
will not be called a win or a loss.**

## Isolation

`own_data_finetune.py` writes only to `--out-dir`. It never imports the Flask
app, never touches the database, and never reads or writes `ml_models/`. The
shipped graders are exactly what they were before this ran, provably so —
the fresh baseline above is the proof.

---

*(Results appended here once the Colab run finishes.)*
