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

# Result: favourable everywhere, conclusive almost nowhere

Run once on a Colab T4 on 2026-10-07, exactly as pre-declared above. The
pre-flight passed before any weight moved: all three input files hashed to
their locked values, no holdout essay appeared verbatim in training, and both
shipped-baseline files covered the same essays with the same labels (recorded
`shipped_baseline_git_head` `6351d86`, the commit that produced them). 102
training triples, 26 dev, both locked holdouts scored once. Grid chose **lr
3e-5, epoch 3, dev MAE 1.038 points**.

## By the bar written down in advance, this is not a win

The pre-declaration said a believable win needs the fine-tuned model ahead on
MAE **or** QWK, **on both clean holdouts**, for at least one criterion, with
the interval clear of zero. Nothing meets that:

| criterion | metric | holdout2 | essay2 | both clear of zero? |
|---|---|---|---|---|
| Thesis | MAE | +0.167 | −0.125 | no |
| Thesis | QWK | +0.169 | +0.036 | no |
| Evidence | MAE | +0.250 | +0.750 | no |
| Evidence | QWK | +0.146 | **+0.268\*** | no — only essay2 clears |

\* interval clear of zero. Positive = fine-tuned better.

**So the honest headline is: not a win by the stated bar.** Of 16 paired
comparisons (2 holdouts × 2 criteria × 4 metrics), **two** are distinguishable
from noise:

- **Thesis within-1 on holdout2: +41.7pp [+16.7, +66.7]** — 50% → 92%.
- **Evidence QWK on essay2: +0.268 [+0.018, +0.547]** — 0.55 → 0.82.

The other fourteen contain zero.

## It is also, clearly, not a loss

**Distinguishable losses for the fine-tuned model: zero.** Not one of the
sixteen comparisons shows the shipped pipeline ahead in a way this data can
detect. And the direction of the point estimates is lopsided:

- **QWK: fine-tuned ahead on all four** (+0.146, +0.169, +0.268, +0.036).
- **MAE: fine-tuned ahead on three of four**, the exception being essay2
  Thesis at −0.125 — an eighth of a point on 8 essays.
- **within-1: ahead or tied on all four** (+41.7pp, +41.7pp, 0, 0).

Fifteen of sixteen point estimates are favourable or neutral. That is a weak
signal by the standard of statistical significance and a consistent one by the
standard of direction, and at n = 12 and n = 8 those two standards were always
going to disagree. The pre-declaration anticipated exactly this: *"The expected
outcome is a null or a mixed result."*

## One real behavioural difference: closer, but less often exact

A pattern runs through every cell, and it is not noise-shaped:

| | within-1 | exact |
|---|---|---|
| holdout2 Thesis | +41.7pp | −25.0pp |
| holdout2 Evidence | +41.7pp | −8.3pp |
| essay2 Thesis | 0 | −12.5pp |
| essay2 Evidence | 0 | 0 |

**The fine-tuned model lands near the right score more often and on it less
often.** Exact-match drops in three of four cells while within-1 rises or
holds in all four. For this application that trade is the right way round — a
teacher reviewing a suggestion cares whether it is roughly right, and every
score is reviewed before a student sees it — but it should be stated rather
than buried, because "exact match got worse" is a true sentence about this
model.

## The documented Evidence weakness: better on blankness, not on padding

`essay2_results.md` documented the failure mode precisely: essays with no
evidence at all scored as though they had some. On that specific failure the
fine-tuned model is clearly better:

| essay | true | shipped | fine-tuned |
|---|---|---|---|
| holdout2 #5 (no evidence) | 0 | 2 | **0** |
| holdout2 #4 (no evidence) | 0 | 2 | **1** |
| essay2 #3 (no evidence) | 0 | 2 | **1** |
| essay2 #4 (personal anecdote, no evidence) | 0 | 6 | **2** |

essay2 #4 is the case `essay2_results.md` singled out — *"I used ChatGPT to
help me study... I got a good grade"*, scored **6/10** by the shipped model.
The fine-tuned model gives it 2.

But the *other* half of the weakness — facts stated without being tied to an
argument — got **worse**:

| essay | true | shipped | fine-tuned |
|---|---|---|---|
| holdout2 #11 (facts, no argument) | 4 | 4 | **8** |
| holdout2 #9 | 2 | 5 | **6** |
| essay2 #8 ("several reported cases") | 5 | 9 | **9** |

So: better at recognising *nothing*, no better — and on holdout2 #11, four
points worse — at recognising *decoration*. **This is why the padding/tying
probes have to be rerun against the new model rather than assumed settled by
these holdout numbers.** Phase 2 does that.

## Overfitting

No flag tripped, but the curves deserve a sentence rather than a boolean.
Train MSE falls monotonically at both learning rates (0.208 → 0.025 and
0.185 → 0.016) — a 67M-parameter model fitting 102 examples, as expected.
Dev MAE at the chosen lr 3e-5 bottoms at epoch 3 (1.038) and **ticks back up
at epoch 4 (1.115, +0.077)**. That is a real uptick; it did not trip the flag
only because the flag's threshold is +0.1. At lr 2e-5 dev MAE was still
falling at epoch 4 (1.269), so that run never had the chance to turn over.

Read plainly: the chosen checkpoint sits one epoch before the first sign of
the curve turning, which is where early stopping is supposed to put it, and
there is no evidence of damage at that point. But four epochs is as far as
this grid looked, and "no overfitting signal" here means "not yet," not
"won't."

## Cost

67.0M parameters, 269 MB on disk. On Colab's CPU (2 cores, 1 torch thread):
cold load 0.05s, then **449 ms/submission** (max 536) one at a time, padded to
384 tokens. Comfortably inside a 5-second budget even on that hardware, but
that is Colab's CPU, not this app's — a real end-to-end measurement in the
running Flask app is Phase 2's job, not something to infer from this number.

## Verdict

**A favourable mixed result that does not clear the bar it was measured
against.** Two distinguishable wins, zero distinguishable losses, fifteen of
sixteen point estimates favourable or neutral, and one clear qualitative
improvement on the documented Evidence failure mode — against which sits a
worsening on the other half of that same failure mode, an exact-match
regression, and sample sizes that make most of this unprovable either way.

What this licenses: trying it, with the numbers stated as they are. What it
does not license: the sentence "fine-tuning beats the shipped pipeline." At
n = 12 and n = 8 that sentence is not available from this data, and the
pre-declaration said so before the data existed.
