# bert_rubric_scorer — what these weights are

The weight files in this directory are **not in git** (269 MB; see
`.gitignore`). This file is, and it records where they came from, what they
were measured to do, and what they were measured *not* to do.

## Provenance

| | |
|---|---|
| produced by | `scripts/own_data_finetune.py`, run on Google Colab (Tesla T4) |
| run date | 2026-10-07 (`run_at` 2026-10-07T17:12:03Z) |
| base model | `distilbert-base-uncased`, regression head, `num_labels=1` |
| training data | `training_data/rubric_conditioning_train.json`, sha256 `c6dccc50f8223d1c` — all 128 Thesis/Evidence rows from Essay 1 and Essay 2, derived from `sample_answers.json` as of commit `75868b6` |
| split | 102 train / 26 dev, stratified by criterion, seed 20261007 |
| chosen hyperparameters | lr 3e-5, epoch 3 (dev MAE 1.038 points), picked on the dev slice only |
| script sha256 | `70ad1ab10c69ca91` |
| shipped baseline compared against | git `6351d86` |
| evidence | [`training_data/results/own_data_finetune_results.md`](../../training_data/results/own_data_finetune_results.md), with raw `own_data_finetune.json` and the run log beside it |

**Verified on placement (2026-10-07):** loading these files locally and
scoring both locked holdouts reproduces all 40 recorded `fine_tuned_pred`
values exactly. These are the weights the experiment scored, not a lookalike.

## What the measurement actually showed

Stated the way the pre-declaration required, not the way it would read best.

**This did not clear the bar set before the run.** That bar was: MAE or QWK
ahead on *both* clean holdouts for at least one criterion, with the interval
clear of zero. Nothing met it. Of sixteen paired comparisons, two are
distinguishable from noise:

- **Thesis within-1 on holdout2: 50% → 92%** (+41.7pp [+16.7, +66.7])
- **Evidence QWK on essay2: 0.55 → 0.82** (+0.268 [+0.018, +0.547])

The other fourteen contain zero.

**It also showed no detectable regression anywhere.** Zero distinguishable
losses. QWK favours the fine-tuned model in all four cells, MAE in three of
four; fifteen of sixteen point estimates are favourable or neutral. At n = 12
and n = 8 that is a consistent direction without statistical proof, which is
the most this data can give.

**A real behavioural trade:** within-1 rises or holds in all four cells while
exact match *falls* in three. The model lands near the right score more often
and on it less often. For a system where a teacher reviews every suggestion
before a student sees it, that is the right way round — but "exact match got
worse" is true and is not buried here.

**The documented Evidence weakness split in two.** Better on essays with no
evidence at all: the Essay 2 anecdote that `essay2_results.md` singled out
(*"I used ChatGPT to help me study… I got a good grade"*, true 0) went from
**6/10 to 2/10**. No better, and in one case four points worse, on facts
stated without being tied to an argument: holdout2 #11 (true 4) went from
**4 to 8**.

**Overfitting:** no flag tripped, but at the chosen lr 3e-5 dev MAE ticks up
at epoch 4 (1.038 → 1.115). It missed the flag only because the threshold was
+0.1 and the rise was +0.077. The chosen checkpoint sits one epoch before the
curve turns. "No signal" means "not yet", not "won't".

## The thing to know before trusting this on a new criterion

Measured on these weights after placement, over the 20 distinct holdout
essays:

| | humans (the labels) | this model |
|---|---|---|
| correlation between Thesis% and Evidence% | 0.73 – 0.76 | **0.9997** |
| mean gap between the two criteria | 21 – 26 pp | **1.6 pp** |

Swapping in a **nonsense** criterion segment (*"Banana: Is the essay about
fruit?"*) still correlates **0.9986** with the Thesis segment, and replacing
the description with the constant `criterion: unspecified` moves the output by
~0.03 on a 0–1 scale.

**So this model is not reading the criterion.** It computes one latent
essay-quality score and rescales it by that criterion's `max_points`. This is
the same null `rubric_conditioning_results.md` reported, now confirmed on the
production weights: the training data contains two criterion descriptions,
which is not enough diversity for conditioning to become necessary, so the
model never learned to use it.

It still scores Thesis and Evidence acceptably **because the labels for those
two are themselves correlated at ~0.74** — good essays tend to score well on
both. Where they diverge, the model fails in a predictable direction, and
these are exactly the worst errors in the holdout:

| essay | true Thesis | true Evidence | model Evidence | error |
|---|---|---|---|---|
| holdout2 #9 | 4/5 (80%) | 2/10 (20%) | 6/10 | **+4** |
| holdout2 #11 | 5/5 (100%) | 4/10 (40%) | 8/10 | **+4** |

A strong-thesis/weak-evidence essay gets its Evidence score dragged up to
match its Thesis score. That is the mechanism behind the
"facts-without-argument got worse" finding above, and it is a property of the
architecture, not of these particular essays.

### Consequence for `CRITERION_DESCRIPTIONS`

The lookup table in `app/grading/bert_scorer.py` is **not** a
plug-in-a-description-and-it-works extension point, however much it looks like
one. Adding an entry for a new criterion would produce confident, in-range,
plausible-looking scores that are really just general essay quality wearing
that criterion's name — and `rubric_conditioning_results.md` already has the
worked example: a confident, one-sided essay with no counterargument at all
scored **4/5** on "Counterargument" by exactly this architecture.

A criterion that is not in that table falls through to `grading_failed` and a
teacher grades it by hand. **That is the correct and safe behaviour, and it is
why the table is keyed to the two criteria that were actually measured.**
Adding a third entry means first validating it the way these two were: a
locked holdout, scored once, with the interval reported.
