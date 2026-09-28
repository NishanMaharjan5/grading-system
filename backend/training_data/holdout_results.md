# Held-out evaluation

Two sets of 12 essays each, both scored against the models in `ml_models/` by
`scripts/evaluate_holdout.py`. That script only loads models; it never
retrains and never touches `sample_answers.json`, and it prints identical
numbers until the models are retrained. Raw per-essay predictions and every
interval are saved in `results/*.json`.

| set | file | role |
|---|---|---|
| 2 | `holdout2_team_scored.json` | **Final test set.** Committed (`3623730`) before any result was computed on it. Not used to tune, retrain or choose anything. |
| 1 | `holdout_team_scored.json` | **Development set.** Studied in detail before the round-two changes, so any improvement on it is *not* evidence. |

## Read these numbers with these limits

- **Team-scored.** The essays and their scores were written by the project
  team, not by independent teachers. This measures whether the model
  generalises to paragraph-length text against the team's own judgment; it does
  not measure agreement with real graders.
- **Twelve essays per set.** Every interval below is a bootstrap 95% interval
  (essays resampled 10,000 times, fixed seed). They are wide, and a single
  essay's label moving by a point visibly moves the point estimates. QWK is
  indicative at best.
- **One prompt throughout.** Every essay, training row and holdout row is
  about regulating social media. Nothing here says how the grader behaves on a
  different question.
- **Written by the same people.** The training data and both holdout sets share
  authors, and therefore habits of phrasing. A held-out set written by the same
  team is easier than one written by strangers.
- **The "guess the mean" baseline is generous.** It predicts the rounded mean of
  the very labels being scored, which a real grader would not know in advance.

## Set 2 (final test set), BEFORE

Models: trained on the original 76 examples (Thesis n=33, Evidence n=43),
unchanged since `c14d20d`. Run before any new training data was added.

Leakage guard: no duplicates and no near-duplicates against the training data.
Eleven rows share only a topic or generic wording with a training example (top
cosine 0.76, and nothing shares a specific statistic or law with it), which is
weaker than set 1, where the FTC settlement and the EU Digital Services Act
were reused.

| criterion | | MAE | within 1 | exact | QWK |
|---|---|---|---|---|---|
| Thesis (0-5) | model | 1.00 [0.67, 1.33] | 83% [58%, 100%] | 17% [0%, 42%] | 0.73 [0.47, 0.84] |
| | guess the mean (3) | 1.75 [0.83, 2.17] | 33% | 8% | 0.00 |
| | MAE gain over guessing | 0.75 [-0.08, 1.08] | | | |
| Evidence (0-10) | model | 1.92 [1.17, 2.67] | 50% [25%, 75%] | 17% [0%, 42%] | 0.63 [0.40, 0.78] |
| | guess the mean (4) | 2.75 [1.67, 3.42] | 17% | 8% | 0.00 |
| | MAE gain over guessing | 0.83 [-0.58, 1.67] | | | |

Intervals are 95% bootstrap intervals over the 12 essays. "MAE gain over
guessing" is the baseline's MAE minus the model's, with its own interval; a
gain whose interval includes 0 cannot be told apart from no gain at this
sample size.

### Per-essay, worst first (error = predicted − true)

Thesis:

| essay | words | true | predicted | error |
|---|---|---|---|---|
| 5 | 40 | 1 | 3 | +2 |
| 12 | 52 | 0 | 2 | +2 |
| 1 | 71 | 5 | 4 | -1 |
| 2 | 66 | 5 | 4 | -1 |
| 4 | 43 | 0 | 1 | +1 |
| 6 | 56 | 1 | 2 | +1 |
| 8 | 78 | 5 | 4 | -1 |
| 9 | 34 | 4 | 3 | -1 |
| 10 | 42 | 4 | 5 | +1 |
| 11 | 58 | 5 | 4 | -1 |
| 3 | 43 | 4 | 4 | 0 |
| 7 | 35 | 3 | 3 | 0 |

Evidence:

| essay | words | true | predicted | error |
|---|---|---|---|---|
| 5 | 40 | 0 | 4 | +4 |
| 12 | 52 | 2 | 6 | +4 |
| 3 | 43 | 6 | 9 | +3 |
| 4 | 43 | 0 | 3 | +3 |
| 9 | 34 | 2 | 5 | +3 |
| 7 | 35 | 2 | 4 | +2 |
| 1 | 71 | 8 | 7 | -1 |
| 2 | 66 | 8 | 7 | -1 |
| 6 | 56 | 5 | 6 | +1 |
| 11 | 58 | 4 | 5 | +1 |
| 8 | 78 | 9 | 9 | 0 |
| 10 | 42 | 7 | 7 | 0 |

What the table shows, described and not acted on (holdout2 is final):

- **Thesis:** ten of twelve essays are within one point. The two misses are
  both +2 on essays whose true thesis is 0 or 1 (essay 5, a vague "regulation
  is a word people use" paragraph; essay 12, a list of statistics).
- **Evidence:** six of twelve are within one point, and **every one of the six
  misses is an over-score** of +2 to +4 on an essay the team scored low or
  middling on evidence. The two essays with no evidence at all (4 and 5, both
  true 0) were predicted 3 and 4. Essays the team scored 7 to 9 (1, 2, 8, 10)
  are within one point.
- The error is one-sided: the model rarely scores a weak essay low, which is
  the same direction as the two worst misses on set 1 (essays 7 and 11).

**AFTER results are added below once the new training rows are in and the
models retrained.**

## Set 1 (development set)

*Studied before the round-two fixes. Its numbers are here for continuity, not
as evidence of improvement.*

Models: the same original 76-example models. The set was first run before
intervals existed; the point estimates below are unchanged, and the intervals
were added by the updated script.

| criterion | | MAE | within 1 | exact | QWK |
|---|---|---|---|---|---|
| Thesis (0-5) | model | 1.00 [0.42, 1.75] | 83% [58%, 100%] | 42% [17%, 67%] | 0.67 [0.16, 0.95] |
| | guess the mean (3) | 1.83 [1.00, 2.17] | 25% | 8% | 0.00 |
| | MAE gain over guessing | 0.83 [-0.17, 1.58] | | | |
| Evidence (0-10) | model | 1.67 [0.75, 2.67] | 50% [25%, 75%] | 42% [17%, 67%] | 0.71 [0.42, 0.90] |
| | guess the mean (4) | 2.75 [1.58, 3.33] | 33% | 0% | 0.00 |
| | MAE gain over guessing | 1.08 [-0.67, 2.25] | | | |

Point estimates beat the guess-the-mean baseline on both criteria, but the
interval on the gain includes zero for both (Thesis [−0.17, 1.58], Evidence
[−0.67, 2.25]), so 12 essays cannot establish it. (This corrects the first
write-up, which said the model cleared the baseline "by a wide margin".)

### Leakage guard

Nothing crossed the duplicate threshold (text ratio ≥ 0.85 or cosine ≥ 0.90).
Fifteen rows crossed the softer disclosure threshold (cosine ≥ 0.60): the
essay reuses a real fact or argument shape that a training example also used,
in different wording (the 2019 FTC/Facebook $5B settlement, the EU Digital
Services Act, "kids and social media"). Two bear on how to read the results:

- **Essay 1** (FTC/$5B) and **essay 11** (Cambridge Analytica, also citing the
  FTC/$5B fine) both echo training rows 51 and 72. Essay 1 was a clean hit and
  essay 11 was not, so the overlap did not make essay 11 easier, but it may
  flatter essay 1.
- **Essay 6** (EU Digital Services Act) echoes training row 22 and was scored
  close to true (Thesis −1, Evidence exact).

### Per-essay, worst first

Thesis:

| essay | words | true | predicted | error |
|---|---|---|---|---|
| 7 | 31 | 1 | 5 | +4 |
| 11 | 47 | 1 | 4 | +3 |
| 6 | 62 | 5 | 4 | -1 |
| 8 | 64 | 5 | 4 | -1 |
| 9 | 36 | 0 | 1 | +1 |
| 10 | 51 | 4 | 5 | +1 |
| 12 | 51 | 4 | 5 | +1 |
| 1 | 64 | 5 | 5 | 0 |
| 2 | 55 | 5 | 5 | 0 |
| 3 | 35 | 3 | 3 | 0 |
| 4 | 44 | 1 | 1 | 0 |
| 5 | 17 | 0 | 0 | 0 |

Evidence:

| essay | words | true | predicted | error |
|---|---|---|---|---|
| 11 | 47 | 5 | 10 | +5 |
| 7 | 31 | 0 | 4 | +4 |
| 10 | 51 | 2 | 5 | +3 |
| 12 | 51 | 5 | 8 | +3 |
| 8 | 64 | 8 | 6 | -2 |
| 9 | 36 | 0 | 2 | +2 |
| 2 | 55 | 5 | 6 | +1 |
| 1 | 64 | 8 | 8 | 0 |
| 3 | 35 | 5 | 5 | 0 |
| 4 | 44 | 1 | 1 | 0 |
| 5 | 17 | 0 | 0 | 0 |
| 6 | 62 | 8 | 8 | 0 |

The two worst misses are the same two essays on both criteria:

- **Essay 7** ("There are arguments on both sides... it is a hard question"): a
  fence-sitting essay with no stance and no evidence, scored Thesis 5 (true 1)
  and Evidence 4 (true 0). It contains "while", which the reasoning-connective
  feature counts, and enough on-topic vocabulary to land near strong training
  examples in embedding space.
- **Essay 11** (Cambridge Analytica/FTC facts, no argument connecting them):
  scored Evidence 10 (true 5) and Thesis 4 (true 1). This is the documented
  Evidence weakness, and it is not confined to Evidence: production feeds the
  *same full essay text* to every criterion's model (`app/grading/engine.py`),
  so a fact-heavy paragraph with no thesis inflates the Thesis score too.

Essay 4, also fact-heavy with no argument, was scored exactly right (1 and 1).
Its facts are a generic list of platforms, countries and years; essay 11's are
a named scandal with a dollar figure and a year, a pattern that appears,
worded differently, in several *high*-scoring training examples.

### Does essay length explain the errors?

| | Thesis MAE | Evidence MAE |
|---|---|---|
| paragraph-length (1, 2, 6, 8, 10, 12) | 0.67 | 1.50 |
| short (3, 4, 5, 7, 9, 11) | 1.33 | 1.83 |

Word counts confirm the groups differ (51 to 64 words against 17 to 47), and
the short group has the larger error. But the gap is carried entirely by
essays 7 and 11: four of the six short essays have zero Thesis error. Without
those two, the short group's error is zero on both criteria. So the errors
track content (fence-sitting; facts resembling a strong-evidence template but
not tied to the argument), not length. Length and that failure mode happen to
be correlated in this sample of twelve.
