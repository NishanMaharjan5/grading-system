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

## Round two: what changed

22 rows (11 essays) were added to `sample_answers.json`, taking it from 76 to
98: six deliberately weak-thesis essays, three strong ones, two middling. They
were written to target the two failure modes set 1 exposed — fence-sitting
prose scored as a strong thesis, and facts recited without an argument scored
as strong evidence. The leakage guard cleared them against **both** holdout
files first: no duplicates, and no new training essay shares a named law or
incident with set 2 (the only such overlap, the Digital Services Act, is with
set 1, the development set).

Then the models were retrained and every measurement below was taken again.
**Nothing was tuned after seeing the set 2 AFTER numbers.**

### Leave-one-out on the training data

| criterion | | MAE | within 1 | exact | QWK |
|---|---|---|---|---|---|
| Thesis | before (n=33) | 0.85 | 82% | 39% | 0.66 |
| | after (n=44) | 1.11 | 64% | 32% | 0.53 |
| | *baseline before → after* | *1.39 → 1.48* | *52% → 45%* | *21% → 18%* | *0.00* |
| Evidence | before (n=43) | 1.49 | 58% | 16% | 0.83 |
| | after (n=54) | 1.52 | 52% | 30% | 0.79 |
| | *baseline before → after* | *3.30 → 3.07* | *14% → 19%* | *14% → 19%* | *0.00* |

These two columns are **not directly comparable**: they are leave-one-out on
two different datasets, and the new one is deliberately harder (it contains
fence-sitting and fact-dump essays that are hard to place). The baseline moved
too. Read as a gain over the matching baseline, Thesis fell from 0.54 to 0.37
MAE points and Evidence from 1.81 to 1.55 — so on its own training
distribution the model did not get better, and by most measures got slightly
worse.

### Padding probe (`scripts/padding_probe.py`)

Fresh paired probes, checked against training data so they measure behaviour
rather than memorisation. "Padding" adds names, numbers and years that support
nothing; "tying" connects the same statistic to a claim.

| criterion | measure | before | after | wanted |
|---|---|---|---|---|
| Evidence | mean padding gain | +2.25 | **+1.75** | lower |
| Evidence | mean tying gap | +1.33 | +0.67 | higher |
| Evidence | untied statistic alone (of 10) | 5, 6, 7 | 5, 7, 7 | lower |
| Thesis | mean padding gain | +0.00 | **−1.25** | lower |
| Thesis | untied statistic alone (of 5) | 1, 3, 2 | 1, 2, 1 | lower |

**No regression on padding**, which is what the earlier work bought: decorating
a vague claim buys half a point less on Evidence than before, and on Thesis
padding now *costs* a point instead of being free. But the other two lines are
worse or flat. The tying gap — the grader's ability to tell a statistic used in
an argument from the same statistic left hanging — shrank from 1.33 to 0.67
points, and a bare statistic still scores 5 to 7 out of 10. The documented
weakness is intact.

## Set 2 (final test set): BEFORE vs AFTER

**This is the headline comparison.** Paired bootstrap: the same resampled
essays are scored under both models, so the interval reflects only what the
retrain changed.

| criterion | metric | before | after | change | 95% CI on the change |
|---|---|---|---|---|---|
| Thesis | MAE | 1.00 | 1.17 | +0.17 | [-0.50, +0.75] |
|  | within 1 | 83% | 50% | -33pp | [-75pp, +8pp] |
|  | exact | 17% | 33% | +17pp | [-17pp, +50pp] |
|  | QWK | 0.73 | 0.63 | -0.10 | [-0.37, +0.15] |
| Evidence | MAE | 1.92 | 1.75 | -0.17 | [-1.08, +0.75] |
|  | within 1 | 50% | 25% | -25pp | [-58pp, +8pp] |
|  | exact | 17% | 25% | +8pp | [-25pp, +42pp] |
|  | QWK | 0.63 | 0.63 | -0.00 | [-0.26, +0.17] |

**The retrain did not improve the final test set.** Thesis MAE rose 0.17 and
its QWK fell 0.10; Evidence MAE fell 0.17 and its QWK was flat. Every interval
comfortably contains zero, so with 12 essays none of these movements can be
told apart from noise — including the ones that look like improvements. The
honest summary is *no measurable change*, not a gain and not a loss.

The "within 1" drop on both criteria (83%→50%, 50%→25%) is the one movement
large enough to be worth explaining, and the per-essay tables explain it.

### Per-essay change, Thesis

| essay | true | before | after | \|error\| before | \|error\| after | change |
|---|---|---|---|---|---|---|
| 5 | 1 | 3 | 1 | 2 | 0 | -2 |
| 12 | 0 | 2 | 0 | 2 | 0 | -2 |
| 1 | 5 | 4 | 3 | 1 | 2 | +1 |
| 2 | 5 | 4 | 3 | 1 | 2 | +1 |
| 3 | 4 | 4 | 3 | 0 | 1 | +1 |
| 6 | 1 | 2 | 1 | 1 | 0 | -1 |
| 8 | 5 | 4 | 3 | 1 | 2 | +1 |
| 9 | 4 | 3 | 2 | 1 | 2 | +1 |
| 10 | 4 | 5 | 2 | 1 | 2 | +1 |
| 11 | 5 | 4 | 3 | 1 | 2 | +1 |
| 4 | 0 | 1 | 1 | 1 | 1 | 0 |
| 7 | 3 | 3 | 3 | 0 | 0 | 0 |

### Per-essay change, Evidence

| essay | true | before | after | \|error\| before | \|error\| after | change |
|---|---|---|---|---|---|---|
| 3 | 6 | 9 | 6 | 3 | 0 | -3 |
| 1 | 8 | 7 | 5 | 1 | 3 | +2 |
| 5 | 0 | 4 | 2 | 4 | 2 | -2 |
| 8 | 9 | 9 | 7 | 0 | 2 | +2 |
| 10 | 7 | 7 | 5 | 0 | 2 | +2 |
| 2 | 8 | 7 | 6 | 1 | 2 | +1 |
| 4 | 0 | 3 | 2 | 3 | 2 | -1 |
| 6 | 5 | 6 | 5 | 1 | 0 | -1 |
| 11 | 4 | 5 | 4 | 1 | 0 | -1 |
| 12 | 2 | 6 | 5 | 4 | 3 | -1 |
| 7 | 2 | 4 | 4 | 2 | 2 | 0 |
| 9 | 2 | 5 | 5 | 3 | 3 | 0 |

### Which essays changed most, and why

The retrain **traded one bias for another**:

- **The over-scoring it was meant to fix, it fixed.** Every weak essay that was
  over-scored moved the right way. Thesis: essay 5 (a vague "regulation is a
  word people use" paragraph) 3→1 against a true 1, and essay 12 (a list of
  statistics) 2→0 against a true 0 — both now exactly right. Evidence: essay 3
  9→6 against a true 6, essay 5 4→2, essay 4 3→2, essay 12 6→5.
- **But it introduced under-scoring of strong essays.** Six essays the team
  scored 4 or 5 on Thesis (1, 2, 8, 9, 10, 11) all dropped to 2 or 3, each
  moving from one point out to two. On Evidence, the three strongest essays
  fell away from their true scores: essay 1 (true 8) 7→5, essay 8 (true 9)
  9→7, essay 10 (true 7) 7→5.

That is a straightforward consequence of what was added: of the 11 new essays,
six carry a Thesis score of 0 or 1 and only three score 4 or 5. The added data
pulled the whole prediction range downward. It cured the over-scoring by making
the grader pessimistic rather than by making it discriminate better — which is
also why the tying gap in the padding probe shrank. The two biggest movers on
each criterion cancel out, which is why MAE barely moved while "within 1"
fell.

**No further tuning was done in response to these numbers**, by design. Fixing
the under-scoring now would mean choosing training data by its effect on the
final test set, which would burn it exactly as set 1 has been burned.

## Set 1 (development set): BEFORE vs AFTER

*Set 1's failure modes are what the new training rows were written to fix, so
its improvement is* **not** *evidence that the model got better. It is reported
for completeness, and as a demonstration of why the two sets are kept apart.*

| criterion | metric | before | after | change | 95% CI on the change |
|---|---|---|---|---|---|
| Thesis | MAE | 1.00 | 0.75 | -0.25 | [-0.92, +0.42] |
|  | within 1 | 83% | 92% | +8pp | [+0pp, +25pp] |
|  | exact | 42% | 33% | -8pp | [-50pp, +33pp] |
|  | QWK | 0.67 | 0.86 | +0.20 | [-0.04, +0.57] |
| Evidence | MAE | 1.67 | 1.42 | -0.25 | [-1.00, +0.42] |
|  | within 1 | 50% | 58% | +8pp | [-17pp, +33pp] |
|  | exact | 42% | 25% | -17pp | [-50pp, +17pp] |
|  | QWK | 0.71 | 0.81 | +0.09 | [-0.03, +0.28] |

Set 1 improved on both criteria: Thesis MAE 1.00→0.75 with QWK 0.67→0.86,
Evidence MAE 1.67→1.42 with QWK 0.71→0.81. The two essays whose failures
motivated the new training data are exactly the ones that improved most —
essay 7 (the fence-sitter) went from 4 points out to 1 on Thesis and from 4 to
1 on Evidence, and essay 11 (facts without an argument) improved on both.

**Set 1 says the fixes worked. Set 2 says they did not generalise.** Set 2 is
the one to believe: it was locked before any of this was measured, and its
essays were never read while choosing training data. Had only set 1 been kept,
this round would have been written up as a clear success.

## Set 1 (development set), BEFORE numbers in full

*The before/after comparison for this set is above. What follows is its
detailed BEFORE state, kept because the failure modes it describes are what
round two set out to fix.*

Models: the original 76-example models. The set was first run before intervals
existed; the point estimates below are unchanged, and the intervals were added
by the updated script.

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
write-up, which said the model cleared the baseline "by a wide margin".) The
same caveat applies to set 2 in both states.

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
