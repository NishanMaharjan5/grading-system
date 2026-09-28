# Independent benchmark: ASAP-AES set 1

The first check of this project's grading pipeline against **essays nobody
here wrote, scored by people who have never seen this project**. Everything
previously reported came from essays the team wrote and scored itself, on one
prompt, which cannot tell us whether the method works — only whether it
reproduces our own judgment.

- **Data:** ASAP-AES (Hewlett Foundation / Kaggle, 2012), essay set 1 — 1,783
  persuasive essays by US grade-7/8 students, argued to a newspaper about the
  effects of computers. Each was scored 1-6 by two human raters; the label is
  their sum, 2-12.
- **Script:** `scripts/asap_benchmark.py`. It touches no database, builds no
  Flask app, and never reads or writes `ml_models/`. The production models are
  untouched by this experiment.
- **Split:** committed in `f3c5d17` **before anything was fit** — train 1,248,
  dev 267, test 268, seed 20260928, stratified by score. Every choice was made
  on dev. Test was scored once, at the end, and nothing was tuned afterwards.
- **What it measures:** the featurisation we actually ship — the same
  `embedder.py` and `features.py` the live grader uses, with Ridge at the
  production alpha of 1.0.

## Headline

**The embeddings are not carrying the grading. Counting words is.**

On a public dataset with real teacher scores, a Ridge model on **word count
alone** scores QWK 0.757. The full shipped pipeline scores 0.765 — a
difference that is **not statistically distinguishable**. Embeddings on their
own score 0.621, *worse than word count*, and adding them to the handcrafted
features does not improve on those features alone.

This is the same finding the project recorded early on (embeddings encode
topic, not quality), now confirmed on independent data, and it is stronger
than expected: the embedding component is not merely weak, it is close to
inert once length is known.

## Cleaning

ASAP redacts named entities as `@PERSON1`, `@CAPS1`, `@NUM1` and similar.
**1,663 of 1,783 essays (93.3%) contained at least one; 17,224 occurrences
were replaced** with neutral lowercase phrases ("someone", "an organization",
"a number"), and `@CAPS` — a redacted capitalised word with no sensible
neutral noun — was removed.

This is not cosmetic. `@NUM1` contains a digit, so leaving the placeholders in
would have let `digit_count` reward an essay for *having been anonymised*.
Encoding artefacts were also normalised: 83 essays held non-ASCII characters
(mostly curly quotes) and one held `â€`-style mojibake.

## Truncation

**The encoder truncates 9 out of 10 of these essays, silently.**

| | |
|---|---|
| `all-MiniLM-L6-v2` max sequence length | **256 tokens** |
| essays exceeding it | **1,596 / 1,783 (89.5%)** |
| token length | median 428, mean 431, max 983 |
| what the median essay loses | ~172 tokens, **40% of itself** |

So the single-pass embedding of a typical ASAP essay represents roughly its
first 60% and discards the rest — including, in a persuasive essay, the
conclusion.

### Does fixing it help? No.

Chunked embeddings (split into ~170-word pieces, embed each, mean-pool —
2.6 chunks per essay on average) were compared with the single pass **on dev**:

| variant | dev MAE | dev QWK | dev exact | dev within 1 |
|---|---|---|---|---|
| single-pass | 0.700 | 0.781 | 43% | 88% |
| chunked | 0.700 | 0.780 | 43% | 88% |

**Identical MAE, and a QWK difference of 0.001 in favour of the single pass.**
Recovering the 40% of text that truncation throws away changes nothing
measurable. Single-pass was therefore used on test.

That reads as a contradiction until you put it next to the headline: if the
embedding contributes almost nothing beyond length, then neither does the part
of it that got cut off.

## Test results

Scored once. 268 essays, predictions rounded and clamped to 2-12, QWK over the
full 2-12 label range, paired bootstrap 95% intervals from 2,000 resamples
(fixed seed).

| system | MAE | QWK | exact | within 1 |
|---|---|---|---|---|
| (i) guess-the-mean | 1.220 [1.101, 1.340] | 0.000 [0.000, 0.000] | 19% [14%, 24%] | 75% [70%, 80%] |
| (ii) word count only | 0.675 [0.597, 0.757] | 0.757 [0.706, 0.798] | 44% [38%, 50%] | 89% [85%, 93%] |
| (iii) handcrafted only | 0.638 [0.560, 0.720] | 0.776 [0.733, 0.813] | 46% [40%, 52%] | 91% [88%, 95%] |
| (iv) embeddings only | 0.993 [0.899, 1.082] | 0.621 [0.535, 0.687] | 27% [22%, 32%] | 77% [72%, 82%] |
| (v) embeddings + handcrafted **(what we ship)** | 0.716 [0.638, 0.806] | 0.765 [0.708, 0.809] | 41% [35%, 47%] | 88% [84%, 92%] |

### Which differences are real

| comparison | ΔQWK (95% CI) | ΔMAE (95% CI) | distinguishable? |
|---|---|---|---|
| embeddings + handcrafted vs embeddings only | +0.144 [+0.093, +0.205] | +0.276 [+0.179, +0.366] | **yes** |
| embeddings + handcrafted vs handcrafted only | -0.011 [-0.055, +0.026] | -0.078 [-0.168, +0.004] | no |
| embeddings only vs handcrafted only | -0.155 [-0.243, -0.085] | -0.354 [-0.466, -0.243] | **yes** |
| handcrafted only vs word count only | +0.019 [-0.010, +0.051] | +0.037 [-0.015, +0.097] | no |
| word count only vs guess-the-mean | +0.757 [+0.706, +0.798] | +0.545 [+0.422, +0.675] | **yes** |

Reading these:

- **Adding embeddings to the handcrafted features is not distinguishable from
  the handcrafted features alone** — and both point estimates favour dropping
  them (QWK −0.011, MAE −0.078 worse *with* embeddings).
- **Embeddings alone are clearly worse than handcrafted features alone**
  (QWK −0.155, interval well clear of zero).
- **Handcrafted features are not distinguishable from word count alone.** The
  other eight features are not earning their place on this dataset either.
- The only decisive win anywhere is **anything over guess-the-mean**.

### Is the weak embedding result just an untuned alpha?

Ridge alpha was held at the production value of 1.0 rather than tuned for
ASAP, because the question is how the shipped pipeline behaves. To check that
this is not what sinks the embeddings, alpha was swept **on dev only**:

| Ridge alpha | dev MAE | dev QWK |
|---|---|---|
| 0.1 | 0.921 | 0.605 |
| 1.0 *(production)* | 0.921 | 0.608 |
| 10.0 | 0.884 | 0.627 |
| 100.0 | 0.820 | 0.661 |
| 1000.0 | 0.880 | 0.569 |

Tuning helps — the best dev QWK is 0.661 at alpha 100, up from 0.608 — but
that is still far below handcrafted-only (0.776) and word-count-only (0.757)
on test. **The conclusion survives: the embeddings underperform a word count
whether or not the regulariser is tuned.**

## Human ceiling

The two raters agree with each other at **QWK 0.739** on the test essays
(0.721 across all 1,783), agreeing exactly 66% of the time and within one
point essentially always.

Set against the model's headline 0.765, that invites the claim that the model
has matched or beaten human graders. **It has not, and the comparison is
rigged.** The model is scored against `domain1_score`, which is *two raters
added together* — an averaged, smoothed target. Each human is scored against
a *single* other rater, which is the noisiest possible target. The model is
being graded on an easier exam.

Put both on the same task — predict what rater 1 said, on the raters' own 1-6
scale:

| predicting what rater 1 said (1-6 scale) | QWK | MAE |
|---|---|---|
| our model (halved onto 1-6) | 0.587 [0.487, 0.670] | 0.451 |
| rater 2 (a human doing the same job) | 0.739 [0.662, 0.794] | 0.340 |
| **human advantage** | **+0.152 [+0.073, +0.242]** | |

**The human is better by 0.152 QWK, and the interval clears zero.** On a
like-for-like task, this pipeline is measurably worse than a second human
rater. The naive comparison would have reversed that conclusion.

## What this does and does not establish

**Does:**

- The method beats guessing decisively on real, independently scored essays.
- The embedding half of the featurisation is close to inert on this data; a
  word count reproduces nearly all of the performance.
- Truncation, despite affecting 90% of essays, costs nothing measurable —
  because the embeddings contribute little in the first place.
- A second human rater is still measurably better than the model at the same
  task.

**Does not:**

- **Different population.** US grade 7-8 students on a computers-and-society
  prompt. Our demo rubric is a different prompt and a different cohort.
- **Different construct.** ASAP set 1 is a single *holistic* 2-12 score. We
  predict per-criterion Thesis (0-5) and Evidence (0-10). A pipeline that
  tracks holistic quality may do better or worse on a single criterion, and
  this says nothing about which.
- **Anonymised text.** 93% of these essays have had entities stripped, which
  is exactly the kind of concrete naming the Evidence criterion is meant to
  reward. If anything this handicaps the features we care most about.
- **This validates the method, not the Essay 1 rubric.** No conclusion here
  transfers to the production rubric's numbers.
- **Length is a confound, not a scoring rule.** Word count predicting ASAP
  scores well is a known property of the corpus (longer essays tend to be
  better ones), not evidence that length *should* determine a grade. A grader
  that leans on length is easy to game by padding — which the project's own
  padding probe already shows.

## Reproducing

```bash
cd backend
./venv/bin/python scripts/asap_benchmark.py --write-split   # already committed
./venv/bin/python scripts/asap_benchmark.py
```

The dataset is licensed and is not in the repo; `backend/data/` is gitignored.
Place `essays.xlsx` at `backend/data/asap/essays.xlsx`. Embeddings are cached
under `backend/data/asap/.cache/`. The run is deterministic: repeated runs
produce byte-identical output, and the script refuses to start if the split
file's ids no longer match the fingerprint recorded in it.
