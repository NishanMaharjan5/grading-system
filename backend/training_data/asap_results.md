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

---

# Experiment: fine-tuned DistilBERT (pre-declared)

**This section was written and committed before the model was trained or the
test split was touched.** It fixes the design so the result cannot be chosen
after the fact. Nothing below was edited once test numbers existed.

## Why

The benchmark above found the *frozen* embedding contributes almost nothing:
word count alone matches the shipped pipeline. That is a statement about
frozen `all-MiniLM-L6-v2` features, not about transformers in general. A model
fine-tuned end to end on these essays learns its own representation, so it is
a fair test of whether the *approach* was the limit or only the frozen
features were.

## Design, fixed in advance

| | |
|---|---|
| model | `distilbert-base-uncased`, single-output regression head |
| size | 67.0M parameters, 268 MB fp32 |
| max length | **512 tokens** — 466/1783 essays (**26.1%**) still exceed it and are truncated (at 256, the frozen encoder's limit, 89.5% did) |
| target | score rescaled to 0-1 as `(score - 2) / 10`, MSE loss |
| prediction | rescale back, round to integer, clamp to 2-12 — same as every other system here |
| data | the **train split only**, from the locked `asap_split.json` (fingerprint `828dbd347ea2cf81`), same cleaned text as the benchmark above |
| grid | learning rate ∈ {2e-5, 3e-5}; up to 4 epochs; **best epoch and learning rate chosen by dev QWK** |
| batch size | 8 — measured to fit this machine; 16 falls off a memory cliff (117 s/step vs 1.37 s/step) |
| seed | 20260928, fixed |
| hardware | Apple M1, 8 GB, MPS |

## The two candidates, fixed in advance

- **(A) fine-tuned alone** — the DistilBERT prediction.
- **(B) fine-tuned + handcrafted Ridge, averaged** — `w · finetuned + (1-w) · ridge`,
  with **w chosen on dev** from {0.25, 0.5, 0.75}. The Ridge half is the
  handcrafted-features system already benchmarked above, refit on train only.

## Honesty conditions, fixed in advance

- The test split is scored **once**, at the end, after every choice is locked
  on dev.
- **This test set has already been used** to evaluate the five systems above.
  It is no longer virgin: each additional look raises the chance that some
  system wins by luck. These two candidates are pre-declared to limit that,
  but the honest reading is that test is now a *repeatedly used* benchmark,
  not a fresh one. A genuinely clean comparison would need a new split or a
  new corpus.
- A difference whose bootstrap interval includes zero will be reported as
  **not distinguishable**, whichever direction it points.


## Amendment, recorded before any test number existed

The first attempt ran on the Apple M1 at batch size 8 and froze the machine
after its first epoch; it was killed. The only number it produced was a dev
score after epoch 1 at learning rate 2e-5 (dev QWK 0.789, dev MAE 0.618).
**The test split was never scored**, and nothing was written.

The run moves to a Google Colab T4 GPU. **The design above is unchanged**:
the same model, 512-token limit, target and loss, learning-rate and epoch grid,
batch size 8, seed, both candidates, and one look at test. Only the hardware
line is superseded, and there is no time cap, because the grid now runs to
completion.

Three safeguards were added to the script. None of them changes what is
measured:

- **Pre-flight, before any training.** It checks that the split is the locked
  one, and that the two reference systems reproduce their published
  benchmark numbers exactly from the benchmark's cached embeddings. If they
  don't, the run stops before a new candidate has seen test.
- **No full run without a GPU** unless explicitly overridden.
- **One look at test.** A second run into a folder that already holds results
  is refused.

GPU training isn't bit-for-bit reproducible between runs, even with a fixed
seed. The output records the GPU, the library versions and a hash of the
scripts that ran.

---

# Result: fine-tuned DistilBERT

Run once on a Google Colab T4 (2026-09-28 19:20 UTC), exactly as pre-declared above
(including the Colab amendment). The scripts that ran are byte-identical to
the committed ones: the run recorded `asap_finetune.py` as `4b32dc9be05af209`
and `asap_benchmark.py` as `dded0a5d0eeab799`. The pre-flight passed: the split
was the locked one, and both reference systems reproduced their published test
QWK exactly (0.776 and 0.765) from the cached embeddings. Only then was
anything trained, and test was scored once, at the end. The run's own output
is in `results/asap_finetune.json`, which has every metric at full precision
with its interval, and in `results/finetune_log.txt`.

## Headline

**On the like-for-like comparison, the fine-tuned model is statistically
indistinguishable from a second human rater on this benchmark.** Both are
asked to predict what rater 1 said, on the raters' own 1-6 scale, for the same
268 test essays:

| predicting what rater 1 said (1-6) | QWK | MAE |
|---|---|---|
| fine-tuned DistilBERT (halved onto 1-6) | 0.710 [0.638, 0.769] | 0.358 |
| rater 2 (a human doing the same job) | 0.739 [0.662, 0.794] | 0.340 |
| **human advantage** | **+0.030 [−0.031, +0.093]** — not distinguishable | |

**This does not show that the model equals or beats a human grader.** The
human is still ahead on the point estimate. With 268 essays, the interval runs
from the model being ahead by 0.03 to the human being ahead by 0.09, and the
data cannot tell those apart. Not finding a difference is not the same as
finding none.

What did change is the size of the gap. On the same comparison, the shipped
frozen-embedding pipeline trailed a human by **0.152 [+0.073, +0.242]**, a gap
this data could clearly see. Fine-tuning shrank it to one it cannot.

## Test results

The test split was scored once. These are the same 268 essays, with the same
rounding and clamping and the same 2,000 bootstrap resamples (seed 0) as the
five earlier systems, so every interval below is directly comparable.

| system | MAE | QWK | exact | within 1 |
|---|---|---|---|---|
| (i) guess-the-mean | 1.220 [1.101, 1.340] | 0.000 [0.000, 0.000] | 19% [14%, 24%] | 75% [70%, 80%] |
| (ii) word count only | 0.675 [0.597, 0.757] | 0.757 [0.706, 0.798] | 44% [38%, 50%] | 89% [85%, 93%] |
| (iii) handcrafted only | 0.638 [0.560, 0.720] | 0.776 [0.733, 0.813] | 46% [40%, 52%] | 91% [88%, 95%] |
| (iv) frozen embeddings only | 0.993 [0.899, 1.082] | 0.621 [0.535, 0.687] | 27% [22%, 32%] | 77% [72%, 82%] |
| (v) frozen embeddings + handcrafted *(shipped)* | 0.716 [0.638, 0.806] | 0.765 [0.708, 0.809] | 41% [35%, 47%] | 88% [84%, 92%] |
| **(A) fine-tuned DistilBERT** | **0.552 [0.481, 0.627]** | **0.841 [0.799, 0.871]** | **52% [46%, 58%]** | **93% [90%, 96%]** |
| (B) fine-tuned + handcrafted, w = 0.5 | 0.549 [0.474, 0.627] | 0.832 [0.791, 0.862] | 51% [46%, 57%] | 94% [91%, 97%] |

The two comparisons pre-declared for the fine-tuned model, as paired
bootstrap differences in QWK:

| comparison | ΔQWK (95% CI) | distinguishable? |
|---|---|---|
| fine-tuned vs handcrafted only | +0.065 [+0.029, +0.099] | **yes** |
| fine-tuned vs frozen embeddings + handcrafted *(shipped)* | +0.076 [+0.041, +0.118] | **yes** |

**Fine-tuned DistilBERT (67M parameters) reaches test QWK 0.841 and beats
both reference systems distinguishably.** Both intervals clear zero with room
to spare. Neither comparison is close to the line, so the result doesn't rest
on the choice of interval method.

**The blend (B) is not the recommended system.** Averaging in the handcrafted
Ridge did not improve on the fine-tuned model alone. Its QWK is lower (0.832
against 0.841) and its MAE is essentially the same (0.549 against 0.552). A
paired A-vs-B interval was not pre-declared and was not computed, so this is
not a claim that B is worse, only that the handcrafted features added nothing
once the model was fine-tuned. On dev the blend was fractionally ahead (0.839
against 0.834) and on test it was fractionally behind; differences that small
flip from one sample to the next.

## How the model was chosen (dev only)

| lr | epoch | train MSE | dev QWK | dev MAE | epoch time |
|---|---|---|---|---|---|
| 2e-5 | 1 | 0.0207 | 0.796 | 0.618 | 57 s |
| 2e-5 | 2 | 0.0088 | 0.834 | 0.558 | 63 s |
| 2e-5 | 3 | 0.0077 | 0.829 | 0.562 | 62 s |
| 2e-5 | 4 | 0.0066 | 0.809 | 0.573 | 63 s |
| 3e-5 | 1 | 0.0212 | 0.825 | 0.558 | 63 s |
| **3e-5** | **2** | **0.0088** | **0.834** ← chosen | **0.581** | 63 s |
| 3e-5 | 3 | 0.0074 | 0.789 | 0.745 | 63 s |
| 3e-5 | 4 | 0.0060 | 0.819 | 0.539 | 63 s |

Both learning rates peaked on dev at epoch 2. After that, training loss kept
falling while dev QWK fell back: the model had started to fit the training
essays rather than the task. That is why the design picked the epoch on dev
rather than training to the end. The two epoch-2 runs tied at 0.834 to three
decimals (0.8337 against 0.8336 at full precision), so the choice between
them was close to a coin flip. Only the chosen one was ever scored on test.

Candidate B's blend weight, also chosen on dev:

| w (fine-tuned share) | dev QWK | dev MAE |
|---|---|---|
| 0.25 | 0.807 | 0.584 |
| **0.5** | **0.839** ← chosen | **0.528** |
| 0.75 | 0.837 | 0.528 |

## Truncation

**470 of the 1,783 essays (26%) reached the 512-token limit.** Per the count
in the pre-declaration, 466 of them were longer than the limit and lost their
endings. The model reached the numbers above with about a quarter of the
essays cut short. Whether reading them in full would help further was not
tested. For comparison, the frozen encoder's 256-token limit cut 89.5% of
these essays.

## Cost, and what serving it would take

| | |
|---|---|
| parameters | 67.0M |
| saved model | 269 MB |
| training | 8.9 min for the whole grid (8 epochs) on a Tesla T4; peak GPU memory 2.1 GB |
| GPU inference | 15 ms per essay, in batches of 8 |
| **CPU inference** | **477 ms per essay** on average (slowest 541 ms), scored one at a time on a single CPU thread, every essay padded to the full 512 tokens (the worst case) |
| cold load | 0.05 s, but measured straight after saving, so the file was probably still in memory; a genuinely cold start from disk will be slower |

**It meets a 5-second turnaround on CPU alone, with a wide margin.** Even in
the worst case (a full 512-token essay on one thread, no GPU), one essay takes
under half a second. Serving it doesn't need a GPU.

**Where the weights are.** The 269 MB of weights are on Google Drive, in
`My Drive/asap_finetune/output/best_model/`. They are not on the development
machine and not in the repo. If they are brought local, their place is
`backend/ml_experiments/`, which is gitignored. `*.safetensors` and any
`best_model/` directory are also ignored repo-wide, as a backstop.

**What it would take to use this in the live grading path:**

- **It is a holistic scorer, not a per-criterion one.** It gives one 2-12
  grade for ASAP set 1's prompt. It does not produce Thesis and Evidence
  scores, so it cannot replace the rubric model.
- **Grading the Essay 1 rubric this way would need data we don't have.** This
  model learned from 1,248 human-scored essays; the rubric model has 44 Thesis
  and 54 Evidence examples. It would take labelled essays at that scale for
  each criterion, a fine-tuning run per criterion (or one model with two
  outputs), and a locked test set of its own.
- **The engineering is modest.** Load the model once per process, as the
  current embedder is; tokenize with truncation at 512; predict and rescale.
  torch and transformers are already dependencies, because
  sentence-transformers needs them. The weights would live in artifact
  storage, not in git.
- **Teacher review stays** whatever model scores first.

## Caveats

- **This is not our grading task.** ASAP essay set 1 is a single holistic
  score, from grade 7-8 students, on a different prompt: a letter to a
  newspaper about the effects of computers. It is not the Thesis/Evidence
  rubric, and a fine-tuned DistilBERT is not the grading engine we ship.
- **What it does show:** fine-tuning BERT, the approach the original proposal
  specified and that the build replaced with frozen embeddings to fit the
  timeline, works well on a real human-scored benchmark when it is done
  properly. It clearly outperforms the frozen-embedding approach we shipped.
- **What it does not show:** that our production Thesis and Evidence scores
  would improve by the same amount. They would need their own fine-tuning
  effort (labelled data at scale, training and a locked test set), which has
  not been attempted within the project's one-to-two-week timeline.
- **This test set has now been used seven times.** The two new candidates
  were pre-declared, and both of the fine-tuned model's comparisons clear zero
  comfortably, but the test set is no longer fresh. A genuinely independent
  confirmation would need a new split or a new corpus.
- **One training run, one seed.** GPU training is not bit-for-bit
  reproducible, and a different seed would move these numbers somewhat. The
  intervals capture sampling of essays, not training randomness.

## Verdict

**On this benchmark, the evidence supports replacing frozen embeddings with a
fine-tuned model.** The gain over the shipped pipeline is distinguishable
(+0.076 QWK), and the gap to a human rater shrinks from clearly visible to
not detectable. The frozen-embedding pipeline does not hold up against it.

**In production, nothing changes yet.** The shipped Thesis/Evidence engine and
its models stay exactly as they are, because this result does not carry over
to them without data we don't have. It is a documented benchmark finding. It
also says what the grading engine's successor should be: a fine-tuned model
per criterion, once there are enough labelled essays to fine-tune it on.
