# Automated Assignment Grading and Feedback System

Students submit work against a teacher-authored rubric. A grading engine scores
each rubric criterion and drafts plain-English feedback, a teacher reviews and
approves or overrides that suggestion, and only then does the student see a
grade.

**Stack:** Flask + SQLAlchemy + PostgreSQL, React/Vite, JWT auth with
student/teacher roles.

## How grading works

Text submissions are scored by a **Ridge regression per rubric criterion**,
trained on frozen Sentence-BERT (`all-MiniLM-L6-v2`) embeddings concatenated
with hand-crafted features (digit/percentage/year counts, hedging words,
reasoning connectives, citation words). In the shipped engine BERT is a frozen
feature extractor and is never fine-tuned. A fine-tuned BERT was evaluated
separately and did much better on a public benchmark; *How the approach was
evaluated* below explains why it still doesn't ship.

Regression rather than classification because rubric scores are ordinal —
predicting 0 for a true 5 should not cost the same as predicting 4. Measured on
the first 35 labeled examples, classification on embeddings alone scored
*worse than always guessing the mean*; see `backend/app/grading/features.py`
for why, and the commit history for the numbers.

Feedback is template-based rather than model-generated: the scores come from a
small model trained on few examples, and fluent prose on top of an uncertain
number would sound more authoritative than the grade deserves.

**Code submissions** (Python only) take a different path entirely: no model is
involved. Each criterion carries stdin/stdout test cases, the submitted program
is run against them in a subprocess, and the criterion is worth the fraction of
its tests that pass. Wrong answers, crashes and timeouts are all ordinary
grades; only the harness itself breaking sends a submission to `grading_failed`.

**Security: [backend/SANDBOX.md](backend/SANDBOX.md) has the full account.**
In short, student code runs under **macOS Seatbelt (`sandbox-exec`)**, which
enforces the file and network rules **at the kernel level**. The student's
process cannot read anything in the repo (including `backend/.env`, which holds
`JWT_SECRET` and the database credentials), cannot write outside its own
scratch directory, cannot open any network connection, and cannot start child
processes. This was verified with raw syscall bypass tests: the test programs
call libc directly through `ctypes`, skipping every Python-level guard, and the
kernel still refuses them with `EPERM`. Positive controls prove the same probes
succeed when the sandbox is removed, so the tests can't pass by being broken.
Kernel-enforced resource limits (CPU, file size, process count) and in-process
Python guards sit on top as defense-in-depth.

This is still not a container, and it has real limits: it is **macOS-only**,
`sandbox-exec` is a **deprecated (though functional) Apple API**, and memory is
capped by polling rather than by the kernel. On Linux, the equivalent would be
**Landlock** for path-based file rules, **seccomp-bpf** for denying process and
network syscalls, or a **container** for all of it. Until one of those is
built, grading refuses to run on any platform without Seatbelt, rather than
running student code unsandboxed.

## How the approach was evaluated

The grading approach was tested several ways, each time against something it
had to beat. Every result is reported below, whether or not it favoured what
was built:

| approach | tested on | result |
|---|---|---|
| Classifier on frozen Sentence-BERT embeddings | the first 35 labelled answers | **worse than always guessing the mean** (Thesis MAE 2.50 vs 1.39) |
| Ridge on frozen embeddings + handcrafted features (**what ships**) | leave-one-out on 98 labelled answers | beats guessing (Thesis MAE 1.11 vs 1.48; Evidence 1.52 vs 3.07) |
| More training data for the shipped model | a held-out set locked before retraining | **no measurable change** |
| The shipped features, on public data | ASAP-AES, 268 test essays scored by teachers | QWK 0.765, but **word count alone reaches 0.757** |
| **Fine-tuned DistilBERT** | the same 268 essays | **QWK 0.841**: distinguishably better, and **statistically indistinguishable from a second human rater** |
| The shipped pipeline on a **second topic** (Essay 2, AI tools in schoolwork) | a holdout locked before any Essay 2 training data | **Thesis generalised** (QWK 0.79); **Evidence did not** (gain over baseline 0.12, CI [−1.88, +1.62] — not distinguishable from zero) |

The first three rows use answers the team wrote and scored itself; that
write-up is `backend/training_data/holdout_results.md`. The last two use
**ASAP-AES set 1**: 1,783 public essays, each scored by two real teachers.
The ASAP split was committed before anything was fit. Every choice was made
on a separate dev split, and the test split was scored once per system. The
full write-up is
**[backend/training_data/asap_results.md](backend/training_data/asap_results.md)**.

### Frozen embeddings don't carry the grading

| system (ASAP test, n=268) | MAE | QWK |
|---|---|---|
| guess-the-mean | 1.220 | 0.000 |
| word count only | 0.675 | 0.757 |
| handcrafted only | 0.638 | 0.776 |
| frozen embeddings only | 0.993 | 0.621 |
| frozen embeddings + handcrafted *(shipped)* | 0.716 | 0.765 |
| **fine-tuned DistilBERT** | **0.552** | **0.841** |

A Ridge model on word count alone can't be told apart from the shipped
pipeline (ΔQWK 0.008). Frozen embeddings on their own do *worse* than word
count. Adding them to the handcrafted features doesn't improve on those
features alone (ΔQWK −0.011, CI [−0.055, +0.026]), and tuning the Ridge alpha
doesn't change that. Frozen `all-MiniLM-L6-v2` vectors capture what an essay
is about, not how good it is. The project's own data had already pointed that
way (see `app/grading/features.py`).

### Fine-tuning works

The original proposal specified fine-tuning BERT; the build replaced it with
frozen embeddings to fit the timeline. Fine-tuning DistilBERT end to end on
the same ASAP training essays changes the picture. The design was
pre-declared before training, and the model was run once on a Colab GPU. It
beats the shipped pipeline by **+0.076 QWK [+0.041, +0.118]** and the
handcrafted features by +0.065 [+0.029, +0.099]. Both intervals are clear of
zero.

Raw QWK flatters any model on this dataset, because the model is scored
against *two raters' scores added together*, while each human is compared
with one other rater's score. So the model and a human are put on the same
task: predict what rater 1 said.

| predicting what rater 1 said | QWK | gap to a second human rater |
|---|---|---|
| shipped frozen-embedding pipeline | 0.587 | 0.152 [+0.073, +0.242]: clearly behind |
| fine-tuned DistilBERT | 0.710 | 0.030 [−0.031, +0.093]: **not distinguishable** |
| a second human rater | 0.739 | |

**Fine-tuned, the model reaches near-human agreement: on this benchmark it is
statistically indistinguishable from a second human rater.** That is not the
same as matching one. The human is still ahead on the point estimate, and
with 268 essays a human advantage of up to 0.09 can't be ruled out.

### Does it work on a second topic?

Every result above comes from one subject: social-media regulation. A grader
can look like it judges argument quality when it has only learned one debate's
vocabulary. **Essay 2** is the control — a different prompt (*should students
be allowed to use AI tools like ChatGPT for schoolwork?*), the same pipeline,
no code changed. Full write-up:
**[backend/training_data/essay2_results.md](backend/training_data/essay2_results.md)**.

The holdout was committed before a single Essay 2 training row existed, and
scored once. The two rubrics' models are genuinely separate — the engine keys
them by criterion row id, so retraining on the combined file left Essay 1's
models **byte-identical**.

| Essay 2 holdout (8 essays) | MAE | baseline | gain over baseline | QWK |
|---|---|---|---|---|
| Thesis | **0.88** | 1.62 | **0.75** [−0.25, +1.38] | **0.79** |
| Evidence | 2.50 | 2.62 | 0.12 [−1.88, +1.62] | 0.55 |

**Thesis generalised to the new topic. Evidence did not.** State it that way
round, per criterion, and not as "the method generalises":

- **Thesis reached QWK 0.79** on a subject it had never been trained on, with
  seven of eight essays within one point, from models fit on 15 examples.
- **Evidence's gain over guessing the mean is 0.12, interval [−1.88, +1.62]** —
  not distinguishable from zero. On this topic it is no better than predicting
  a constant.

**The Evidence failure mode reproduced on a topic the model never trained on**,
which is the more important half of the result. Every Evidence error was an
over-score, and the worst were the documented weakness exactly: a personal
anecdote with no evidence at all scored **6/10** against a true 0, and three
real facts listed with no argument connecting them scored **9/10** against a
true 5.

A weakness that survives a change of subject is a limitation of the method,
not a quirk of the social-media training set. That is stronger evidence than
anything the Essay 1 holdouts could give, because there the training and test
essays shared a topic.

### What this means for the shipped engine

**Nothing in the shipped engine changed because of this.** It is a benchmark
finding reported alongside the model that ships, not a late swap. The
fine-tuned model gives one holistic 2-12 score for a single ASAP prompt. It
learned from 1,248 teacher-scored essays by grade 7-8 students, and it
produces no Thesis or Evidence scores. Doing the same for those criteria would
need labelled essays on that scale (the rubric model has 44 and 54), a
fine-tuning run, and a locked test set of their own. That hasn't been
attempted within the project's one-to-two-week timeline.

What the result does establish is where the weak point was: the frozen
embeddings, not the idea of using BERT. It also shows that a fine-tuned
successor would be practical to serve. It takes under half a second per
essay on a single CPU thread in the worst case, with no GPU. Its 269 MB of
weights are kept out of the repo.

Three caveats apply. ASAP set 1 differs from the Essay 1 rubric in prompt,
age group and scoring scheme. Its test split has now been used for seven
systems. And the fine-tuned result comes from one training run with one seed.

## Setup

Requires Python 3.11, Node 20, and PostgreSQL 16.

```bash
# PostgreSQL (Homebrew; no containers)
brew install postgresql@16 && brew services start postgresql@16
createdb grading_system_dev
psql -d postgres -c "CREATE ROLE grading_test LOGIN PASSWORD 'testpass';"

# Backend
cd backend
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
cp .env.example .env          # then fill in DATABASE_URL and JWT_SECRET
FLASK_APP=run.py ./venv/bin/flask db upgrade
./venv/bin/python run.py      # http://localhost:8000

# Frontend
cd frontend && npm install && npm run dev   # http://localhost:5173
```

Generate a JWT secret with
`python3 -c "import secrets; print(secrets.token_hex(32))"`.

Set `TEACHER_SIGNUP_CODE` in `.env` to stop anyone registering as a teacher and
approving their own grades. Left empty, the gate is off.

## Tests

The suite runs against its own database, builds the schema by applying the
migrations, and truncates between tests — so it never depends on run order and
never touches your dev data. It also redirects the model store at a temporary
directory, so a test run cannot overwrite trained models in
`backend/ml_models/`.

```bash
cd backend
make test        # everything (~80s; most of it is loading Sentence-BERT once)
make test-fast   # skips the tests that load Sentence-BERT (~15s)
```

Point it at a different database with
`make test TEST_DATABASE_URL=postgresql+psycopg2://user:pass@host/dbname`.
**Anything in that database is destroyed.**

Without `make`:

```bash
cd backend
DATABASE_URL=postgresql+psycopg2://grading_test:testpass@localhost:5432/grading_system_test \
  JWT_SECRET=test-secret ./venv/bin/pytest
```

## Other commands

```bash
cd backend
make run                       # dev server
make train                     # retrain from training_data/sample_answers.json
make upgrade                   # apply migrations
make migrate m="add a column"  # autogenerate a revision -- read it before applying
```

## Two databases

`grading_system_dev` is persistent and holds the demo rubric and real
submissions. `grading_system_test` is disposable and is wiped by every test
run. They are deliberately separate: an early test run using a shared database
destroyed live data.

## Layout

```
backend/
  app/
    grading/     embedder, features, model store, engine, feedback templates,
                 code_runner + _sandbox_runner (the code path)
    models/      User, Rubric, RubricCriterion, TestCase, Submission, Grade
    routes/      auth, rubrics, submissions (incl. the teacher review flow)
  SANDBOX.md     what the code sandbox does and does not contain
  migrations/    Alembic; see migrations/README for the workflow
  scripts/       train_grader.py, seed_demo_rubric.py, reset_dev_data.py,
                 backfill_word_limits.py,
                 evaluate_holdout.py + leakage_guard.py + padding_probe.py
                 (held-out evaluation; none of them train or save a model),
                 asap_benchmark.py + asap_finetune.py (public-dataset benchmark;
                 the fine-tune runs on Colab via asap_finetune_colab.ipynb)
  tests/         pytest suite
  training_data/ labeled sample answers, the two holdout sets,
                 holdout_results.md, essay2_results.md, asap_results.md
                 + asap_split.json, and results/ (saved evaluation runs)
  data/          benchmark corpora -- gitignored, never committed
  ml_experiments/ fine-tuning outputs and weights -- gitignored
frontend/        React + Vite
```

## Known limitations

- **The shipped grader is not the best approach measured.** It uses frozen
  embeddings, and on the public benchmark it does no better than a word count
  and clearly trails a human rater. A fine-tuned model did distinguishably
  better (see *How the approach was evaluated*). Fine-tuning the Thesis and
  Evidence criteria would need far more labelled essays than the 98 available.
  Until then, teacher review is what makes the scores safe to use: no AI score
  reaches a student unapproved.
- **The encoder reads at most ~226 words, and word limits are how that is
  handled.** `all-MiniLM-L6-v2` takes 256 tokens (~226 words) and drops the
  rest without warning: two 377-word essays differing only after that point
  embed *identically*. On ASAP it would affect **89.5% of essays**, the median
  losing 40% of itself.

  **The mitigation is a word limit on the rubric, not chunked embeddings in
  production.** Text rubrics carry optional `min_words`/`max_words`; a new one
  defaults to **max 200**, chosen to sit under the ~226-word read so the
  grader sees the whole essay. Submissions outside the range are refused — in
  the form as you type, and again at the API, which is the authority. Essay 1
  and Essay 2 are backfilled to 20-200 (`scripts/backfill_word_limits.py`).

  Chunked embeddings were the obvious alternative and were measured: embedding
  each ~170-word piece and mean-pooling scored **identically to the single
  pass on dev** (MAE 0.700 both, QWK 0.781 vs 0.780) for ~2.6x the cost. There
  is nothing to recover by reading further when the embedding contributes
  almost nothing to begin with, so capping the input is the cheaper and more
  honest fix: it keeps the grader's view complete instead of hiding that it
  was partial.

  The limits are a default, not a cage. A teacher can set a higher cap, and
  the rubric form says plainly what it costs — past 226 words the grader stops
  reading and the end of those essays is scored by the teacher alone. Rubrics
  created before this feature keep null limits and stay unrestricted, and no
  submission already recorded is re-checked. The hand-crafted features always
  read the whole text, which is part of why this went unnoticed for so long.
- **The grader is trained on 128 labeled examples across two rubrics** —
  Essay 1 has 98 (44 Thesis, 54 Evidence) and Essay 2 has 30 (15 each).
  Leave-one-out MAE on Essay 1 is 1.11 for Thesis (baseline 1.48) and 1.52 for
  Evidence (baseline 3.07); on Essay 2, 0.73 and 2.40. With this few examples
  the numbers move noticeably if a single example changes, and Essay 2's 15
  per criterion is a quarter of Essay 1's.
- **Adding training data did not improve the held-out result.** Two sets of 12
  paragraph-length essays were built (`backend/training_data/`, full write-up
  in `holdout_results.md`, run with `scripts/evaluate_holdout.py`). Set 1 was
  studied and its failures used to write 22 new training rows; set 2 was
  committed untouched beforehand and read only afterwards. After retraining,
  **set 1 improved on both criteria and set 2 did not move**:

  | | set 1 (studied) | set 2 (untouched) |
  |---|---|---|
  | Thesis MAE | 1.00 → 0.75 | 1.00 → 1.17 |
  | Evidence MAE | 1.67 → 1.42 | 1.92 → 1.75 |

  Every one of those changes has a bootstrap 95% interval containing zero, so
  with 12 essays none is distinguishable from noise. Set 2 is the one to
  believe, and it says the round bought nothing measurable. Keeping only set 1
  would have made this look like a clear win.
- **The Evidence grader rewards what evidence looks like, not whether it is
  relevant or true.** Padding a vague answer with names and numbers once
  lifted it from 0-2 to 4-8 out of 10; adversarial and paired examples cut
  that, and `scripts/padding_probe.py` now measures it on fresh probes: the
  padding gain is +1.75 points on Evidence (was +2.25), and on Thesis padding
  now costs a point rather than being free. But **a statistic stated without
  being used in an argument still scores 5-7 out of 10**, and the gap between
  a tied and an untied statistic narrowed (1.33 → 0.67 points). What the
  retrain mostly did was make the grader more pessimistic — it fixed the
  over-scoring of weak essays and started under-scoring strong ones, dropping
  six of set 2's strongest essays by two points each. **This is now confirmed
  on a second, unrelated topic:** on the Essay 2 holdout every Evidence error
  was an over-score, a personal anecdote with no evidence scored 6/10, and
  three facts listed without an argument scored 9/10. The review step is the
  backstop: no AI score reaches a student unapproved.
- **Both holdout sets were written by the project team**, essays and scores
  alike, and every essay in the training data and both holdout sets answers
  the same prompt about regulating social media. So this measures
  generalisation against the team's own judgment on one question, not
  agreement with real teachers, and says nothing about a different prompt.
  QWK on 12 essays is indicative at best.
- **Approval is final.** Re-approving returns 409 and there is no correction
  path for a mistaken approval; that would need a revise endpoint with an
  audit trail.
- **Code grading is macOS-only** and relies on `sandbox-exec`, an Apple API
  that is deprecated but functional. On any other platform, or if a future
  macOS removes it, submissions go to `grading_failed` for manual grading
  instead of running unsandboxed. A Linux port needs Landlock, seccomp-bpf or
  a container; see [backend/SANDBOX.md](backend/SANDBOX.md).
- **The sandbox is kernel-enforced, but it is not a container.** The profile
  denies reads of the repo and common credential directories rather than
  everything outside the scratch directory, because a true default-deny breaks
  whenever Apple's Python toolchain changes. Student code still runs as the
  server's OS user, so anything the profile doesn't mention is reachable. It
  can also confirm whether a guessed filename exists in the repo, though it
  can't read or list it.
- **Memory is capped by polling, not by the kernel**, because macOS refuses to
  set `RLIMIT_AS`/`DATA`/`RSS` and Seatbelt has no memory rule. There is a
  ~100ms window in which a program can exceed the cap before it is killed.
- **Only Python submissions are supported, under the system Python 3.9** —
  not the app's venv, which the sandbox makes unreadable. Code rubrics should
  use the standard library and 3.9 syntax. Test cases compare stdout only, not
  exit codes, stderr or written files.
- **Test cases have no visible/hidden distinction.** They are withheld from
  students entirely, so a student cannot see any example before submitting.
- **Tests build their schema from migrations, but `flask db migrate` is not
  run in CI**, so a model change with no matching revision would not be caught
  automatically.
