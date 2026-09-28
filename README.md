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
reasoning connectives, citation words). BERT is a frozen feature extractor; it
is never fine-tuned.

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
                 evaluate_holdout.py + leakage_guard.py + padding_probe.py
                 (held-out evaluation; none of them train or save a model),
                 asap_benchmark.py (independent public-dataset benchmark)
  tests/         pytest suite
  training_data/ labeled sample answers, the two holdout sets,
                 holdout_results.md, asap_results.md + asap_split.json,
                 and results/ (saved evaluation runs)
  data/          benchmark corpora -- gitignored, never committed
frontend/        React + Vite
```

## Independent benchmark (ASAP-AES)

Every other number in this project comes from essays the team wrote and scored
itself. To check the *method* against strangers' work, the same featurisation
was benchmarked on **ASAP-AES set 1** — 1,783 public essays scored 2-12 by two
real teachers. Full write-up: **[backend/training_data/asap_results.md](backend/training_data/asap_results.md)**;
run with `scripts/asap_benchmark.py`. The split was committed before anything
was fit, and test was scored once.

| system (test, n=268) | MAE | QWK |
|---|---|---|
| guess-the-mean | 1.220 | 0.000 |
| **word count only** | 0.675 | **0.757** |
| handcrafted only | 0.638 | 0.776 |
| embeddings only | 0.993 | 0.621 |
| **embeddings + handcrafted (shipped)** | 0.716 | **0.765** |

**The embeddings are not doing the work; length is.** A Ridge model on word
count alone is not statistically distinguishable from the full pipeline
(ΔQWK 0.008), embeddings alone are *worse* than word count, and adding
embeddings to the handcrafted features does not improve on those features
alone (ΔQWK −0.011, CI [−0.055, +0.026]). Sweeping the Ridge alpha on dev
does not rescue them. This confirms on independent data what
`app/grading/features.py` already said: these embeddings encode topic, not
quality.

Two honest caveats on it: ASAP is grade 7-8 students on a different prompt
with a single holistic score, so **this validates the method, not the Essay 1
rubric**; and word count predicting ASAP scores well is a property of that
corpus, not a licence to grade by length — the padding probe shows where that
leads.

It also does **not** beat human graders. Compared naively, the model (QWK
0.765) looks to edge the raters' agreement with each other (0.739) — but the
model is scored against *two raters summed*, a smoother target than the single
rater each human is judged against. On a like-for-like task, predicting what
rater 1 said, the model scores 0.587 against a human's 0.739; the human is
ahead by 0.152, CI [+0.073, +0.242].

## Known limitations

- **The encoder silently truncates long submissions.** `all-MiniLM-L6-v2`
  reads at most **256 tokens (~226 words)**; anything beyond is dropped
  without warning. Two 377-word essays that differ only after that point embed
  *identically*. Submissions accept 50,000 characters and the seeded
  Reflection rubric asks for 200-300 words, so this is reachable in normal
  use. On ASAP it affects **89.5% of essays**, with the median essay losing
  40% of itself — and yet chunked embeddings (embed each ~170-word piece, then
  mean-pool) scored **identically to the single pass on dev** (MAE 0.700 both,
  QWK 0.781 vs 0.780). So production is deliberately left unchanged: fixing
  truncation is not worth ~2.6x the embedding cost when the embedding
  contributes almost nothing to begin with. The hand-crafted features do read
  the whole text, which is part of why this never showed up.
- **The grader is trained on 98 labeled examples** (44 Thesis, 54 Evidence).
  Leave-one-out MAE is 1.11 for Thesis (baseline 1.48) and 1.52 for Evidence
  (baseline 3.07). With this few examples the numbers move noticeably if a
  single example changes.
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
  six of set 2's strongest essays by two points each. The review step is the
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
