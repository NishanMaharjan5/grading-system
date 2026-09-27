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
  scripts/       train_grader.py, seed_demo_rubric.py, reset_dev_data.py
  tests/         pytest suite
  training_data/ labeled sample answers
frontend/        React + Vite
```

## Known limitations

- **The grader is trained on 76 labeled examples** (33 Thesis, 43
  Evidence). Leave-one-out MAE is 0.85 for Thesis (baseline 1.39) and 1.49 for
  Evidence (baseline 3.30). With this few examples, the numbers move noticeably
  if a single example changes. Tuning has stopped here: further gains on a
  dataset this size are small.
- **The Evidence grader rewards what evidence looks like, not whether it is
  relevant or true.** Padding a vague answer with names and numbers used to
  lift it from 0-2 to 4-8 out of 10. Adversarial and paired examples cut that
  to 1-2 points. But a fluent sentence that states a statistic without using it
  in an argument still scores 5-7 out of 10: the low-scored halves of the pairs
  are sentence fragments, so the model mostly learned that fragments score low,
  not that unconnected numbers do. The review step is the backstop: no AI score
  reaches a student unapproved.
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
