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
the current 35 labeled examples, classification on embeddings alone scored
*worse than always guessing the mean*; see `backend/app/grading/features.py`
for why, and the commit history for the numbers.

Feedback is template-based rather than model-generated: the scores come from a
small model trained on few examples, and fluent prose on top of an uncertain
number would sound more authoritative than the grade deserves.

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
    grading/     embedder, features, per-criterion model store, engine, feedback templates
    models/      User, Rubric, RubricCriterion, Submission, Grade
    routes/      auth, rubrics, submissions (incl. the teacher review flow)
  migrations/    Alembic; see migrations/README for the workflow
  scripts/       train_grader.py, seed_demo_rubric.py
  tests/         pytest suite
  training_data/ labeled sample answers
frontend/        React + Vite
```

## Known limitations

- **The grader is trained on 35 labeled examples.** It beats a
  guess-the-mean baseline but not by much, and the numbers move several points
  if a single example changes. More labeled data is the highest-value
  improvement available.
- **Approval is final.** Re-approving returns 409 and there is no correction
  path for a mistaken approval; that would need a revise endpoint with an
  audit trail.
- **Code submissions are not graded.** The rubric carries a `code` type, but
  the sandboxed test runner does not exist yet — only the text path works.
- **Tests build their schema from migrations, but `flask db migrate` is not
  run in CI**, so a model change with no matching revision would not be caught
  automatically.
