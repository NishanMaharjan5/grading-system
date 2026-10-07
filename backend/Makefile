# Run from backend/ -- everything here assumes ./venv exists.
VENV := ./venv/bin
export FLASK_APP := run.py

# The suite builds its own schema in this database and truncates between tests,
# so never point it at a database holding anything you want to keep.
export TEST_DATABASE_URL ?= postgresql+psycopg2://grading_test:testpass@localhost:5432/grading_system_test

.PHONY: test test-fast migrate upgrade run train reset-dev

test:  ## run the whole test suite against the test database
	DATABASE_URL="$(TEST_DATABASE_URL)" JWT_SECRET=test-secret $(VENV)/pytest

test-fast:  ## skip the tests that load Sentence-BERT
	DATABASE_URL="$(TEST_DATABASE_URL)" JWT_SECRET=test-secret $(VENV)/pytest -k "not AutoGradeEndToEnd" -m "not slow"

upgrade:  ## apply migrations to the database in .env
	$(VENV)/flask db upgrade

migrate:  ## autogenerate a revision after changing a model (read it before applying)
	$(VENV)/flask db migrate -m "$(m)"

run:  ## start the dev server
	$(VENV)/python run.py

train:  ## retrain the RETIRED Ridge models (seconds) -- these no longer grade students
	@echo "NOTE: production text grading uses the fine-tuned model in"
	@echo "      ml_models/bert_rubric_scorer/, which this does NOT retrain."
	@echo "      That one needs a GPU and ~15-20 min on Colab:"
	@echo "      scripts/own_data_finetune_colab.ipynb. See its PROVENANCE.md."
	$(VENV)/python scripts/train_grader.py

reset-dev:  ## wipe the dev database and re-seed the demo rubric (destructive; dry-run without CONFIRM=1)
	$(VENV)/python scripts/reset_dev_data.py $(if $(CONFIRM),--yes,)
