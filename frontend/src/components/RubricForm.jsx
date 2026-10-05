import { useEffect, useState } from "react";

import { ApiError } from "../api/client";
import { fieldErrorsFromDetail, rubricsApi } from "../api/rubrics";
import { refreshShownErrors, sameErrors } from "../forms/errors";
import { fromDateTimeLocal, toDateTimeLocal } from "../grading/deadlines";

// The embedder reads ~226 words and silently drops the rest, so a cap at or
// under this keeps the grader reading the whole essay. Mirrors
// GRADER_WORD_LIMIT in backend/app/word_limits.py.
const GRADER_WORD_LIMIT = 226;
const DEFAULT_MAX_WORDS = 200;

const blankCriterion = () => ({ name: "", max_points: "", test_cases: [] });
const blankTestCase = () => ({ stdin: "", expected_output: "" });

/**
 * Create/edit form for a rubric and its criteria.
 *
 * Validation runs twice on purpose. The checks here mirror the backend's rules
 * so an obvious mistake is marked next to the offending input without a round
 * trip; the backend stays the authority, and whatever it rejects is mapped
 * back onto the same fields. The two must agree -- if a rule changes server
 * side, `validate` below has to change with it.
 */
export default function RubricForm({ initial, onSaved, onCancel }) {
  const editing = Boolean(initial);

  const [title, setTitle] = useState(initial?.title ?? "");
  const [description, setDescription] = useState(initial?.description ?? "");
  const [type, setType] = useState(initial?.type ?? "text");
  // A new text rubric is capped by default; an existing one keeps whatever it
  // has, including no limit at all.
  const [dueDate, setDueDate] = useState(toDateTimeLocal(initial?.due_date));
  const [minWords, setMinWords] = useState(
    initial?.min_words != null ? String(initial.min_words) : "",
  );
  const [maxWords, setMaxWords] = useState(
    initial ? (initial.max_words != null ? String(initial.max_words) : "") : String(DEFAULT_MAX_WORDS),
  );
  const [criteria, setCriteria] = useState(
    initial?.criteria?.map((c) => ({
      name: c.name,
      max_points: String(c.max_points),
      test_cases: (c.test_cases ?? []).map((t) => ({ stdin: t.stdin, expected_output: t.expected_output })),
    })) ?? [blankCriterion()],
  );
  const [errors, setErrors] = useState({});
  const [busy, setBusy] = useState(false);

  const criterionError = (index, field) => errors.criteria?.[index]?.[field];
  const overGraderLimit = type === "text" && Number(maxWords) > GRADER_WORD_LIMIT;

  function updateCriterion(index, field, value) {
    setCriteria(criteria.map((row, i) => (i === index ? { ...row, [field]: value } : row)));
  }

  function updateTestCase(criterionIndex, caseIndex, field, value) {
    setCriteria(criteria.map((row, i) => (i !== criterionIndex ? row : {
      ...row,
      test_cases: row.test_cases.map((c, j) => (j === caseIndex ? { ...c, [field]: value } : c)),
    })));
  }

  function setTestCases(criterionIndex, next) {
    setCriteria(criteria.map((row, i) => (i === criterionIndex ? { ...row, test_cases: next } : row)));
  }

  function validate() {
    const found = { criteria: {} };

    if (!title.trim()) found.title = "Give the rubric a title.";
    else if (title.trim().length > 200) found.title = "Title must be 200 characters or fewer.";

    if (criteria.length === 0) found.criteriaForm = "Add at least one criterion.";

    if (type === "text") {
      const min = minWords.trim() === "" ? null : Number(minWords);
      const max = maxWords.trim() === "" ? null : Number(maxWords);
      if (min !== null && (!Number.isInteger(min) || min < 0)) {
        found.min_words = "Enter a whole number of words, or leave blank for no limit.";
      }
      if (max !== null && (!Number.isInteger(max) || max < 1)) {
        found.max_words = "Enter a whole number of words, or leave blank for no limit.";
      }
      if (!found.min_words && !found.max_words && min !== null && max !== null && min > max) {
        found.max_words = `Max words must be at least the minimum (${min}).`;
      }
    }

    const seen = new Map();
    criteria.forEach((row, index) => {
      const rowErrors = {};
      const name = row.name.trim();

      if (!name) rowErrors.name = "Name is required.";
      else if (name.length > 100) rowErrors.name = "Name must be 100 characters or fewer.";
      else if (seen.has(name.toLowerCase())) {
        rowErrors.name = `Duplicate of criterion ${seen.get(name.toLowerCase()) + 1}.`;
      } else seen.set(name.toLowerCase(), index);

      const points = Number(row.max_points);
      if (row.max_points === "" || Number.isNaN(points)) rowErrors.max_points = "Enter a number.";
      else if (points <= 0) rowErrors.max_points = "Must be greater than 0.";

      // A code criterion is graded purely by its tests, so it needs at least one.
      if (type === "code" && row.test_cases.length === 0) {
        rowErrors.test_cases = "Add at least one test case.";
      }

      if (Object.keys(rowErrors).length) found.criteria[index] = rowErrors;
    });

    if (!Object.keys(found.criteria).length) delete found.criteria;
    return found;
  }

  // Errors already on screen follow the fields as they're edited, instead of
  // waiting for the next submit. See forms/errors.js for exactly which ones.
  useEffect(() => {
    setErrors((shown) => {
      if (!Object.keys(shown).length) return shown;
      const next = refreshShownErrors(shown, validate());
      return sameErrors(shown, next) ? shown : next;
    });
    // validate() reads exactly these values
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [title, criteria, type, minWords, maxWords]);

  async function handleSubmit(event) {
    event.preventDefault();

    const found = validate();
    if (Object.keys(found).length) {
      setErrors(found);
      return;
    }

    setErrors({});
    setBusy(true);
    try {
      const payload = {
        title: title.trim(),
        description: description.trim(),
        type,
        // null clears a limit; the server refuses limits on a code rubric
        due_date: fromDateTimeLocal(dueDate),
        min_words: type === "text" && minWords.trim() !== "" ? Number(minWords) : null,
        max_words: type === "text" && maxWords.trim() !== "" ? Number(maxWords) : null,
        criteria: criteria.map((row) => ({
          name: row.name.trim(),
          max_points: Number(row.max_points),
          ...(type === "code" ? { test_cases: row.test_cases } : {}),
        })),
      };
      const saved = editing
        ? await rubricsApi.update(initial.id, payload)
        : await rubricsApi.create(payload);
      onSaved(saved);
    } catch (cause) {
      setErrors(
        cause instanceof ApiError && cause.status === 422
          ? fieldErrorsFromDetail(cause.detail)
          : { form: cause instanceof ApiError ? cause.message : "Could not save the rubric." },
      );
    } finally {
      setBusy(false);
    }
  }

  const totalPoints = criteria.reduce((sum, row) => sum + (Number(row.max_points) || 0), 0);

  return (
    <form onSubmit={handleSubmit} noValidate className="card">
      <h2>{editing ? `Edit “${initial.title}”` : "New rubric"}</h2>

      {errors.form && (
        <p className="alert" role="alert">
          {errors.form}
        </p>
      )}

      <label htmlFor="rubric-title">Title</label>
      <input
        id="rubric-title"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        aria-invalid={Boolean(errors.title)}
        aria-describedby={errors.title ? "rubric-title-error" : undefined}
      />
      {errors.title && (
        <p className="field-error" id="rubric-title-error">
          {errors.title}
        </p>
      )}

      <label htmlFor="rubric-type">Type</label>
      <select
        id="rubric-type"
        value={type}
        onChange={(e) => setType(e.target.value)}
        disabled={editing}
      >
        <option value="text">Written answer — graded by the text model</option>
        <option value="code">Python code — graded by running test cases</option>
      </select>
      {editing && <p className="hint">A rubric's type cannot be changed after it is created.</p>}

      <label htmlFor="rubric-description">Description (optional)</label>
      <textarea
        id="rubric-description"
        rows={2}
        value={description}
        onChange={(e) => setDescription(e.target.value)}
      />

      <label htmlFor="rubric-due-date">Due date (optional)</label>
      <input
        id="rubric-due-date"
        type="datetime-local"
        value={dueDate}
        onChange={(e) => setDueDate(e.target.value)}
      />
      <p className="hint">
        {dueDate
          ? "After this moment the assignment stops accepting submissions, including resubmissions."
          : "Leave blank for no deadline — students can submit at any time."}
      </p>

      {type === "text" && (
        <fieldset className="word-limits">
          <legend>Length (optional)</legend>
          <p className="hint">
            Leave either blank for no limit. Students see the range before they submit, and a
            submission outside it is refused.
          </p>
          <div className="criterion-row">
            <div className="criterion-row__field">
              <label htmlFor="rubric-min-words">Min words</label>
              <input
                id="rubric-min-words"
                type="number"
                min="0"
                step="1"
                value={minWords}
                onChange={(e) => setMinWords(e.target.value)}
                aria-invalid={Boolean(errors.min_words)}
                aria-describedby={errors.min_words ? "rubric-min-words-error" : undefined}
              />
              {errors.min_words && (
                <p className="field-error" id="rubric-min-words-error">{errors.min_words}</p>
              )}
            </div>
            <div className="criterion-row__field">
              <label htmlFor="rubric-max-words">Max words</label>
              <input
                id="rubric-max-words"
                type="number"
                min="1"
                step="1"
                value={maxWords}
                onChange={(e) => setMaxWords(e.target.value)}
                aria-invalid={Boolean(errors.max_words)}
                aria-describedby={
                  [errors.max_words ? "rubric-max-words-error" : null,
                   overGraderLimit ? "rubric-max-words-warning" : null].filter(Boolean).join(" ") || undefined
                }
              />
              {errors.max_words && (
                <p className="field-error" id="rubric-max-words-error">{errors.max_words}</p>
              )}
            </div>
          </div>

          {overGraderLimit && (
            <p className="warning" id="rubric-max-words-warning" role="status">
              <strong>Past {GRADER_WORD_LIMIT} words, the AI grader stops reading.</strong> It takes in
              about {GRADER_WORD_LIMIT} words and ignores the rest, so anything after that point is
              not part of the suggested score. You can still set a higher limit — you will just be
              grading the end of these essays yourself.
            </p>
          )}
        </fieldset>
      )}

      <fieldset className="criteria">
        <legend>Criteria</legend>
        {errors.criteriaForm && (
          <p className="alert" role="alert">
            {errors.criteriaForm}
          </p>
        )}

        {criteria.map((row, index) => (
          <div className="criterion-block" key={index}>
            <div className="criterion-row">
            <div className="criterion-row__field">
              <label htmlFor={`criterion-name-${index}`}>Name</label>
              <input
                id={`criterion-name-${index}`}
                value={row.name}
                onChange={(e) => updateCriterion(index, "name", e.target.value)}
                aria-invalid={Boolean(criterionError(index, "name"))}
              />
              {criterionError(index, "name") && (
                <p className="field-error">{criterionError(index, "name")}</p>
              )}
            </div>

            <div className="criterion-row__field criterion-row__field--points">
              <label htmlFor={`criterion-points-${index}`}>Max points</label>
              <input
                id={`criterion-points-${index}`}
                type="number"
                min="0"
                step="any"
                value={row.max_points}
                onChange={(e) => updateCriterion(index, "max_points", e.target.value)}
                aria-invalid={Boolean(criterionError(index, "max_points"))}
              />
              {criterionError(index, "max_points") && (
                <p className="field-error">{criterionError(index, "max_points")}</p>
              )}
            </div>

            <button
              type="button"
              className="button--plain"
              onClick={() => setCriteria(criteria.filter((_, i) => i !== index))}
              disabled={criteria.length === 1}
              title={criteria.length === 1 ? "A rubric needs at least one criterion" : "Remove"}
            >
              Remove
            </button>
            </div>

            {type === "code" && (
              <div className="test-cases">
                <p className="hint">
                  Each test runs the program with the given input and compares what it prints.
                  Trailing spaces and blank lines at the ends are ignored; everything else must match.
                </p>
                {criterionError(index, "test_cases") && (
                  <p className="field-error">{criterionError(index, "test_cases")}</p>
                )}

                {row.test_cases.map((testCase, caseIndex) => (
                  <div className="test-case" key={caseIndex}>
                    <div className="criterion-row__field">
                      <label htmlFor={`stdin-${index}-${caseIndex}`}>Input (stdin)</label>
                      <textarea
                        id={`stdin-${index}-${caseIndex}`}
                        rows={2}
                        value={testCase.stdin}
                        onChange={(e) => updateTestCase(index, caseIndex, "stdin", e.target.value)}
                      />
                    </div>
                    <div className="criterion-row__field">
                      <label htmlFor={`expected-${index}-${caseIndex}`}>Expected output</label>
                      <textarea
                        id={`expected-${index}-${caseIndex}`}
                        rows={2}
                        value={testCase.expected_output}
                        onChange={(e) => updateTestCase(index, caseIndex, "expected_output", e.target.value)}
                      />
                    </div>
                    <button
                      type="button"
                      className="button--plain"
                      onClick={() => setTestCases(index, row.test_cases.filter((_, j) => j !== caseIndex))}
                    >
                      Remove
                    </button>
                  </div>
                ))}

                <button
                  type="button"
                  className="button--plain"
                  onClick={() => setTestCases(index, [...row.test_cases, blankTestCase()])}
                >
                  + Add test case
                </button>
              </div>
            )}
          </div>
        ))}

        <button type="button" className="button--plain" onClick={() => setCriteria([...criteria, blankCriterion()])}>
          + Add criterion
        </button>
      </fieldset>

      <p className="muted">Total: {totalPoints} points across {criteria.length} criteria</p>

      <div className="row">
        <button type="submit" disabled={busy}>
          {busy ? "Saving…" : editing ? "Save changes" : "Create rubric"}
        </button>
        {onCancel && (
          <button type="button" className="button--secondary" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>
    </form>
  );
}
