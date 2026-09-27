import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ApiError } from "../api/client";
import { rubricsApi } from "../api/rubrics";
import { reviewErrorsFromDetail, submissionsApi } from "../api/submissions";
import { refreshShownErrors, sameErrors } from "../forms/errors";
import { criterionFeedbackForDisplay } from "../grading/feedbackText";

/** Client-side mirror of the backend's override rules, keyed by criterion id. */
function validateScores(criteria, scores) {
  const found = { criteria: {} };
  criteria.forEach((criterion) => {
    const raw = scores[criterion.id];
    const rowErrors = {};
    if (raw === undefined || String(raw).trim() === "") rowErrors.score = "Enter a score.";
    else {
      const value = Number(raw);
      if (Number.isNaN(value)) rowErrors.score = "Enter a number.";
      else if (value < 0) rowErrors.score = "Cannot be negative.";
      else if (value > criterion.max_points) rowErrors.score = `Cannot be more than ${criterion.max_points}.`;
    }
    if (Object.keys(rowErrors).length) found.criteria[criterion.id] = rowErrors;
  });
  if (!Object.keys(found.criteria).length) delete found.criteria;
  return found;
}

/**
 * Grading one submission.
 *
 * Manual entry is the primary path, not a fallback. Any rubric without a
 * trained model produces grading_failed, which is the normal outcome rather
 * than an error, so the score inputs are always present and always usable.
 *
 * Where the engine did produce a score it is shown as a suggestion to read and
 * act on, never written into the input for the teacher. Pre-filling would let
 * someone approve a machine's number by clicking save without having looked at
 * it, which is the one thing this review step exists to prevent.
 */
export default function ReviewSubmission() {
  const { submissionId } = useParams();
  const navigate = useNavigate();
  const id = Number(submissionId);

  const [state, setState] = useState({ status: "loading" });
  const [scores, setScores] = useState({});     // criterion_id -> string
  const [feedback, setFeedback] = useState({}); // criterion_id -> string
  const [summary, setSummary] = useState("");
  const [errors, setErrors] = useState({});
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    submissionsApi
      .get(id, { signal: controller.signal })
      .then(async (submission) => {
        const rubric = await rubricsApi.get(submission.rubric_id, { signal: controller.signal });
        setState({ status: "ok", submission, rubric });
      })
      .catch((cause) => {
        if (cause?.name === "AbortError") return;
        setState({
          status: "error",
          message: cause instanceof ApiError ? cause.message : "Could not load this submission.",
        });
      });
    return () => controller.abort();
  }, [id]);

  // Errors already on screen follow the score inputs as they're edited --
  // including when "Use this score" fills one in. Declared up here, before
  // the loading branches, because hooks can't sit behind an early return.
  useEffect(() => {
    if (state.status !== "ok") return;
    setErrors((shown) => {
      if (!Object.keys(shown).length) return shown;
      const next = refreshShownErrors(shown, validateScores(state.rubric.criteria, scores));
      return sameErrors(shown, next) ? shown : next;
    });
  }, [scores, state]);

  if (state.status === "loading") return <p className="page muted">Loading…</p>;
  if (state.status === "error") return <p className="page alert">{state.message}</p>;

  const { submission, rubric } = state;
  const criteria = rubric.criteria;
  const aiFor = (criterionId) => submission.grades.find((g) => g.criterion_id === criterionId) ?? null;
  const hasAiScore = (criterionId) => (aiFor(criterionId)?.ai_score ?? null) !== null;
  const everyCriterionHasAi = criteria.length > 0 && criteria.every((c) => hasAiScore(c.id));
  const alreadyApproved = submission.status === "approved";
  const isCode = rubric.type === "code";

  function onStale(cause) {
    if (cause instanceof ApiError && cause.status === 409) {
      setErrors({ form: "This submission has already been approved — the queue was out of date." });
      return true;
    }
    return false;
  }

  /** Empty body: only valid when the engine scored every criterion. */
  async function acceptAllAsIs() {
    setErrors({});
    setBusy(true);
    try {
      await submissionsApi.review(id, {});
      navigate("/teacher/review", { replace: true, state: { approvedId: id } });
    } catch (cause) {
      if (!onStale(cause)) {
        setErrors(
          cause instanceof ApiError && cause.status === 422
            ? { form: cause.detail }
            : { form: cause instanceof ApiError ? cause.message : "Could not approve." },
        );
      }
    } finally {
      setBusy(false);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();

    const found = validateScores(criteria, scores);
    if (Object.keys(found).length) {
      setErrors(found);
      return;
    }

    setErrors({});
    setBusy(true);
    // Order matters: the backend reports errors by index into this array.
    const order = criteria.map((c) => c.id);
    try {
      await submissionsApi.review(id, {
        criterion_scores: criteria.map((criterion) => {
          const row = { criterion_id: criterion.id, final_score: Number(scores[criterion.id]) };
          const text = (feedback[criterion.id] ?? "").trim();
          if (text) row.final_feedback = text;
          return row;
        }),
        ...(summary.trim() ? { summary: summary.trim() } : {}),
      });
      navigate("/teacher/review", { replace: true, state: { approvedId: id } });
    } catch (cause) {
      if (!onStale(cause)) {
        setErrors(
          cause instanceof ApiError && cause.status === 422
            ? reviewErrorsFromDetail(cause.detail, order)
            : { form: cause instanceof ApiError ? cause.message : "Could not save this grade." },
        );
      }
    } finally {
      setBusy(false);
    }
  }

  const enteredTotal = criteria.reduce((sum, c) => sum + (Number(scores[c.id]) || 0), 0);

  return (
    <div className="page">
      <p className="muted">
        <Link to="/teacher/review">← Review queue</Link>
      </p>
      <h1>{rubric.title}</h1>
      <p className="student-line">
        Submitted by <strong>{submission.student_name}</strong>{" "}
        <span className="muted">
          {submission.student_email} · {new Date(submission.created_at).toLocaleString()}
        </span>
      </p>

      {alreadyApproved && (
        <p className="alert" role="alert">
          This submission has already been approved and cannot be changed.
        </p>
      )}

      <details className="card" open>
        <summary>The student's work</summary>
        {isCode ? (
          <pre className="code">{submission.content}</pre>
        ) : (
          <p className="submitted-content">{submission.content}</p>
        )}
      </details>

      {errors.form && (
        <p className="alert" role="alert">
          {errors.form}
        </p>
      )}

      {everyCriterionHasAi && !alreadyApproved && (
        <div className="card accept-all">
          <p className="muted">
            The engine scored every criterion. You can accept all of its scores unchanged, or grade below.
          </p>
          <button type="button" className="button--secondary" onClick={acceptAllAsIs} disabled={busy}>
            Accept all AI scores as-is
          </button>
        </div>
      )}

      <form onSubmit={handleSubmit} noValidate>
        <h2>Scores and feedback</h2>
        {criteria.map((criterion) => {
          const ai = aiFor(criterion.id);
          const showsAi = hasAiScore(criterion.id);
          const rowError = errors.criteria?.[criterion.id];

          return (
            <div className="card criterion-review" key={criterion.id}>
              <div className="rubric__head">
                <h3 className="rubric__title">{criterion.name}</h3>
                <span className="muted">out of {criterion.max_points}</span>
              </div>
              {criterion.description && <p className="muted">{criterion.description}</p>}

              {/* Only rendered where the engine actually produced a score --
                  never for a grading_failed submission. */}
              {showsAi && (
                <div className="ai-suggestion">
                  <p className="ai-suggestion__head">
                    AI suggestion: <strong>{ai.ai_score}</strong> / {criterion.max_points}
                  </p>
                  {ai.ai_feedback && (
                    <p className={`feedback muted${isCode ? " feedback--code" : ""}`}>
                      {criterionFeedbackForDisplay(ai.ai_feedback, {
                        criterionName: criterion.name,
                        score: ai.ai_score,
                        maxPoints: criterion.max_points,
                        description: criterion.description,
                        hideDescription: true,  // already shown just above this block
                      })}
                    </p>
                  )}
                  <button
                    type="button"
                    className="button--plain"
                    onClick={() => setScores({ ...scores, [criterion.id]: String(ai.ai_score) })}
                    disabled={alreadyApproved}
                  >
                    Use this score
                  </button>
                </div>
              )}

              <label htmlFor={`score-${criterion.id}`}>Score</label>
              <input
                id={`score-${criterion.id}`}
                type="number"
                min="0"
                max={criterion.max_points}
                step="any"
                value={scores[criterion.id] ?? ""}
                onChange={(e) => setScores({ ...scores, [criterion.id]: e.target.value })}
                aria-invalid={Boolean(rowError?.score)}
                disabled={alreadyApproved}
              />
              {rowError?.score && <p className="field-error">{rowError.score}</p>}

              <label htmlFor={`feedback-${criterion.id}`}>Feedback (optional)</label>
              <textarea
                id={`feedback-${criterion.id}`}
                rows={2}
                value={feedback[criterion.id] ?? ""}
                onChange={(e) => setFeedback({ ...feedback, [criterion.id]: e.target.value })}
                disabled={alreadyApproved}
              />
              {rowError?.feedback && <p className="field-error">{rowError.feedback}</p>}
              <p className="hint">Left blank, feedback is written from the score you give.</p>
            </div>
          );
        })}

        <label htmlFor="summary">Overall feedback (optional)</label>
        <textarea
          id="summary"
          rows={3}
          value={summary}
          onChange={(e) => setSummary(e.target.value)}
          disabled={alreadyApproved}
        />
        <p className="hint">Left blank, an overall summary is written from the scores.</p>

        <p className="muted">
          Total entered: {enteredTotal} / {rubric.total_points}
        </p>

        <div className="row">
          <button type="submit" disabled={busy || alreadyApproved}>
            {busy ? "Saving…" : "Approve and release grade"}
          </button>
          <Link to="/teacher/review" className="button button--secondary">Cancel</Link>
        </div>
      </form>
    </div>
  );
}
