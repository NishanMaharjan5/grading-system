import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ApiError } from "../api/client";
import { rubricsApi } from "../api/rubrics";
import { MAX_CONTENT_LENGTH, submissionsApi } from "../api/submissions";

export default function SubmitWork() {
  const { rubricId } = useParams();
  const navigate = useNavigate();
  const id = Number(rubricId);

  const [rubric, setRubric] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [content, setContent] = useState("");
  const [contentError, setContentError] = useState(null);
  const [formError, setFormError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const controller = new AbortController();

    // If this rubric has already been submitted to, there is nothing to do here
    // -- go to the existing submission rather than letting the form 409.
    Promise.all([
      rubricsApi.get(id, { signal: controller.signal }),
      submissionsApi.list({ signal: controller.signal }),
    ])
      .then(([loaded, submissions]) => {
        const existing = submissions.find((s) => s.rubric_id === id);
        if (existing) {
          navigate(`/student/submissions/${existing.id}`, { replace: true });
          return;
        }
        setRubric(loaded);
      })
      .catch((cause) => {
        if (cause?.name === "AbortError") return;
        setLoadError(cause instanceof ApiError ? cause.message : "Could not load this assignment.");
      });

    return () => controller.abort();
  }, [id, navigate]);

  async function handleSubmit(event) {
    event.preventDefault();
    setFormError(null);

    if (!content.trim()) {
      setContentError("Write your answer before submitting.");
      return;
    }
    if (content.length > MAX_CONTENT_LENGTH) {
      setContentError(`Your answer is ${content.length.toLocaleString()} characters; the limit is ${MAX_CONTENT_LENGTH.toLocaleString()}.`);
      return;
    }

    setContentError(null);
    setBusy(true);
    try {
      const created = await submissionsApi.create(id, content);
      navigate(`/student/submissions/${created.id}`, { replace: true, state: { justSubmitted: true } });
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 409) {
        // Submitted from somewhere else between loading this page and now.
        const existing = (await submissionsApi.list().catch(() => [])).find((s) => s.rubric_id === id);
        if (existing) {
          navigate(`/student/submissions/${existing.id}`, { replace: true });
          return;
        }
      }
      if (cause instanceof ApiError && cause.status === 422) setContentError(cause.detail);
      else setFormError(cause instanceof ApiError ? cause.message : "Could not submit your work.");
    } finally {
      setBusy(false);
    }
  }

  if (loadError) return <p className="page alert">{loadError}</p>;
  if (!rubric) return <p className="page muted">Loading…</p>;

  return (
    <div className="page">
      <p className="muted">
        <Link to="/student">← All assignments</Link>
      </p>
      <h1>{rubric.title}</h1>
      {rubric.description && <p>{rubric.description}</p>}

      <div className="card">
        <p className="muted">You will be marked on {rubric.total_points} points across:</p>
        <ul className="criteria-summary">
          {rubric.criteria.map((criterion) => (
            <li key={criterion.id}>
              <strong>{criterion.name}</strong> <span className="muted">/ {criterion.max_points}</span>
              {criterion.description && <div className="muted">{criterion.description}</div>}
            </li>
          ))}
        </ul>
      </div>

      <form onSubmit={handleSubmit} noValidate>
        {formError && (
          <p className="alert" role="alert">
            {formError}
          </p>
        )}

        <label htmlFor="content">Your answer</label>
        <textarea
          id="content"
          rows={14}
          value={content}
          onChange={(e) => setContent(e.target.value)}
          aria-invalid={Boolean(contentError)}
          aria-describedby={contentError ? "content-error" : undefined}
        />
        {contentError && (
          <p className="field-error" id="content-error">
            {contentError}
          </p>
        )}
        <p className="hint">
          {content.length.toLocaleString()} / {MAX_CONTENT_LENGTH.toLocaleString()} characters
        </p>
        <p className="hint">You can only submit once for this assignment.</p>

        <div className="row">
          <button type="submit" disabled={busy}>
            {busy ? "Submitting…" : "Submit work"}
          </button>
        </div>
      </form>
    </div>
  );
}
