import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ApiError } from "../api/client";
import { rubricsApi } from "../api/rubrics";
import { MAX_CONTENT_LENGTH, submissionsApi } from "../api/submissions";
import { checkLength, describeRange, hasWordLimit } from "../grading/wordCount";

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
    // Same rule the server enforces, so the answer arrives before the request.
    const { message } = checkLength(content, rubric);
    if (message) {
      setContentError(message);
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

  const isCode = rubric.type === "code";
  const limited = hasWordLimit(rubric);
  const length = checkLength(content, rubric);
  // Only complain about "too short" once they have started writing: an empty
  // box is not a mistake yet.
  const outOfRange = length.state === "long" || (length.state === "short" && content.trim() !== "");

  return (
    <div className="page">
      <p className="muted">
        <Link to="/student">← All assignments</Link>
      </p>
      <h1>{rubric.title}</h1>
      {rubric.description && <p>{rubric.description}</p>}

      <div className="card">
        {limited && (
          <p className="submission-range">
            <strong>Length:</strong> {describeRange(rubric.min_words, rubric.max_words)}.
          </p>
        )}
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

        {isCode && (
          <div className="card code-guide" id="code-guide">
            <h2 className="rubric__title">How your program is tested</h2>
            <ul>
              <li>
                It runs once per test with <strong>Python 3.9</strong>, using the{" "}
                <strong>standard library only</strong> (no numpy, requests or other installed packages).
              </li>
              <li>
                Each test&apos;s input arrives on <strong>standard input</strong>. Read it with{" "}
                <code>input()</code> or <code>sys.stdin</code>.
              </li>
              <li>
                You&apos;re marked on what it <strong>prints</strong>. Trailing spaces and blank lines at
                the start or end are ignored; everything else has to match exactly.
              </li>
              <li>
                It can&apos;t read files, use the network or start other programs, and a run is stopped
                after 5 seconds of processing time.
              </li>
            </ul>
          </div>
        )}

        <label htmlFor="content">{isCode ? "Your Python program" : "Your answer"}</label>
        <textarea
          id="content"
          rows={isCode ? 18 : 14}
          className={isCode ? "code-input" : undefined}
          spellCheck={isCode ? false : undefined}
          autoCapitalize={isCode ? "off" : undefined}
          autoCorrect={isCode ? "off" : undefined}
          value={content}
          onChange={(e) => setContent(e.target.value)}
          aria-invalid={Boolean(contentError) || outOfRange}
          aria-describedby={[isCode ? "code-guide" : null, contentError ? "content-error" : null].filter(Boolean).join(" ") || undefined}
        />
        {contentError && (
          <p className="field-error" id="content-error">
            {contentError}
          </p>
        )}
        <p className="hint">
          {limited ? (
            <>
              <span className={`word-count word-count--${length.state}`}>
                {length.words.toLocaleString()} {length.words === 1 ? "word" : "words"}
              </span>
              {" · "}
              {describeRange(rubric.min_words, rubric.max_words)}
              {" · "}
            </>
          ) : null}
          {content.length.toLocaleString()} / {MAX_CONTENT_LENGTH.toLocaleString()} characters
        </p>
        <p className="hint">You can only submit once for this assignment.</p>

        <div className="row">
          <button type="submit" disabled={busy || outOfRange}>
            {busy ? "Submitting…" : "Submit work"}
          </button>
          {outOfRange && <span className="muted">{length.message}</span>}
        </div>
      </form>
    </div>
  );
}
