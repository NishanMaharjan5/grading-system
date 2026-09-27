import { useCallback, useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";

import { ApiError } from "../api/client";
import { rubricsApi } from "../api/rubrics";
import { submissionsApi } from "../api/submissions";

/**
 * Work waiting on this teacher.
 *
 * Both statuses belong here and neither is an error state: ai_graded has a
 * suggestion to check, grading_failed simply has no suggestion because the
 * rubric has no trained model. The queue says which is which so a teacher
 * knows whether there is anything to review or whether they are grading from
 * scratch.
 *
 * Criterion names live on the rubric, not the submission, so the two are
 * matched up here.
 */
const STATUS_COPY = {
  ai_graded: { label: "AI suggestion ready", tone: "pending", hint: "Review the suggested scores." },
  grading_failed: { label: "Needs manual grading", tone: "neutral", hint: "No trained model for this rubric." },
};

/** First few lines of a program, with its line breaks kept -- collapsing code
    onto one line (as a text preview does) makes it unreadable. */
const PREVIEW_LINES = 6;
function codePreview(source) {
  const lines = source.replace(/\s+$/, "").split("\n");
  return lines.length > PREVIEW_LINES ? [...lines.slice(0, PREVIEW_LINES), "…"].join("\n") : lines.join("\n");
}

export default function ReviewQueue() {
  const location = useLocation();
  const justApproved = location.state?.approvedId;

  const [state, setState] = useState({ status: "loading" });

  const load = useCallback(async (signal) => {
    try {
      const [pending, rubrics] = await Promise.all([
        submissionsApi.pending({ signal }),
        rubricsApi.list({ signal }),
      ]);
      const byId = new Map(rubrics.map((r) => [r.id, r]));
      setState({ status: "ok", pending, byId });
    } catch (cause) {
      if (cause?.name === "AbortError") return;
      setState({
        status: "error",
        message: cause instanceof ApiError ? cause.message : "Could not load the review queue.",
      });
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal);
    return () => controller.abort();
  }, [load]);

  if (state.status === "loading") return <p className="page muted">Loading…</p>;
  if (state.status === "error") return <p className="page alert">{state.message}</p>;

  const { pending, byId } = state;

  return (
    <div className="page">
      <h1>Review queue</h1>

      {justApproved && (
        <p className="notice" role="status">
          Grade approved and released to the student.
        </p>
      )}

      {pending.length === 0 ? (
        <p className="muted">Nothing waiting. Submissions appear here as students send work in.</p>
      ) : (
        <ul className="rubric-list">
          {pending.map((submission) => {
            const rubric = byId.get(submission.rubric_id);
            const copy = STATUS_COPY[submission.status] ?? { label: submission.status, tone: "neutral", hint: "" };

            return (
              <li key={submission.id} className="card rubric">
                <div className="rubric__head">
                  <h3 className="rubric__title">{rubric?.title ?? `Rubric ${submission.rubric_id}`}</h3>
                  <span className={`badge badge--${copy.tone}`}>{copy.label}</span>
                </div>

                <p className="student-line">
                  <strong>{submission.student_name}</strong>{" "}
                  <span className="muted">{submission.student_email}</span>
                </p>

                <p className="muted">
                  Submitted {new Date(submission.created_at).toLocaleString()} · {copy.hint}
                </p>

                {rubric?.type === "code" ? (
                  <pre className="code code--preview">{codePreview(submission.content)}</pre>
                ) : (
                  <p className="submitted-preview">
                    {submission.content.slice(0, 180)}
                    {submission.content.length > 180 ? "…" : ""}
                  </p>
                )}

                <div className="row">
                  <Link to={`/teacher/review/${submission.id}`} className="button">
                    {submission.status === "ai_graded" ? "Review suggestion" : "Grade this"}
                  </Link>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
