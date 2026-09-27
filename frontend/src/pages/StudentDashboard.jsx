import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { ApiError } from "../api/client";
import { rubricsApi } from "../api/rubrics";
import { statusCopy, submissionsApi } from "../api/submissions";

/**
 * Assignments a student can submit against.
 *
 * The rubric list and the student's own submissions are separate endpoints, so
 * they are matched up here by rubric_id. A rubric the student has already
 * submitted to shows the way to that submission rather than a form that is
 * guaranteed to be refused -- one submission per rubric is enforced by a
 * unique constraint.
 */
export default function StudentDashboard() {
  const [state, setState] = useState({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      rubricsApi.list({ signal: controller.signal }),
      submissionsApi.list({ signal: controller.signal }),
    ])
      .then(([rubrics, submissions]) => {
        const byRubric = new Map(submissions.map((s) => [s.rubric_id, s]));
        setState({ status: "ok", rubrics, byRubric });
      })
      .catch((cause) => {
        if (cause?.name === "AbortError") return;
        setState({
          status: "error",
          message: cause instanceof ApiError ? cause.message : "Could not load your assignments.",
        });
      });
    return () => controller.abort();
  }, []);

  if (state.status === "loading") return <p className="page muted">Loading…</p>;
  if (state.status === "error") return <p className="page alert">{state.message}</p>;

  const { rubrics, byRubric } = state;

  return (
    <div className="page">
      <h1>Your assignments</h1>

      {rubrics.length === 0 ? (
        <p className="muted">No assignments have been set yet.</p>
      ) : (
        <ul className="rubric-list">
          {rubrics.map((rubric) => {
            const submission = byRubric.get(rubric.id);
            const copy = submission ? statusCopy(submission.status) : null;

            return (
              <li key={rubric.id} className="card rubric">
                <div className="rubric__head">
                  <h3 className="rubric__title">{rubric.title}</h3>
                  {submission ? (
                    <span className={`badge badge--${copy.tone}`}>{copy.label}</span>
                  ) : (
                    <span className="badge">Not submitted</span>
                  )}
                </div>

                {rubric.description && <p className="muted">{rubric.description}</p>}

                <p className="muted">
                  {rubric.criteria.length} {rubric.criteria.length === 1 ? "criterion" : "criteria"} ·{" "}
                  {rubric.total_points} points
                </p>

                <ul className="criteria-summary">
                  {rubric.criteria.map((criterion) => (
                    <li key={criterion.id}>
                      {criterion.name} <span className="muted">/ {criterion.max_points}</span>
                    </li>
                  ))}
                </ul>

                <div className="row">
                  {submission ? (
                    <Link to={`/student/submissions/${submission.id}`} className="button button--secondary">View your submission</Link>
                  ) : (
                    <Link to={`/student/submit/${rubric.id}`} className="button">Submit work</Link>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
