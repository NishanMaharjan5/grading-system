import { useEffect, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";

import { ApiError } from "../api/client";
import { rubricsApi } from "../api/rubrics";
import { statusCopy, submissionsApi } from "../api/submissions";

/**
 * A student's view of one submission.
 *
 * Scores only exist here once the teacher has approved: the API sends students
 * final_score/final_summary and nothing else, so there is no AI suggestion in
 * hand to render even by accident. Before approval this page deliberately
 * shows no number at all rather than a placeholder that reads like a grade.
 *
 * Criterion names come from the rubric -- the submission payload carries only
 * criterion_id.
 */
export default function SubmissionView() {
  const { submissionId } = useParams();
  const location = useLocation();
  const justSubmitted = Boolean(location.state?.justSubmitted);

  const [state, setState] = useState({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    submissionsApi
      .get(Number(submissionId), { signal: controller.signal })
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
  }, [submissionId]);

  if (state.status === "loading") return <p className="page muted">Loading…</p>;
  if (state.status === "error") return <p className="page alert">{state.message}</p>;

  const { submission, rubric } = state;
  const copy = statusCopy(submission.status);
  const criterionName = (id) => rubric.criteria.find((c) => c.id === id)?.name ?? `Criterion ${id}`;
  const criterionMax = (id) => rubric.criteria.find((c) => c.id === id)?.max_points;
  const isApproved = submission.status === "approved";

  return (
    <div className="page">
      <p className="muted">
        <Link to="/student">← All assignments</Link>
      </p>
      <h1>{rubric.title}</h1>

      {justSubmitted && (
        <p className="notice" role="status">
          Your work was submitted successfully.
        </p>
      )}

      <div className="card">
        <div className="rubric__head">
          <h2 className="rubric__title">Status</h2>
          <span className={`badge badge--${copy.tone}`}>{copy.label}</span>
        </div>
        <p>{copy.detail}</p>
        <p className="muted">Submitted {new Date(submission.created_at).toLocaleString()}</p>
      </div>

      {isApproved && (
        <div className="card">
          <h2 className="rubric__title">
            Your grade: {submission.final_total} / {rubric.total_points}
          </h2>
          {submission.final_summary && <p>{submission.final_summary}</p>}

          <ul className="criteria-summary">
            {submission.grades.map((grade) => (
              <li key={grade.criterion_id}>
                <strong>
                  {criterionName(grade.criterion_id)}: {grade.final_score}
                  {criterionMax(grade.criterion_id) !== undefined && ` / ${criterionMax(grade.criterion_id)}`}
                </strong>
                {grade.final_feedback && <div className="muted">{grade.final_feedback}</div>}
              </li>
            ))}
          </ul>
        </div>
      )}

      <details className="card">
        <summary>What you submitted</summary>
        <p className="submitted-content">{submission.content}</p>
      </details>
    </div>
  );
}
