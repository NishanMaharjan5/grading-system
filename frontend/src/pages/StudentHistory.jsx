import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { ApiError } from "../api/client";
import { rubricsApi } from "../api/rubrics";
import { statusCopy, submissionsApi } from "../api/submissions";

/**
 * Everything this student has ever submitted, in one list.
 *
 * The dashboard answers "what do I still have to do"; this answers "what have
 * I handed in, and what did I get". Those are different questions, and the
 * dashboard's rubric-by-rubric layout cannot show the second one — a student
 * had to open each assignment in turn to find their grades.
 *
 * The API already scopes /api/submissions to the signed-in student, so there
 * is nothing to filter here; the rubrics are fetched only to put a title and a
 * total next to each row.
 */
export default function StudentHistory() {
  const [state, setState] = useState({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      submissionsApi.list({ signal: controller.signal }),
      rubricsApi.list({ signal: controller.signal }),
    ])
      .then(([submissions, rubrics]) => {
        const byId = new Map(rubrics.map((r) => [r.id, r]));
        // Newest first: the thing you just handed in is the thing you are
        // most likely looking for.
        const ordered = [...submissions].sort(
          (a, b) => new Date(b.created_at) - new Date(a.created_at),
        );
        setState({ status: "ok", submissions: ordered, byId });
      })
      .catch((cause) => {
        if (cause?.name === "AbortError") return;
        setState({
          status: "error",
          message: cause instanceof ApiError ? cause.message : "Could not load your submissions.",
        });
      });
    return () => controller.abort();
  }, []);

  if (state.status === "loading") return <p className="page muted">Loading…</p>;
  if (state.status === "error") return <p className="page alert">{state.message}</p>;

  const { submissions, byId } = state;
  const graded = submissions.filter((s) => s.final_total !== null && s.final_total !== undefined);

  return (
    <div className="page">
      <h1>Your submissions</h1>

      {submissions.length === 0 ? (
        <p className="muted">
          You haven't submitted anything yet. <Link to="/student">See your assignments</Link>.
        </p>
      ) : (
        <>
          <p className="muted">
            {submissions.length} submission{submissions.length === 1 ? "" : "s"}
            {graded.length > 0 && ` · ${graded.length} graded`}
          </p>

          <ul className="rubric-list">
            {submissions.map((submission) => {
              const rubric = byId.get(submission.rubric_id);
              const copy = statusCopy(submission.status);
              const total = submission.final_total;
              const outOf = rubric?.total_points;

              return (
                <li key={submission.id} className="card rubric">
                  <div className="rubric__head">
                    <h3 className="rubric__title">
                      {rubric?.title ?? `Assignment ${submission.rubric_id}`}
                    </h3>
                    <span className={`badge badge--${copy.tone}`}>{copy.label}</span>
                  </div>

                  <p className="muted">
                    Submitted {new Date(submission.created_at).toLocaleString()}
                  </p>

                  {total !== null && total !== undefined ? (
                    <p className="history__grade">
                      <strong>
                        {total}
                        {outOf !== undefined && ` / ${outOf}`}
                      </strong>
                    </p>
                  ) : (
                    <p className="muted">{copy.detail}</p>
                  )}

                  <div className="row">
                    <Link to={`/student/submissions/${submission.id}`} className="button button--secondary">
                      {total !== null && total !== undefined ? "View grade and feedback" : "View submission"}
                    </Link>
                  </div>
                </li>
              );
            })}
          </ul>
        </>
      )}
    </div>
  );
}
