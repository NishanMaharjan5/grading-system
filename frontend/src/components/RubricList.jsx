/**
 * The teacher's rubrics.
 *
 * `locked` comes from the server rather than being inferred here. Once a
 * student has submitted, the backend refuses edits and deletes with a 409, so
 * those buttons are disabled with the reason stated -- offering a button that
 * is guaranteed to fail is worse than not offering it.
 */
export default function RubricList({ rubrics, onEdit, onDelete, onExport, exportingId, busyId }) {
  if (rubrics.length === 0) {
    return <p className="muted">No rubrics yet. Create one to get started.</p>;
  }

  return (
    <ul className="rubric-list">
      {rubrics.map((rubric) => (
        <li key={rubric.id} className="card rubric">
          <div className="rubric__head">
            <h3 className="rubric__title">{rubric.title}</h3>
            {rubric.locked && (
              <span className="badge badge--locked" title="Work has been submitted against this rubric">
                Locked
              </span>
            )}
          </div>

          {rubric.description && <p className="muted">{rubric.description}</p>}

          <p className="muted">
            {rubric.criteria.length} {rubric.criteria.length === 1 ? "criterion" : "criteria"} ·{" "}
            {rubric.total_points} points ·{" "}
            {rubric.submission_count} {rubric.submission_count === 1 ? "submission" : "submissions"}
          </p>

          <ul className="criteria-summary">
            {rubric.criteria.map((criterion) => (
              <li key={criterion.id}>
                {criterion.name} <span className="muted">/ {criterion.max_points}</span>
              </li>
            ))}
          </ul>

          <div className="row">
            <button
              type="button"
              className="button--secondary"
              onClick={() => onEdit(rubric)}
              disabled={rubric.locked}
              title={rubric.locked ? "Cannot edit: work has already been submitted" : "Edit this rubric"}
            >
              Edit
            </button>
            <button
              type="button"
              className="button--secondary"
              onClick={() => onExport(rubric)}
              disabled={exportingId === rubric.id}
              title={rubric.submission_count
                ? "Download every submission's grades as a CSV"
                : "No submissions yet — the file will have just its header row"}
            >
              {exportingId === rubric.id ? "Preparing…" : "Export grades"}
            </button>
            <button
              type="button"
              className="button--secondary button--danger"
              onClick={() => onDelete(rubric)}
              disabled={rubric.locked || busyId === rubric.id}
              title={rubric.locked ? "Cannot delete: work has already been submitted" : "Delete this rubric"}
            >
              {busyId === rubric.id ? "Deleting…" : "Delete"}
            </button>
            {rubric.locked && (
              <span className="muted">Editing is closed once work has been submitted.</span>
            )}
          </div>
        </li>
      ))}
    </ul>
  );
}
