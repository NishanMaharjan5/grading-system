import { useCallback, useEffect, useState } from "react";

import { ApiError } from "../api/client";
import { rubricsApi } from "../api/rubrics";
import RubricForm from "../components/RubricForm";
import RubricList from "../components/RubricList";

export default function TeacherDashboard() {
  const [rubrics, setRubrics] = useState(null); // null = not loaded yet
  const [loadError, setLoadError] = useState(null);
  const [actionError, setActionError] = useState(null);
  const [editing, setEditing] = useState(null); // null = closed, "new" = create, object = edit
  const [deletingId, setDeletingId] = useState(null);

  const load = useCallback(async (signal) => {
    try {
      setRubrics(await rubricsApi.list({ signal }));
      setLoadError(null);
    } catch (cause) {
      if (cause?.name === "AbortError") return;
      setLoadError(cause instanceof ApiError ? cause.message : "Could not load rubrics.");
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal);
    return () => controller.abort();
  }, [load]);

  function handleSaved() {
    setEditing(null);
    setActionError(null);
    load();
  }

  async function handleDelete(rubric) {
    if (!window.confirm(`Delete “${rubric.title}”? This cannot be undone.`)) return;

    setActionError(null);
    setDeletingId(rubric.id);
    try {
      await rubricsApi.remove(rubric.id);
      await load();
    } catch (cause) {
      // The button is disabled for locked rubrics, so a 409 here means the
      // list is stale -- someone submitted since it loaded. Say so and refresh.
      setActionError(
        cause instanceof ApiError && cause.status === 409
          ? "That rubric now has a submission, so it can no longer be deleted."
          : cause instanceof ApiError
            ? cause.message
            : "Could not delete the rubric.",
      );
      await load();
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <div className="page">
      <div className="page__head">
        <h1>Your rubrics</h1>
        {!editing && (
          <button type="button" onClick={() => setEditing("new")}>
            New rubric
          </button>
        )}
      </div>

      {actionError && (
        <p className="alert" role="alert">
          {actionError}
        </p>
      )}

      {editing && (
        <RubricForm
          initial={editing === "new" ? null : editing}
          onSaved={handleSaved}
          onCancel={() => setEditing(null)}
        />
      )}

      {loadError && <p className="alert">{loadError}</p>}

      {rubrics === null && !loadError ? (
        <p className="muted">Loading…</p>
      ) : (
        rubrics && <RubricList rubrics={rubrics} onEdit={setEditing} onDelete={handleDelete} busyId={deletingId} />
      )}
    </div>
  );
}
