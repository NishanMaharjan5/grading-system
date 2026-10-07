import { useCallback, useEffect, useState } from "react";

import { ApiError } from "../api/client";
import { rubricsApi } from "../api/rubrics";
import { submissionsApi } from "../api/submissions";
import DashboardInsights from "../components/DashboardInsights";
import DashboardStats from "../components/DashboardStats";
import RubricForm from "../components/RubricForm";
import RubricList from "../components/RubricList";
import { computeDashboardStats } from "../grading/dashboardStats";
import { computeStruggleInsights } from "../grading/dashboardInsights";

export default function TeacherDashboard() {
  const [rubrics, setRubrics] = useState(null); // null = not loaded yet
  const [stats, setStats] = useState(null);
  const [insights, setInsights] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [actionError, setActionError] = useState(null);
  const [editing, setEditing] = useState(null); // null = closed, "new" = create, object = edit
  const [deletingId, setDeletingId] = useState(null);
  const [exportingId, setExportingId] = useState(null);

  const load = useCallback(async (signal) => {
    try {
      // Three calls, not one: the stats row needs the full submission list
      // (for the 7-day count and the AI-acceptance rate) and the review
      // queue (for the pending count), neither of which the rubric list
      // alone carries. All three already scope to this teacher server-side.
      const [rubricList, submissions, pending] = await Promise.all([
        rubricsApi.list({ signal }),
        submissionsApi.list({ signal }),
        submissionsApi.pending({ signal }),
      ]);
      setRubrics(rubricList);
      setStats(computeDashboardStats({ rubrics: rubricList, submissions, pending }));
      setInsights(computeStruggleInsights({ rubrics: rubricList, submissions }));
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

  async function handleExport(rubric) {
    setActionError(null);
    setExportingId(rubric.id);
    try {
      await rubricsApi.exportCsv(rubric.id);
    } catch (cause) {
      setActionError(
        cause instanceof ApiError
          ? `Could not export ${rubric.title}: ${cause.message}`
          : `Could not export ${rubric.title}.`,
      );
    } finally {
      setExportingId(null);
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

      {stats && <DashboardStats stats={stats} />}
      {insights && <DashboardInsights insights={insights} />}

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
        rubrics && (
          <RubricList
            rubrics={rubrics}
            onEdit={setEditing}
            onDelete={handleDelete}
            onExport={handleExport}
            busyId={deletingId}
            exportingId={exportingId}
          />
        )
      )}
    </div>
  );
}
