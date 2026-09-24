import { useEffect, useState } from "react";

import { ApiError, authApi } from "../api/client";

/**
 * Placeholder that proves the plumbing: an authenticated GET that reaches the
 * backend, carries the JWT, and comes back with the server's own view of who
 * this is. Replaced once the real dashboards land.
 */
export default function WhoAmI() {
  const [state, setState] = useState({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    authApi
      .me({ signal: controller.signal })
      .then((data) => setState({ status: "ok", data }))
      .catch((cause) => {
        if (cause?.name === "AbortError") return;
        setState({
          status: "error",
          message: cause instanceof ApiError ? cause.message : "Request failed",
        });
      });
    return () => controller.abort();
  }, []);

  if (state.status === "loading") return <p className="muted">Checking your session…</p>;
  if (state.status === "error") return <p className="alert">{state.message}</p>;

  return (
    <div className="card">
      <p className="muted">Verified against the backend:</p>
      <dl className="kv">
        <dt>Name</dt>
        <dd>{state.data.name}</dd>
        <dt>Email</dt>
        <dd>{state.data.email}</dd>
        <dt>Role</dt>
        <dd>{state.data.role}</dd>
      </dl>
    </div>
  );
}
