import { Outlet, useNavigate } from "react-router-dom";

import { useAuth } from "../auth/useAuth";

export default function Layout() {
  const { user, role, logout } = useAuth();
  const navigate = useNavigate();

  function handleLogout() {
    logout();
    navigate("/login", { replace: true });
  }

  return (
    <div className="shell">
      <header className="shell__header">
        <span className="shell__title">Grading System</span>
        <span className="shell__spacer" />
        <span className="muted">
          {user?.email} · <span className="badge">{role}</span>
        </span>
        <button type="button" className="button--plain" onClick={handleLogout}>
          Sign out
        </button>
      </header>
      <main className="shell__main">
        <Outlet />
      </main>
    </div>
  );
}
