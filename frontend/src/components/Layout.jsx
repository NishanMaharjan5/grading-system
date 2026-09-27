import { NavLink, Outlet, useNavigate } from "react-router-dom";

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
        <nav className="shell__nav">
          {role === "teacher" ? (
            <>
              <NavLink to="/teacher" end>Rubrics</NavLink>
              <NavLink to="/teacher/review">Review queue</NavLink>
            </>
          ) : (
            <NavLink to="/student" end>Assignments</NavLink>
          )}
        </nav>
        <span className="shell__spacer" />
        <span className="shell__user muted">
          <span className="shell__email" title={user?.email}>{user?.email}</span>
          <span className="badge">{role}</span>
        </span>
        <button type="button" className="button--plain shell__signout" onClick={handleLogout}>
          Sign out
        </button>
      </header>
      <main className="shell__main">
        <Outlet />
      </main>
    </div>
  );
}
