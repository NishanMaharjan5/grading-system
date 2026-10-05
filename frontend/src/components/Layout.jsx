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
        <span className="shell__brand">
          {/* the mark is a rubric: ruled criteria, the top one marked off */}
          <svg className="shell__mark" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <rect x="3" y="2.5" width="18" height="19" rx="3" fill="currentColor" opacity="0.12" />
            <rect x="3" y="2.5" width="18" height="19" rx="3" fill="none" stroke="currentColor" strokeWidth="1.5" />
            <path d="M7.5 9.2l2 2 4.8-4.8" fill="none" stroke="currentColor" strokeWidth="1.9"
                  strokeLinecap="round" strokeLinejoin="round" />
            <path d="M7.5 14.5h9M7.5 17.8h5.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
          <span className="shell__title">Grading System</span>
        </span>
        <nav className="shell__nav">
          {role === "teacher" ? (
            <>
              <NavLink to="/teacher" end>Rubrics</NavLink>
              <NavLink to="/teacher/review">Review queue</NavLink>
            </>
          ) : (
            <>
              <NavLink to="/student" end>Assignments</NavLink>
              <NavLink to="/student/submissions">Your submissions</NavLink>
            </>
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
