import { Navigate, useLocation } from "react-router-dom";

import { useAuth } from "../auth/useAuth";

export const dashboardFor = (role) => (role === "teacher" ? "/teacher" : "/student");

/**
 * Gate for anything behind a login.
 *
 * `role` optionally restricts a route to one role. A signed-in user on the
 * wrong route is sent to their own dashboard rather than shown a dead end --
 * the backend still returns 403 for the underlying API calls either way, so
 * this is about not offering a door that cannot open, not about enforcement.
 */
export function ProtectedRoute({ role, children }) {
  const { isAuthenticated, role: currentRole } = useAuth();
  const location = useLocation();

  if (!isAuthenticated) {
    // Remember where they were headed so login can send them back.
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  if (role && currentRole !== role) {
    return <Navigate to={dashboardFor(currentRole)} replace />;
  }

  return children;
}

/** Keeps a signed-in user off the login/register pages. */
export function PublicOnlyRoute({ children }) {
  const { isAuthenticated, role } = useAuth();
  if (isAuthenticated) return <Navigate to={dashboardFor(role)} replace />;
  return children;
}
