import { Link } from "react-router-dom";

import { dashboardFor } from "../routes/ProtectedRoute";
import { useAuth } from "../auth/useAuth";

export default function NotFound() {
  const { isAuthenticated, role } = useAuth();
  return (
    <div className="page page--narrow">
      <h1>Page not found</h1>
      <p className="muted">
        <Link to={isAuthenticated ? dashboardFor(role) : "/login"}>Go back</Link>
      </p>
    </div>
  );
}
