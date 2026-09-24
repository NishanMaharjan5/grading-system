import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

import { ApiError } from "../api/client";
import { useAuth } from "../auth/useAuth";
import { dashboardFor } from "../routes/ProtectedRoute";

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const user = await login(email, password);
      // Back where they were headed, or their own dashboard.
      const target = location.state?.from?.pathname ?? dashboardFor(user?.role);
      navigate(target, { replace: true });
    } catch (cause) {
      setError(
        cause instanceof ApiError
          ? cause.detail || "Could not sign in."
          : "Something went wrong. Please try again.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page page--narrow">
      <h1>Sign in</h1>

      <form onSubmit={handleSubmit} noValidate>
        {error && (
          <p className="alert" role="alert">
            {error}
          </p>
        )}

        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          autoComplete="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />

        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />

        <button type="submit" disabled={busy}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>

      <p className="muted">
        No account? <Link to="/register">Create one</Link>
      </p>
    </div>
  );
}
