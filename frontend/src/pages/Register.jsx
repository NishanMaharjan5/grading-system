import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { ApiError } from "../api/client";
import { useAuth } from "../auth/useAuth";
import { dashboardFor } from "../routes/ProtectedRoute";

export default function Register() {
  const { register } = useAuth();
  const navigate = useNavigate();

  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
    role: "student",
    teacher_code: "",
  });
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const update = (field) => (event) => setForm({ ...form, [field]: event.target.value });

  async function handleSubmit(event) {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const payload = {
        name: form.name,
        email: form.email,
        password: form.password,
        role: form.role,
      };
      // Only send the code when it applies, so an empty string never trips the gate.
      if (form.role === "teacher" && form.teacher_code) payload.teacher_code = form.teacher_code;

      const user = await register(payload);
      navigate(dashboardFor(user?.role), { replace: true });
    } catch (cause) {
      setError(
        cause instanceof ApiError
          ? cause.detail || "Could not create the account."
          : "Something went wrong. Please try again.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page page--narrow">
      <h1>Create an account</h1>

      <form onSubmit={handleSubmit} noValidate>
        {error && (
          <p className="alert" role="alert">
            {error}
          </p>
        )}

        <label htmlFor="name">Name</label>
        <input id="name" value={form.name} onChange={update("name")} required />

        <label htmlFor="email">Email</label>
        <input id="email" type="email" autoComplete="email" value={form.email} onChange={update("email")} required />

        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          autoComplete="new-password"
          value={form.password}
          onChange={update("password")}
          required
        />
        <p className="hint">At least 8 characters.</p>

        <label htmlFor="role">I am a</label>
        <select id="role" value={form.role} onChange={update("role")}>
          <option value="student">Student</option>
          <option value="teacher">Teacher</option>
        </select>

        {form.role === "teacher" && (
          <>
            <label htmlFor="teacher_code">Teacher signup code</label>
            <input id="teacher_code" value={form.teacher_code} onChange={update("teacher_code")} />
            <p className="hint">Only needed if your institution has set one.</p>
          </>
        )}

        <button type="submit" disabled={busy}>
          {busy ? "Creating…" : "Create account"}
        </button>
      </form>

      <p className="muted">
        Already have an account? <Link to="/login">Sign in</Link>
      </p>
    </div>
  );
}
