/**
 * The single place that talks to the backend.
 *
 * Everything goes through request(): it attaches the JWT, parses the response,
 * and turns a non-2xx into an ApiError carrying the backend's own `detail`
 * string so forms can show the real reason a call failed.
 *
 * 401 handling is delegated rather than done here: the client has no business
 * knowing about React or the router, so AuthContext registers a callback with
 * setUnauthorizedHandler() and owns what "logged out" means.
 */

import { getToken } from "../auth/token";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(status, detail, body) {
    super(detail || `Request failed (${status})`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.body = body;
  }
}

let onUnauthorized = null;

export function setUnauthorizedHandler(handler) {
  onUnauthorized = handler;
}

async function request(path, { method = "GET", body, auth = true, signal } = {}) {
  const headers = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";

  if (auth) {
    const token = getToken();
    if (token) headers.Authorization = `Bearer ${token}`;
  }

  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (cause) {
    if (cause?.name === "AbortError") throw cause;
    // fetch only rejects for network-level problems, which is worth saying
    // plainly rather than surfacing "Failed to fetch".
    throw new ApiError(0, "Could not reach the server. Is the backend running?", null);
  }

  // 204 and other empty bodies must not go through response.json().
  const isJson = (response.headers.get("content-type") || "").includes("application/json");
  const payload = response.status === 204 || !isJson ? null : await response.json().catch(() => null);

  if (!response.ok) {
    if (response.status === 401 && auth && onUnauthorized) onUnauthorized();
    throw new ApiError(response.status, payload?.detail, payload);
  }

  return payload;
}

export const api = {
  get: (path, options) => request(path, { ...options, method: "GET" }),
  post: (path, body, options) => request(path, { ...options, method: "POST", body }),
  put: (path, body, options) => request(path, { ...options, method: "PUT", body }),
  delete: (path, options) => request(path, { ...options, method: "DELETE" }),
};

export const authApi = {
  // register/login are the two calls that must not send a stale Authorization header
  register: (payload) => request("/api/auth/register", { method: "POST", body: payload, auth: false }),
  login: (email, password) =>
    request("/api/auth/login", { method: "POST", body: { email, password }, auth: false }),
  me: () => request("/api/auth/me"),
};
