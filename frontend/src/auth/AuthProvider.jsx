import { useCallback, useEffect, useMemo, useState } from "react";

import { authApi, setUnauthorizedHandler } from "../api/client";
import { AuthContext } from "./AuthContext";
import { clearToken, getToken, setToken, userFromToken } from "./token";

/**
 * Holds the signed-in user for the whole app.
 *
 * State is seeded synchronously from localStorage rather than fetched on mount.
 * That matters: with an async check there is a first render where the user
 * looks logged out, and ProtectedRoute would bounce a perfectly valid session
 * to the login page on every refresh. userFromToken() also rejects an expired
 * token, so a stale entry never renders a logged-in shell that 401s instantly.
 */
export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => userFromToken(getToken()));

  const logout = useCallback(() => {
    clearToken();
    setUser(null);
  }, []);

  // Any 401 from anywhere means the session is over -- one place decides that.
  useEffect(() => {
    setUnauthorizedHandler(logout);
    return () => setUnauthorizedHandler(null);
  }, [logout]);

  const adopt = useCallback((accessToken) => {
    setToken(accessToken);
    const next = userFromToken(accessToken);
    setUser(next);
    return next;
  }, []);

  const login = useCallback(
    async (email, password) => adopt((await authApi.login(email, password)).access_token),
    [adopt],
  );

  const register = useCallback(
    async (payload) => adopt((await authApi.register(payload)).access_token),
    [adopt],
  );

  const value = useMemo(
    () => ({
      user,
      role: user?.role ?? null,
      isAuthenticated: user !== null,
      login,
      register,
      logout,
    }),
    [user, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
