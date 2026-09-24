/**
 * JWT storage and decoding.
 *
 * The token is held in a module variable and mirrored into localStorage so a
 * refresh doesn't log the user out. Every localStorage access is wrapped:
 * it throws in some private-browsing modes, and a storage failure should not
 * take the app down.
 *
 * Everything decoded here is for deciding what to *show*. It is not a security
 * boundary -- the payload is readable and editable by anyone holding the token.
 * The backend re-verifies the signature and the role on every request, so a
 * tampered role buys nothing but a broken-looking UI.
 */

const STORAGE_KEY = "grading_system_token";

let inMemoryToken = null;

function readStorage() {
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

function writeStorage(token) {
  try {
    if (token === null) window.localStorage.removeItem(STORAGE_KEY);
    else window.localStorage.setItem(STORAGE_KEY, token);
  } catch {
    // Storage unavailable -- the in-memory copy still works for this tab.
  }
}

export function getToken() {
  if (inMemoryToken === null) inMemoryToken = readStorage();
  return inMemoryToken;
}

export function setToken(token) {
  inMemoryToken = token;
  writeStorage(token);
}

export function clearToken() {
  inMemoryToken = null;
  writeStorage(null);
}

/** Decodes a JWT payload, or returns null if it isn't a readable token. */
export function decodeToken(token) {
  if (!token) return null;
  const payload = token.split(".")[1];
  if (!payload) return null;

  try {
    const base64 = payload.replace(/-/g, "+").replace(/_/g, "/");
    const padded = base64.padEnd(base64.length + ((4 - (base64.length % 4)) % 4), "=");
    // decodeURIComponent/escape round-trip so non-ASCII names survive
    const json = decodeURIComponent(
      atob(padded)
        .split("")
        .map((c) => `%${c.charCodeAt(0).toString(16).padStart(2, "0")}`)
        .join(""),
    );
    return JSON.parse(json);
  } catch {
    return null;
  }
}

export function isExpired(claims) {
  if (!claims?.exp) return false; // no expiry claim -- let the server decide
  return claims.exp * 1000 <= Date.now();
}

/**
 * The signed-in user as the UI understands them, or null when there is no
 * usable token. Expired tokens count as no token, so a stale localStorage
 * entry doesn't render a logged-in shell that 401s on first use.
 */
export function userFromToken(token) {
  const claims = decodeToken(token);
  if (!claims || isExpired(claims)) return null;
  return {
    id: claims.sub,
    email: claims.email,
    role: claims.role,
  };
}
