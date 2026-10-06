/**
 * Light or dark: an explicit choice if one has been made, the OS preference
 * otherwise.
 *
 * Whichever it is, it ends up in one place -- data-theme on <html> -- and
 * index.css defines the dark palette under :root[data-theme="dark"]. Resolving
 * the OS preference here rather than leaving it to a prefers-color-scheme
 * query in the stylesheet is what keeps that palette to a single copy; see the
 * comment above it.
 *
 * localStorage holds only an explicit choice, so "no entry" means "follow the
 * OS". Choosing the theme the OS already prefers still stores it, because the
 * point of choosing is that a later OS change shouldn't undo it.
 *
 * The theme is read by one component today (the header toggle) but changes
 * from two places -- that toggle and the OS -- so it is kept as a small
 * subscribable store rather than component state, and React reads it through
 * useSyncExternalStore in ./useTheme.
 */

const STORAGE_KEY = "grading-system:theme";

/* --paper in each theme, mirrored into <meta name="theme-color"> so a phone's
   browser chrome matches the page instead of staying on the light value. */
const BROWSER_CHROME = { light: "#f7f5f1", dark: "#191816" };

function darkQuery() {
  return window.matchMedia("(prefers-color-scheme: dark)");
}

export function systemTheme() {
  return darkQuery().matches ? "dark" : "light";
}

/** The stored choice, or null if none has been made. */
export function storedTheme() {
  // Guarded like auth/token.js: access alone throws when storage is blocked.
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return stored === "light" || stored === "dark" ? stored : null;
  } catch {
    return null;
  }
}

function apply(theme) {
  document.documentElement.dataset.theme = theme;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", BROWSER_CHROME[theme]);
}

let current = storedTheme() ?? systemTheme();
const listeners = new Set();

export function getTheme() {
  return current;
}

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function change(theme) {
  if (theme === current) return;
  current = theme;
  apply(theme);
  for (const listener of listeners) listener();
}

/** Record and apply an explicit choice. */
export function setTheme(theme) {
  try {
    window.localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // A blocked store costs persistence across refreshes, not the toggle.
  }
  change(theme);
}

/**
 * Apply the resolved theme, and keep following the OS for as long as no
 * explicit choice has been made. Called once, before the app renders.
 */
export function startTheme() {
  apply(current);
  darkQuery().addEventListener("change", () => {
    if (storedTheme() === null) change(systemTheme());
  });
}
