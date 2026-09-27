/**
 * Keeps on-screen validation errors in step with what the user is typing.
 *
 * `shown` is the error object currently on screen; `fresh` is the result of
 * re-running the form's own validation against the current values. Only
 * errors that are already showing are updated: a field that is now valid
 * clears straight away, a field that is still wrong keeps an error (with the
 * current message, so "Enter a number." can become "Must be greater than
 * 0."), and a field that was never flagged stays quiet until the next submit
 * rather than lighting up as soon as someone starts typing.
 *
 * Re-validating the whole form, rather than just the edited field, is what
 * lets fixing one row clear an error on another -- renaming criterion 1 clears
 * the "duplicate" error sitting on criterion 2.
 *
 * `form` is a banner-level message (usually from the server, or a stale-page
 * warning) and is left alone: it describes the last submission, not a field,
 * so it stays until the next one.
 */
export function refreshShownErrors(shown, fresh) {
  const next = {};
  for (const [key, value] of Object.entries(shown)) {
    if (key === "form") {
      next.form = value;
    } else if (key === "criteria") {
      const rows = {};
      for (const [row, fields] of Object.entries(value)) {
        const kept = {};
        for (const field of Object.keys(fields)) {
          const message = fresh.criteria?.[row]?.[field];
          if (message) kept[field] = message;
        }
        if (Object.keys(kept).length) rows[row] = kept;
      }
      if (Object.keys(rows).length) next.criteria = rows;
    } else if (fresh[key]) {
      next[key] = fresh[key];
    }
  }
  return next;
}

/** Same object back when nothing changed, so React can skip the re-render. */
export function sameErrors(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}
