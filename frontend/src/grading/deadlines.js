/**
 * Deadline formatting and the past-due check, mirroring backend/app/deadlines.py.
 *
 * The server is the authority — a page left open across the deadline must not
 * be able to submit — but a student should see that an assignment has closed
 * before they write an essay for it, not after.
 *
 * A rubric with no due_date has no deadline and shows no deadline UI at all.
 */

export function hasDeadline(rubric) {
  return Boolean(rubric?.due_date);
}

export function dueDate(rubric) {
  return rubric?.due_date ? new Date(rubric.due_date) : null;
}

export function isPastDue(rubric, now = new Date()) {
  const due = dueDate(rubric);
  return due !== null && now > due;
}

/** "3 October 2026, 17:00" in the reader's own locale and timezone. */
export function formatDue(rubric) {
  const due = dueDate(rubric);
  if (!due) return null;
  return due.toLocaleString(undefined, {
    weekday: "short", day: "numeric", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

/** How far off the deadline is, in the roundest useful unit. */
export function relativeToDeadline(rubric, now = new Date()) {
  const due = dueDate(rubric);
  if (!due) return null;

  const minutes = Math.round((due - now) / 60000);
  const past = minutes < 0;
  const size = Math.abs(minutes);

  let amount;
  if (size < 1) amount = "less than a minute";
  else if (size < 60) amount = `${size} minute${size === 1 ? "" : "s"}`;
  else if (size < 60 * 24) {
    const hours = Math.round(size / 60);
    amount = `${hours} hour${hours === 1 ? "" : "s"}`;
  } else {
    const days = Math.round(size / (60 * 24));
    amount = `${days} day${days === 1 ? "" : "s"}`;
  }
  return past ? `${amount} ago` : `in ${amount}`;
}

/** The sentence shown when an assignment has closed. Matches the server's. */
export const CLOSED_MESSAGE =
  "The deadline for this assignment has passed, so it is no longer accepting submissions. " +
  "Speak to your teacher if you think this is wrong.";

/** For <input type="datetime-local">, which wants local time with no zone. */
export function toDateTimeLocal(isoString) {
  if (!isoString) return "";
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return "";
  const pad = (n) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
    `T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/** Back the other way: a local-time field value to an ISO instant. */
export function fromDateTimeLocal(value) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toISOString();
}
