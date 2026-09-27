/**
 * Trims generated feedback where it would only repeat what's printed next to it.
 *
 * The backend writes feedback to stand on its own -- "Thesis: 4/5 — solid.
 * ..." -- because it was asked to name its criterion rather than read as
 * generic filler, and it has to make sense wherever it's shown. But on screen
 * it always sits directly under a heading that already says "Thesis" and
 * "4 / 5", so that lead-in is removed for display. The stored text is left
 * as it is.
 *
 * Only the exact generated forms are removed: "<criterion>: " and then
 * "<score>/<max> — " (or "--", which feedback stored before the dash fix
 * uses). Anything a teacher typed is shown exactly as written.
 */

// Matches Python's "{:g}" for the values scores take: 4.0 -> "4", 3.5 -> "3.5".
const formatPoints = (value) => String(Number(value));

const escapeRegExp = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const capitaliseFirst = (text) => text.charAt(0).toUpperCase() + text.slice(1);

export function criterionFeedbackForDisplay(text, { criterionName, score, maxPoints, description, hideDescription }) {
  if (!text) return text;
  let trimmed = text;

  const namePrefix = `${criterionName}: `;
  if (trimmed.startsWith(namePrefix)) {
    trimmed = trimmed.slice(namePrefix.length);
    if (score !== null && score !== undefined && maxPoints !== undefined) {
      const scorePrefix = new RegExp(
        `^${escapeRegExp(formatPoints(score))}/${escapeRegExp(formatPoints(maxPoints))} (—|--) `,
      );
      trimmed = trimmed.replace(scorePrefix, "");
    }
    trimmed = capitaliseFirst(trimmed);
  }

  // Low scores end by quoting the criterion's description. Where that
  // description is already on screen, the quote is dropped.
  if (hideDescription && description) {
    const quoted = ` What this criterion asks for: ${description}`;
    if (trimmed.endsWith(quoted)) trimmed = trimmed.slice(0, -quoted.length);
  }
  return trimmed;
}

/** "Overall 13/15 — solid. Strongest: ..." under a heading that already shows 13 / 15. */
export function summaryForDisplay(text, { total, maxTotal }) {
  if (!text || total === null || total === undefined) return text;
  const prefix = new RegExp(
    `^Overall ${escapeRegExp(formatPoints(total))}/${escapeRegExp(formatPoints(maxTotal))} (—|--) `,
  );
  return prefix.test(text) ? capitaliseFirst(text.replace(prefix, "")) : text;
}
