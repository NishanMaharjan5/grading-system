/**
 * Word counting and word-range checks, mirroring backend/app/word_limits.py.
 *
 * The two must agree exactly. A student watching a live counter tick past the
 * limit and then being refused by the server for a different number would have
 * no way to tell what the rule is. The server stays the authority; this exists
 * so the answer arrives before the submit button, not after it.
 *
 * Word limits only apply to text rubrics. A code rubric is graded by running
 * tests, so counting words says nothing about it.
 */

/** Whitespace-separated tokens — the same crude rule as the server's
    count_words, chosen so a student counting by eye, the live counter and the
    server all agree. */
export function countWords(text) {
  return (text ?? "").split(/\s+/).filter(Boolean).length;
}

/** The range in words, for a hint or a message. Null when unlimited. */
export function describeRange(minWords, maxWords) {
  if (minWords != null && maxWords != null) return `between ${minWords} and ${maxWords} words`;
  if (maxWords != null) return `at most ${maxWords} words`;
  if (minWords != null) return `at least ${minWords} words`;
  return null;
}

/** True when this rubric restricts length at all. */
export function hasWordLimit(rubric) {
  return rubric?.type === "text" && (rubric.min_words != null || rubric.max_words != null);
}

/**
 * Checks a draft against the rubric's range.
 * Returns { words, state, message }:
 *   state "ok" | "short" | "long" | "none"  ("none" when unrestricted)
 * The message is the same sentence the server would answer with.
 */
export function checkLength(text, rubric) {
  const words = countWords(text);
  if (!hasWordLimit(rubric)) return { words, state: "none", message: null };

  const range = describeRange(rubric.min_words, rubric.max_words);
  if (rubric.min_words != null && words < rubric.min_words) {
    return { words, state: "short", message: `This essay must be ${range}. Yours is ${words}, which is too short.` };
  }
  if (rubric.max_words != null && words > rubric.max_words) {
    return { words, state: "long", message: `This essay must be ${range}. Yours is ${words}, which is too long.` };
  }
  return { words, state: "ok", message: null };
}
