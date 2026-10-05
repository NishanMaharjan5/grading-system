/**
 * The teacher dashboard's summary row, computed entirely from data the page
 * already has to fetch for other reasons -- no new tracking, no new
 * endpoint. Three lists go in, four numbers come out:
 *
 *   rubrics      GET /api/rubrics       (already scoped to this teacher)
 *   submissions  GET /api/submissions   (every submission across their
 *                                        rubrics, with each grade's
 *                                        ai_accepted -- already scoped)
 *   pending      GET /api/submissions/pending
 *
 * All three endpoints already enforce "this teacher's own rubrics only"
 * server-side, so there is no separate ownership check to get right here.
 */

const WEEK_MS = 7 * 24 * 60 * 60 * 1000;

export function computeDashboardStats({ rubrics, submissions, pending }, now = Date.now()) {
  const cutoff = now - WEEK_MS;
  const recentSubmissions = submissions.filter((s) => new Date(s.created_at).getTime() >= cutoff).length;

  // ai_accepted is true/false only once a criterion has been reviewed; it is
  // null for anything not yet graded, and is deliberately cleared again if a
  // teacher later revises that score by hand (see GradeRevision) -- a hand
  // correction is the teacher's own judgement, not a reading on the AI, so it
  // rightly drops out of this rate rather than counting as a rejection.
  let accepted = 0;
  let reviewed = 0;
  for (const submission of submissions) {
    for (const grade of submission.grades ?? []) {
      if (grade.ai_accepted === true) {
        accepted += 1;
        reviewed += 1;
      } else if (grade.ai_accepted === false) {
        reviewed += 1;
      }
    }
  }

  return {
    totalRubrics: rubrics.length,
    pendingReviews: pending.length,
    recentSubmissions,
    // null, not 0 -- "nothing reviewed yet" and "the AI is never right" are
    // different facts, and 0% would quietly claim the second.
    aiAcceptanceRate: reviewed === 0 ? null : Math.round((accepted / reviewed) * 100),
  };
}
