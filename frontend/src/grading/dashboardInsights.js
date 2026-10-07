/**
 * The proposal's summary dashboard, line 177: "lists the subjects students
 * found difficult, who needs further assistance, and so on." The four stat
 * tiles above this (see dashboardStats.js) answer "how much is there and how
 * busy has it been" -- this answers the two questions above them, and like
 * that row, it is computed entirely from data the page already has to fetch:
 *
 *   rubrics      GET /api/rubrics     (criteria, with name + max_points)
 *   submissions  GET /api/submissions (every submission across this
 *                                      teacher's rubrics, with each grade's
 *                                      final_score -- already scoped)
 *
 * Only a grade with a final_score counts. That is set by the review/revise
 * endpoints and nowhere else, so it is also exactly "approved or revised" --
 * an ai_graded submission awaiting sign-off has no final_score yet, and
 * folding its AI suggestion in here would mix what a teacher has actually
 * released with what they haven't looked at.
 */

// A teacher's own judgement call, not derived from anything in the data --
// picked because it is the ordinary classroom meaning of "struggling" (a
// solid D), not because the data suggests a cutoff. Simpler than "bottom N"
// and doesn't degrade when there are only one or two students: a bottom-N
// cut would flag someone doing fine just because everyone else did better.
const STRUGGLE_THRESHOLD_PERCENT = 60;

export function computeStruggleInsights({ rubrics, submissions }) {
  const rubricsById = new Map(rubrics.map((r) => [r.id, r]));

  // criterion_id -> running percentage total, for the per-criterion average
  const criterionStats = new Map();
  // student_id -> running points total, for the per-student average. Points
  // rather than an average of percentages, so a 2-point and a 20-point
  // criterion are weighted by how much they actually counted.
  const studentStats = new Map();

  for (const submission of submissions) {
    const rubric = rubricsById.get(submission.rubric_id);
    if (!rubric) continue; // defensive: ownership scoping already guarantees this
    const criteriaById = new Map(rubric.criteria.map((c) => [c.id, c]));

    for (const grade of submission.grades ?? []) {
      if (grade.final_score == null) continue; // not released yet
      const criterion = criteriaById.get(grade.criterion_id);
      if (!criterion) continue; // defensive: a criterion a resubmission-edit removed

      const percent = (grade.final_score / criterion.max_points) * 100;

      let cs = criterionStats.get(criterion.id);
      if (!cs) {
        cs = {
          criterionId: criterion.id,
          rubricId: rubric.id,
          rubricTitle: rubric.title,
          criterionName: criterion.name,
          percentSum: 0,
          count: 0,
        };
        criterionStats.set(criterion.id, cs);
      }
      cs.percentSum += percent;
      cs.count += 1;

      let ss = studentStats.get(submission.student_id);
      if (!ss) {
        ss = {
          studentId: submission.student_id,
          name: submission.student_name,
          email: submission.student_email,
          earned: 0,
          possible: 0,
        };
        studentStats.set(submission.student_id, ss);
      }
      ss.earned += grade.final_score;
      ss.possible += criterion.max_points;
    }
  }

  // Lowest average first -- directly "the subjects students found difficult",
  // with the hardest one on top rather than something to go hunting for.
  const criteriaByDifficulty = [...criterionStats.values()]
    .map((cs) => ({ ...cs, avgPercent: Math.round(cs.percentSum / cs.count) }))
    .sort((a, b) => a.avgPercent - b.avgPercent);

  const strugglingStudents = [...studentStats.values()]
    .map((ss) => ({ ...ss, avgPercent: Math.round((ss.earned / ss.possible) * 100) }))
    .filter((ss) => ss.avgPercent < STRUGGLE_THRESHOLD_PERCENT)
    .sort((a, b) => a.avgPercent - b.avgPercent);

  return { criteriaByDifficulty, strugglingStudents, struggleThreshold: STRUGGLE_THRESHOLD_PERCENT };
}
