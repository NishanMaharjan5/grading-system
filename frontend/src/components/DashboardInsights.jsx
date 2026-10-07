/**
 * The proposal's summary dashboard (line 177): which subjects students found
 * difficult, and who needs further assistance. Two small, static panels --
 * a bar per criterion, a table of students below the struggle threshold --
 * not the fuller analytics page the stat tiles above already say they are a
 * preview of.
 */
export default function DashboardInsights({ insights }) {
  const { criteriaByDifficulty, strugglingStudents, struggleThreshold } = insights;

  return (
    <div className="insights">
      <section className="card insight-card">
        <h2 className="insight-card__title">Where students are struggling</h2>
        {criteriaByDifficulty.length === 0 ? (
          <p className="muted">No grades have been released yet, so there's nothing to compare.</p>
        ) : (
          <ul className="criterion-bars">
            {criteriaByDifficulty.map((c) => (
              <li key={c.criterionId} className="criterion-bar">
                <div className="criterion-bar__head">
                  <span>
                    {c.criterionName}
                    <span className="muted"> — {c.rubricTitle}</span>
                  </span>
                  <span className="criterion-bar__value">{c.avgPercent}% avg</span>
                </div>
                <div className="criterion-bar__track">
                  <div
                    className={`criterion-bar__fill${c.avgPercent < struggleThreshold ? " criterion-bar__fill--low" : ""}`}
                    style={{ width: `${Math.max(c.avgPercent, 2)}%` }}
                  />
                </div>
                <span className="muted criterion-bar__count">
                  {c.count} graded {c.count === 1 ? "submission" : "submissions"}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="card insight-card">
        <h2 className="insight-card__title">Students who may need help</h2>
        {strugglingStudents.length === 0 ? (
          <p className="muted">
            {criteriaByDifficulty.length === 0
              ? "No grades have been released yet, so there's nothing to compare."
              : `No one is currently averaging below ${struggleThreshold}% across your rubrics.`}
          </p>
        ) : (
          <table className="insight-table">
            <thead>
              <tr>
                <th scope="col">Student</th>
                <th scope="col">Average</th>
              </tr>
            </thead>
            <tbody>
              {strugglingStudents.map((s) => (
                <tr key={s.studentId}>
                  <td>
                    {s.name}
                    <span className="muted insight-table__email"> {s.email}</span>
                  </td>
                  <td>
                    <span className="badge badge--pending">{s.avgPercent}% avg</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
