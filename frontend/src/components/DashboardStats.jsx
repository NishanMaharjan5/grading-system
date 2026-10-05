/**
 * A quiet row of numbers above the rubric list: how much there is, how much
 * is waiting, how busy the last week was, how often the engine's first guess
 * is kept. A lightweight preview of a fuller analytics page, not that page --
 * four facts, no charts, nothing clickable.
 */
export default function DashboardStats({ stats }) {
  const { totalRubrics, pendingReviews, recentSubmissions, aiAcceptanceRate } = stats;

  const tiles = [
    { label: "Rubrics", value: totalRubrics },
    { label: "Pending review", value: pendingReviews, emphasis: pendingReviews > 0 },
    { label: "Submissions, last 7 days", value: recentSubmissions },
    {
      label: "AI suggestions kept",
      value: aiAcceptanceRate === null ? "—" : `${aiAcceptanceRate}%`,
      title: aiAcceptanceRate === null ? "No criterion has been reviewed yet" : undefined,
    },
  ];

  return (
    <div className="stats-row">
      {tiles.map((tile) => (
        <div key={tile.label} className={`stat-tile${tile.emphasis ? " stat-tile--emphasis" : ""}`} title={tile.title}>
          <span className="stat-tile__value">{tile.value}</span>
          <span className="stat-tile__label">{tile.label}</span>
        </div>
      ))}
    </div>
  );
}
