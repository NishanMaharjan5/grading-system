import WhoAmI from "../components/WhoAmI";

export default function StudentDashboard() {
  return (
    <div className="page">
      <h1>Your work</h1>
      <WhoAmI />
      <p className="muted">Assignments and submitted work will appear here.</p>
    </div>
  );
}
