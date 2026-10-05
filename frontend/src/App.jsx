import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import { AuthProvider } from "./auth/AuthProvider";
import { useAuth } from "./auth/useAuth";
import Layout from "./components/Layout";
import Login from "./pages/Login";
import NotFound from "./pages/NotFound";
import Register from "./pages/Register";
import ReviewQueue from "./pages/ReviewQueue";
import ReviewSubmission from "./pages/ReviewSubmission";
import StudentDashboard from "./pages/StudentDashboard";
import StudentHistory from "./pages/StudentHistory";
import SubmissionView from "./pages/SubmissionView";
import SubmitWork from "./pages/SubmitWork";
import TeacherDashboard from "./pages/TeacherDashboard";
import { dashboardFor, ProtectedRoute, PublicOnlyRoute } from "./routes/ProtectedRoute";

/** "/" is wherever this particular user belongs. */
function Home() {
  const { isAuthenticated, role } = useAuth();
  return <Navigate to={isAuthenticated ? dashboardFor(role) : "/login"} replace />;
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />

      <Route
        path="/login"
        element={
          <PublicOnlyRoute>
            <Login />
          </PublicOnlyRoute>
        }
      />
      <Route
        path="/register"
        element={
          <PublicOnlyRoute>
            <Register />
          </PublicOnlyRoute>
        }
      />

      <Route
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route
          path="/student"
          element={
            <ProtectedRoute role="student">
              <StudentDashboard />
            </ProtectedRoute>
          }
        />
        <Route
          path="/student/submissions"
          element={
            <ProtectedRoute role="student">
              <StudentHistory />
            </ProtectedRoute>
          }
        />
        <Route
          path="/student/submit/:rubricId"
          element={
            <ProtectedRoute role="student">
              <SubmitWork />
            </ProtectedRoute>
          }
        />
        <Route
          path="/student/submissions/:submissionId"
          element={
            <ProtectedRoute role="student">
              <SubmissionView />
            </ProtectedRoute>
          }
        />
        <Route
          path="/teacher"
          element={
            <ProtectedRoute role="teacher">
              <TeacherDashboard />
            </ProtectedRoute>
          }
        />
        <Route
          path="/teacher/review"
          element={
            <ProtectedRoute role="teacher">
              <ReviewQueue />
            </ProtectedRoute>
          }
        />
        <Route
          path="/teacher/review/:submissionId"
          element={
            <ProtectedRoute role="teacher">
              <ReviewSubmission />
            </ProtectedRoute>
          }
        />
      </Route>

      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}
