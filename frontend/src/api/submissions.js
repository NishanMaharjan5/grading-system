import { api } from "./client";

export const MAX_CONTENT_LENGTH = 50_000;

export const submissionsApi = {
  list: (options) => api.get("/api/submissions", options),
  get: (id, options) => api.get(`/api/submissions/${id}`, options),
  create: (rubricId, content) => api.post("/api/submissions", { rubric_id: rubricId, content }),
};

/**
 * What a student is told about each status.
 *
 * Two of these matter more than they look. `ai_graded` must not imply a score
 * is being withheld arbitrarily or hint at what it is -- a grade isn't real
 * until a teacher releases it. `grading_failed` is the normal path for any
 * rubric without a trained model, so it reads as routine rather than as a
 * fault the student caused or needs to act on.
 */
export const STATUS_COPY = {
  submitted: {
    label: "Received",
    tone: "neutral",
    detail: "Your work has been saved. Grading hasn't run yet.",
  },
  ai_graded: {
    label: "Awaiting your teacher",
    tone: "pending",
    detail:
      "Your work has been received and a first-pass score has been prepared. " +
      "Nothing is final until your teacher reviews and releases it, so there is no grade to show yet.",
  },
  grading_failed: {
    label: "Awaiting your teacher",
    tone: "pending",
    detail:
      "Your work has been received. Automatic grading isn't set up for this rubric, " +
      "so your teacher will grade it directly. There is nothing wrong with your submission.",
  },
  approved: {
    label: "Graded",
    tone: "done",
    detail: "Your teacher has reviewed and released this grade.",
  },
};

export const statusCopy = (status) =>
  STATUS_COPY[status] ?? { label: status, tone: "neutral", detail: "" };
