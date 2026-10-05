import { api } from "./client";

export const MAX_CONTENT_LENGTH = 50_000;

export const submissionsApi = {
  list: (options) => api.get("/api/submissions", options),
  get: (id, options) => api.get(`/api/submissions/${id}`, options),
  create: (rubricId, content) => api.post("/api/submissions", { rubric_id: rubricId, content }),
  pending: (options) => api.get("/api/submissions/pending", options),
  /** Empty body approves the AI's suggestion untouched; a payload overrides it. */
  review: (id, payload) => api.put(`/api/submissions/${id}/review`, payload ?? {}),
  // Correcting a grade that was already released. Same validation as review;
  // each change is recorded in the submission's revision history.
  revise: (id, payload) => api.put(`/api/submissions/${id}/revise`, payload ?? {}),
};

/**
 * Maps one backend `detail` string back onto the score inputs.
 *
 * The API reports the first problem it finds, e.g.
 * "criterion_scores[1].final_score 99 is outside 0..5 for 'Evidence'". The
 * index refers to the array that was sent, so criterion_scores must be built
 * in the same order the inputs are rendered for this to point at the right row.
 */
export function reviewErrorsFromDetail(detail, criteriaOrder = []) {
  if (!detail) return { form: "Something went wrong." };

  const indexed = /^criterion_scores\[(\d+)\]\.?(\w+)?/.exec(detail);
  if (indexed) {
    const [, index, field] = indexed;
    const criterionId = criteriaOrder[Number(index)];
    if (criterionId !== undefined) {
      return { criteria: { [criterionId]: { [field === "final_feedback" ? "feedback" : "score"]: detail } } };
    }
  }

  if (/^Every criterion needs/.test(detail)) return { form: detail };
  if (/^No AI score to approve/.test(detail)) return { form: detail };
  return { form: detail };
}

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
