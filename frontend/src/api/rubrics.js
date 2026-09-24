import { api } from "./client";

export const rubricsApi = {
  list: (options) => api.get("/api/rubrics", options),
  get: (id, options) => api.get(`/api/rubrics/${id}`, options),
  create: (payload) => api.post("/api/rubrics", payload),
  update: (id, payload) => api.put(`/api/rubrics/${id}`, payload),
  remove: (id) => api.delete(`/api/rubrics/${id}`),
};

/**
 * Turns one backend `detail` string into per-field errors.
 *
 * The API reports the first problem it finds as prose, e.g.
 * "criteria[1].max_points must be greater than 0". Pulling the index and field
 * back out lets the message sit under the input it refers to instead of in a
 * banner above a form with six identical-looking rows.
 *
 * Anything unrecognised falls through to `form`, so a message is never
 * swallowed just because it didn't match a pattern.
 */
export function fieldErrorsFromDetail(detail) {
  if (!detail) return { form: "Something went wrong." };

  const perCriterion = /^criteria\[(\d+)\]\.?(\w+)?/.exec(detail);
  if (perCriterion) {
    const [, index, field] = perCriterion;
    return { criteria: { [Number(index)]: { [field || "name"]: detail } } };
  }

  if (/^criteria\b/.test(detail)) return { criteriaForm: detail };
  if (/^title\b/.test(detail)) return { title: detail };
  if (/^type\b/.test(detail)) return { type: detail };
  if (/^Every criterion needs/.test(detail)) return { criteriaForm: detail };

  return { form: detail };
}
