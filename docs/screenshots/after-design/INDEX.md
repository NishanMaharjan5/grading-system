# Frontend screenshots — after the visual design pass

Same pages, same staged data and same numbering as `../after-styling/` (the
before set), so each pair compares directly. Desktop at 1280px, full page;
21-23 at phone width (390px); two dark-mode pages at the end.

The design system behind these is in `frontend/src/index.css`: a warm paper
ground with white cards, deep indigo as the single accent, Source Serif 4 for
headings over the system sans, and one 4/8/12/16/24/32/48 spacing scale.

- `01-login.png` — Login, empty
- `02-login-error.png` — Login, wrong password (401 message)
- `03-register-student.png` — Register, student
- `04-register-teacher.png` — Register, teacher (signup-code field shown)
- `05-teacher-rubrics.png` — Teacher dashboard: rubric list (locked and editable, text and code)
- `06-rubric-create-empty.png` — Create rubric, empty
- `07-rubric-create-errors.png` — Create rubric, inline validation errors
- `08-rubric-create-code.png` — Create rubric, code type with test cases
- `09-rubric-edit.png` — Edit rubric (prefilled, type locked)
- `10-review-queue.png` — Review queue: AI suggestion (blue) vs manual grading (neutral)
- `11-review-ai-suggestion.png` — Review: essay with AI suggestions
- `12-review-manual.png` — Review: no trained model, manual grading
- `13-review-code.png` — Review: code submission with test results
- `14-student-assignments.png` — Student dashboard: graded, awaiting, not submitted
- `15-submit-code.png` — Submit form, code assignment
- `16-student-grade-approved.png` — Released grade with feedback
- `17-student-awaiting-manual.png` — Awaiting teacher: no trained model
- `18-submit-text.png` — Submit form, written assignment
- `19-student-awaiting-ai.png` — Awaiting teacher: first-pass score prepared (no score shown)
- `20-not-found.png` — Not-found page
- `21-mobile-login.png` — Phone width (390px): login
- `22-mobile-student-assignments.png` — Phone width: student dashboard, header wraps
- `23-mobile-review.png` — Phone width: review form, no overflow
- `dark-01-login.png` — Dark mode: login
- `dark-10-review-queue.png` — Dark mode: review queue
