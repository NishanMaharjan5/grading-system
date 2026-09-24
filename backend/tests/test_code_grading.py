"""The sandboxed code path: what the sandbox contains, and how test results
turn into a grade.

The containment tests below assert what this sandbox actually does, which is
less than a container does. See SANDBOX.md.
"""

import os

import pytest

from app.grading.code_runner import MEMORY_BYTES, normalise_output, run_test_cases


class FakeCase:
    """Stands in for a TestCase row -- the runner only reads these two fields."""

    __test__ = False

    def __init__(self, stdin="", expected_output="", id=1):
        self.stdin = stdin
        self.expected_output = expected_output
        self.id = id


def run(source, stdin="", expected=""):
    return run_test_cases(source, [FakeCase(stdin, expected)])[0]


class TestOutputComparison:
    @pytest.mark.parametrize(
        "actual,expected,same",
        [
            ("3\n", "3", True),
            ("3", "3\n\n", True),
            ("3 \n", "3", True),          # trailing spaces ignored
            ("  3", "3", False),          # leading spaces are not
            ("3\n4", "3\n4", True),
            ("3\n\n4", "3\n4", False),    # interior blank lines matter
            ("Three", "three", False),    # case matters
        ],
    )
    def test_normalisation_rules(self, actual, expected, same):
        assert (normalise_output(actual) == normalise_output(expected)) is same


class TestRunning:
    def test_a_correct_program_passes(self):
        assert run("print(input().upper())", "hello", "HELLO")["passed"] is True

    def test_wrong_output_fails_without_erroring(self):
        result = run("print('nope')", "hello", "HELLO")
        assert result["passed"] is False
        assert result["outcome"] == "ok"      # it ran fine, it was just wrong
        assert result["detail"] == "Output did not match."

    def test_a_crash_is_reported_not_raised(self):
        result = run("raise ValueError('boom')")
        assert result["passed"] is False
        assert result["outcome"] == "error"
        assert "ValueError: boom" in result["stderr"]

    def test_a_syntax_error_is_a_failed_test(self):
        assert run("def (:")["outcome"] == "error"

    def test_stdin_reaches_the_program(self):
        assert run("import sys; print(sum(int(n) for n in sys.stdin.read().split()))", "1 2 3", "6")["passed"]

    def test_each_case_runs_independently(self):
        results = run_test_cases(
            "print(int(input()) * 2)",
            [FakeCase("2", "4", 1), FakeCase("5", "10", 2), FakeCase("7", "99", 3)],
        )
        assert [r["passed"] for r in results] == [True, True, False]


class TestResourceLimits:
    def test_an_infinite_loop_is_stopped(self):
        result = run("while True: pass")
        assert result["passed"] is False
        assert result["outcome"] in ("timeout", "error")  # CPU rlimit usually bites first

    def test_a_memory_bomb_is_stopped(self):
        """macOS refuses RLIMIT_AS entirely, so this is caught by the parent
        polling RSS rather than by the kernel."""
        result = run(f"x = bytearray({MEMORY_BYTES * 4})")
        assert result["passed"] is False
        assert result["outcome"] in ("memory", "error")


class TestContainment:
    """What the sandbox stops. Everything here is best-effort except the
    file-write limit, which the kernel enforces."""

    def test_writing_a_file_is_blocked(self, tmp_path):
        target = tmp_path / "written_by_student.txt"
        result = run(f"open({str(target)!r}, 'w').write('x')")
        assert result["outcome"] == "error"
        assert not target.exists()

    def test_reading_an_unrelated_file_is_blocked(self):
        assert run("print(open('/etc/passwd').read())")["outcome"] == "error"

    def test_reading_the_servers_env_file_is_blocked(self):
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        result = run(f"print(open({env_path!r}).read())")
        assert result["outcome"] == "error"
        assert "JWT_SECRET" not in result["actual"]

    def test_opening_a_socket_is_blocked(self):
        assert run("import socket; socket.socket().connect(('1.1.1.1', 80))")["outcome"] == "error"

    def test_spawning_a_process_is_blocked(self):
        assert run("import subprocess; subprocess.run(['whoami'])")["outcome"] == "error"

    def test_os_system_is_blocked(self):
        assert run("import os; os.system('echo hi')")["outcome"] == "error"

    def test_the_program_can_still_use_the_stdlib(self):
        assert run("import json, math; print(json.dumps(round(math.pi, 2)))", "", "3.14")["passed"]

    def test_the_program_can_read_its_own_file(self):
        assert run("print('ok' if open('main.py').read() else 'empty')", "", "ok")["passed"]


class TestCodeRubricGrading:
    """End to end through the submission endpoint."""

    @pytest.fixture
    def code_rubric(self, client, auth, teacher):
        response = client.post(
            "/api/rubrics",
            json={
                "title": "Doubling",
                "type": "code",
                "criteria": [
                    {"name": "Doubles", "max_points": 10, "test_cases": [
                        {"stdin": "2", "expected_output": "4"},
                        {"stdin": "5", "expected_output": "10"},
                    ]},
                    {"name": "Handles zero", "max_points": 5, "test_cases": [
                        {"stdin": "0", "expected_output": "0"},
                    ]},
                ],
            },
            headers=auth(teacher),
        )
        assert response.status_code == 201, response.get_json()
        return response.get_json()

    def submit(self, client, auth, student, rubric, source):
        return client.post(
            "/api/submissions", json={"rubric_id": rubric["id"], "content": source}, headers=auth(student)
        ).get_json()

    def test_a_fully_correct_program_scores_full_marks(self, client, auth, teacher, student, code_rubric):
        body = self.submit(client, auth, student, code_rubric, "print(int(input()) * 2)")
        assert body["status"] == "ai_graded"

        graded = client.get(f"/api/submissions/{body['id']}", headers=auth(teacher)).get_json()
        assert graded["ai_total"] == 15.0
        assert all("passed 1 of 1" in g["ai_feedback"] or "passed 2 of 2" in g["ai_feedback"]
                   for g in graded["grades"])

    def test_a_partly_correct_program_scores_proportionally(self, client, auth, teacher, student, code_rubric):
        # Right for positive numbers, wrong for zero.
        source = "n = int(input())\nprint(n * 2 if n else 99)"
        body = self.submit(client, auth, student, code_rubric, source)

        graded = client.get(f"/api/submissions/{body['id']}", headers=auth(teacher)).get_json()
        by_name = {c["name"]: c["id"] for c in code_rubric["criteria"]}
        scores = {g["criterion_id"]: g["ai_score"] for g in graded["grades"]}
        assert scores[by_name["Doubles"]] == 10.0       # 2 of 2
        assert scores[by_name["Handles zero"]] == 0.0   # 0 of 1
        assert graded["ai_total"] == 10.0

    def test_feedback_names_the_failing_test_with_both_outputs(self, client, auth, teacher, student, code_rubric):
        body = self.submit(client, auth, student, code_rubric, "print('always wrong')")
        graded = client.get(f"/api/submissions/{body['id']}", headers=auth(teacher)).get_json()
        text = "\n".join(g["ai_feedback"] for g in graded["grades"])
        assert "passed 0 of 2" in text
        assert "expected:" in text and "actual:" in text
        assert "always wrong" in text

    def test_a_crashing_program_is_graded_not_failed(self, client, auth, teacher, student, code_rubric):
        """A crash is a wrong answer, not a broken grader."""
        body = self.submit(client, auth, student, code_rubric, "raise SystemError('kaboom')")
        assert body["status"] == "ai_graded"

        graded = client.get(f"/api/submissions/{body['id']}", headers=auth(teacher)).get_json()
        assert graded["ai_total"] == 0.0
        assert "stopped with an error" in "\n".join(g["ai_feedback"] for g in graded["grades"])

    def test_a_hanging_program_is_graded_not_failed(self, client, auth, teacher, student, code_rubric):
        body = self.submit(client, auth, student, code_rubric, "while True: pass")
        assert body["status"] == "ai_graded"
        assert client.get(f"/api/submissions/{body['id']}",
                          headers=auth(teacher)).get_json()["ai_total"] == 0.0

    def test_a_code_rubric_without_tests_cannot_be_created(self, client, auth, teacher):
        response = client.post(
            "/api/rubrics",
            json={"title": "No tests", "type": "code", "criteria": [{"name": "C", "max_points": 5}]},
            headers=auth(teacher),
        )
        assert response.status_code == 422
        assert "test" in response.get_json()["detail"].lower()

    def test_a_text_rubric_may_not_carry_test_cases(self, client, auth, teacher):
        response = client.post(
            "/api/rubrics",
            json={"title": "Essay", "type": "text", "criteria": [
                {"name": "C", "max_points": 5, "test_cases": [{"stdin": "", "expected_output": "x"}]}]},
            headers=auth(teacher),
        )
        assert response.status_code == 422

    def test_a_test_case_needs_an_expected_output(self, client, auth, teacher):
        response = client.post(
            "/api/rubrics",
            json={"title": "Bad", "type": "code", "criteria": [
                {"name": "C", "max_points": 5, "test_cases": [{"stdin": "1"}]}]},
            headers=auth(teacher),
        )
        assert response.status_code == 422

    def test_students_never_see_the_expected_outputs(self, client, auth, student, code_rubric):
        body = client.get(f"/api/rubrics/{code_rubric['id']}", headers=auth(student)).get_json()
        assert "test_cases" not in body["criteria"][0]
        assert body["criteria"][0]["test_case_count"] == 2
        assert "expected_output" not in str(body)

    def test_the_owner_does_see_them(self, client, auth, teacher, code_rubric):
        body = client.get(f"/api/rubrics/{code_rubric['id']}", headers=auth(teacher)).get_json()
        assert [t["expected_output"] for t in body["criteria"][0]["test_cases"]] == ["4", "10"]

    def test_a_code_grade_flows_through_review_unchanged(self, client, auth, teacher, student, code_rubric):
        body = self.submit(client, auth, student, code_rubric, "print(int(input()) * 2)")

        queue = client.get("/api/submissions/pending", headers=auth(teacher)).get_json()
        assert [s["id"] for s in queue] == [body["id"]]

        approved = client.put(f"/api/submissions/{body['id']}/review", json={}, headers=auth(teacher)).get_json()
        assert approved["status"] == "approved"
        assert approved["final_total"] == 15.0
        assert all(g["ai_accepted"] is True for g in approved["grades"])

        seen = client.get(f"/api/submissions/{body['id']}", headers=auth(student)).get_json()
        assert seen["final_total"] == 15.0
        assert "ai_total" not in seen
