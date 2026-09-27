"""The sandboxed code path: what the sandbox contains, and how test results
turn into a grade.

The containment tests below assert what this sandbox actually does, which is
less than a container does. See SANDBOX.md.
"""

import errno
import os
import re
import socket
import subprocess
import tempfile

import pytest

from app.grading import code_runner
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
    """What the sandbox stops, exercised through normal Python calls (open(),
    socket, subprocess) -- these are caught by the in-process guard in
    _sandbox_runner.py before they would even reach the kernel layer, so
    passing here does not by itself prove OS-level enforcement. See
    TestKernelEnforcement below for that: it bypasses the in-process guard
    entirely with raw ctypes syscalls, so a denial there can only have come
    from the Seatbelt profile in code_runner.py."""

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


RAW_OPEN = """
import ctypes, ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
fd = libc.open({path!r}.encode(), 0)  # O_RDONLY
if fd < 0:
    print("DENIED errno=%d" % ctypes.get_errno())
else:
    buf = ctypes.create_string_buffer(4096)
    n = libc.read(fd, buf, 4096)
    libc.close(fd)
    print("LEAKED:" + buf.raw[:n].decode(errors="replace"))
"""

RAW_CONNECT = """
import ctypes, ctypes.util, struct, socket
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
fd = libc.socket(2, 1, 0)  # AF_INET, SOCK_STREAM
if fd < 0:
    print("DENIED errno=%d" % ctypes.get_errno())
else:
    addr = struct.pack("!BBH4s8x", 16, 2, {port}, socket.inet_aton("127.0.0.1"))
    rc = libc.connect(fd, addr, len(addr))
    print("CONNECTED" if rc == 0 else "DENIED errno=%d" % ctypes.get_errno())
"""

RAW_FORK = """
import ctypes, ctypes.util, os
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
pid = libc.fork()
if pid < 0:
    print("DENIED errno=%d" % ctypes.get_errno())
elif pid == 0:
    os._exit(0)
else:
    os.waitpid(pid, 0)
    print("FORKED")
"""

RAW_SPAWN = """
import ctypes, ctypes.util, os
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
argv = (ctypes.c_char_p * 2)(b"/usr/bin/true", None)
pid = ctypes.c_int(0)
rc = libc.posix_spawn(ctypes.byref(pid), b"/usr/bin/true", None, None, argv, None)
if rc != 0:
    print("DENIED errno=%d" % rc)  # posix_spawn returns the error number directly
else:
    os.waitpid(pid.value, 0)
    print("SPAWNED")
"""

RAW_LISTDIR = """
import ctypes, ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.opendir.restype = ctypes.c_void_p
handle = libc.opendir({path!r}.encode())
print("LISTED" if handle else "DENIED errno=%d" % ctypes.get_errno())
"""

ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
PROJECT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "__init__.py")


def denied_errno(output):
    """The errno a probe reported, or None if it didn't report a denial."""
    match = re.search(r"DENIED errno=(\d+)", output or "")
    return int(match.group(1)) if match else None


@pytest.fixture
def listener():
    """A local TCP listener, so the network tests need no internet access and
    can't pass by accident on a machine that simply has no route out."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    yield server.getsockname()[1]
    server.close()


def run_unsandboxed(source):
    """Runs a probe with the same system interpreter but no Seatbelt profile,
    no rlimits and no Python-level guards. Used only as a positive control."""
    with tempfile.TemporaryDirectory() as scratch:
        with open(os.path.join(scratch, "main.py"), "w") as handle:
            handle.write(source)
        completed = subprocess.run(
            [code_runner.SYSTEM_PYTHON, "-I", "-B", "main.py"],
            cwd=scratch, capture_output=True, text=True, timeout=30,
        )
        return completed.stdout


def run_under_profile_only(source):
    """Runs a probe under the real Seatbelt profile from code_runner, but with
    none of the other layers: no rlimits, no Python-level guards, no runner.
    Isolates the profile, so a denial here can only be Seatbelt's."""
    with tempfile.TemporaryDirectory(prefix="grading_") as scratch:
        with open(os.path.join(scratch, "main.py"), "w") as handle:
            handle.write(source)
        completed = subprocess.run(
            [code_runner.SANDBOX_EXEC, "-p", code_runner._build_profile(scratch),
             code_runner.SYSTEM_PYTHON, "-I", "-B", "main.py"],
            cwd=scratch, capture_output=True, text=True, timeout=30,
        )
        return completed.stdout


def first_existing_credential_file():
    """A real file inside one of the extra-denied credential directories, if
    this machine has one. None of them exist on some machines."""
    home = os.path.expanduser("~")
    for name in code_runner.EXTRA_DENIED_READS:
        path = os.path.join(home, name)
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            for entry in sorted(os.listdir(path)):
                candidate = os.path.join(path, entry)
                if os.path.isfile(candidate):
                    return candidate
    return None


class TestProbesCanDetectALeak:
    """Positive controls. Every probe in TestKernelEnforcement is run here
    with the sandbox removed, and must succeed. Without this, a probe that was
    broken -- a typo, a wrong constant, a missing library -- would print
    DENIED for its own reasons and the security tests would pass while
    proving nothing."""

    def test_raw_open_reads_project_files_when_unsandboxed(self):
        output = run_unsandboxed(RAW_OPEN.format(path=PROJECT_FILE))
        assert output.startswith("LEAKED:")
        assert "create_app" in output

    def test_raw_connect_reaches_the_listener_when_unsandboxed(self, listener):
        assert run_unsandboxed(RAW_CONNECT.format(port=listener)).strip() == "CONNECTED"

    def test_raw_fork_succeeds_when_unsandboxed(self):
        assert run_unsandboxed(RAW_FORK).strip() == "FORKED"

    def test_raw_posix_spawn_succeeds_when_unsandboxed(self):
        assert run_unsandboxed(RAW_SPAWN).strip() == "SPAWNED"

    def test_raw_listdir_lists_the_repo_when_unsandboxed(self):
        assert run_unsandboxed(RAW_LISTDIR.format(path=code_runner.PROJECT_ROOT)).strip() == "LISTED"


class TestSeatbeltProfileAlone:
    """The Seatbelt profile, tested in isolation. Through the full runner,
    RLIMIT_NPROC answers fork() first (EAGAIN) because this user already has
    more than 32 processes -- which would leave Seatbelt's own process-fork
    rule untested, and the protection quietly dependent on how busy the
    machine is. Running the profile alone proves each rule holds by itself."""

    def test_profile_refuses_reading_project_files(self):
        assert denied_errno(run_under_profile_only(RAW_OPEN.format(path=PROJECT_FILE))) == errno.EPERM

    def test_profile_refuses_network(self, listener):
        assert denied_errno(run_under_profile_only(RAW_CONNECT.format(port=listener))) == errno.EPERM

    def test_profile_refuses_fork(self):
        assert denied_errno(run_under_profile_only(RAW_FORK)) == errno.EPERM

    def test_profile_refuses_posix_spawn(self):
        assert denied_errno(run_under_profile_only(RAW_SPAWN)) == errno.EPERM

    def test_profile_refuses_listing_the_repo(self):
        """Existence of a guessed name leaks (see below); enumerating what's
        there does not."""
        assert denied_errno(run_under_profile_only(RAW_LISTDIR.format(path=code_runner.PROJECT_ROOT))) == errno.EPERM

    @pytest.mark.skipif(first_existing_credential_file() is None,
                        reason="none of ~/.ssh, ~/.aws, ~/.netrc, ~/.gnupg exists on this machine")
    def test_profile_refuses_credential_files(self):
        output = run_under_profile_only(RAW_OPEN.format(path=first_existing_credential_file()))
        assert denied_errno(output) == errno.EPERM, output

    def test_denial_hides_contents_but_not_existence(self):
        """A documented limitation, pinned so it can't change unnoticed: a
        denied path that exists answers EPERM, one that doesn't answers
        ENOENT. A program can learn whether a guessed filename exists in the
        project; it cannot read it."""
        missing = os.path.join(code_runner.PROJECT_ROOT, "backend", "no_such_file_for_this_test")
        assert denied_errno(run_under_profile_only(RAW_OPEN.format(path=missing))) == errno.ENOENT


class TestKernelEnforcement:
    """The tests the security claim in SANDBOX.md rests on.

    Each program calls libc directly through ctypes, so open(), socket and
    subprocess are never touched and none of the Python-level guards in
    _sandbox_runner.py gets a chance to intervene. If the kernel didn't refuse
    these, they would succeed -- TestProbesCanDetectALeak proves that.

    Every assertion is on EPERM specifically, not just on "denied". That's
    what separates Seatbelt from the other layers: a missing file is ENOENT,
    Unix permissions give EACCES, RLIMIT_NPROC gives EAGAIN, and no route to
    a host is ENETUNREACH. Only the sandbox profile answers EPERM here.
    """

    def test_raw_read_of_project_source_is_refused_by_the_kernel(self):
        result = run(RAW_OPEN.format(path=PROJECT_FILE))
        assert "LEAKED" not in result["actual"]
        assert denied_errno(result["actual"]) == errno.EPERM, result["actual"]

    @pytest.mark.skipif(not os.path.exists(ENV_FILE), reason="no backend/.env on this machine")
    def test_raw_read_of_the_env_file_is_refused_by_the_kernel(self):
        result = run(RAW_OPEN.format(path=ENV_FILE))
        assert "JWT_SECRET" not in result["actual"]
        assert denied_errno(result["actual"]) == errno.EPERM, result["actual"]

    def test_raw_connect_is_refused_by_the_kernel(self, listener):
        result = run(RAW_CONNECT.format(port=listener))
        assert "CONNECTED" not in result["actual"]
        assert denied_errno(result["actual"]) == errno.EPERM, result["actual"]

    def test_raw_fork_is_refused_by_the_kernel(self):
        """EAGAIN is RLIMIT_NPROC answering first, EPERM is Seatbelt; both
        are the kernel. TestSeatbeltProfileAlone proves Seatbelt alone
        refuses it too, so this doesn't rest on the rlimit."""
        result = run(RAW_FORK)
        assert "FORKED" not in result["actual"]
        assert denied_errno(result["actual"]) in (errno.EPERM, errno.EAGAIN), result["actual"]

    def test_raw_posix_spawn_is_refused_by_the_kernel(self):
        """A separate syscall from fork on macOS; denying process-fork was
        found to cover it too, and this keeps that finding from quietly
        becoming untrue. EAGAIN/EPERM as for fork."""
        result = run(RAW_SPAWN)
        assert "SPAWNED" not in result["actual"]
        assert denied_errno(result["actual"]) in (errno.EPERM, errno.EAGAIN), result["actual"]

    def test_the_program_can_still_read_its_own_submission(self):
        """The denial is scoped. The scratch directory holding main.py stays
        readable even through the raw syscall path."""
        result = run(RAW_OPEN.format(path="main.py"))
        assert result["actual"].startswith("LEAKED:")
        assert "ctypes" in result["actual"]  # it read back its own source


class TestFailsLoudlyWithoutTheSandbox:
    """If Seatbelt isn't available, grading must stop rather than quietly
    run student code with only the Python-level guards."""

    def test_non_macos_refuses_to_run(self, monkeypatch):
        monkeypatch.setattr(code_runner.platform, "system", lambda: "Linux")
        with pytest.raises(code_runner.CodeRunnerError, match="Seatbelt"):
            run("print('hi')")

    def test_missing_sandbox_exec_refuses_to_run(self, monkeypatch):
        monkeypatch.setattr(code_runner, "SANDBOX_EXEC", "/nonexistent/sandbox-exec")
        with pytest.raises(code_runner.CodeRunnerError):
            run("print('hi')")

    def test_a_submission_lands_in_grading_failed_not_ungraded_execution(
        self, client, auth, teacher, student, monkeypatch
    ):
        rubric = client.post(
            "/api/rubrics",
            json={"title": "R", "type": "code", "criteria": [
                {"name": "C", "max_points": 1, "test_cases": [{"stdin": "", "expected_output": "hi"}]}]},
            headers=auth(teacher),
        ).get_json()
        monkeypatch.setattr(code_runner, "SANDBOX_EXEC", "/nonexistent/sandbox-exec")

        body = client.post(
            "/api/submissions", json={"rubric_id": rubric["id"], "content": "print('hi')"}, headers=auth(student)
        ).get_json()
        assert body["status"] == "grading_failed"

    def test_every_run_goes_through_sandbox_exec(self, monkeypatch):
        """Guards against a refactor that quietly drops the wrapper: the
        command actually launched must start with sandbox-exec and carry a
        profile that denies the project tree, network and process-fork."""
        launched = []
        real_popen = code_runner.subprocess.Popen

        def spy(command, *args, **kwargs):
            launched.append(command)
            return real_popen(command, *args, **kwargs)

        monkeypatch.setattr(code_runner.subprocess, "Popen", spy)
        run("print('hi')", "", "hi")

        command = launched[0]
        assert command[0] == code_runner.SANDBOX_EXEC
        profile = command[command.index("-p") + 1]
        assert f'(deny file-read* (subpath "{code_runner.PROJECT_ROOT}"))' in profile
        assert "(deny network*)" in profile
        assert "(deny process-fork)" in profile
        assert "(deny file-write*)" in profile
        for name in code_runner.EXTRA_DENIED_READS:
            assert os.path.join(os.path.expanduser("~"), name) in profile


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
