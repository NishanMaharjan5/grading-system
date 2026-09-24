"""Runs a student's Python program against stdin/stdout test cases.

The submitted code is written to a throwaway directory and executed as a
separate process; it is never exec'd, eval'd or imported into the server.
Each case gets its own process, so one hanging test cannot affect the next.

What actually contains the code is described in SANDBOX.md. In short: the
resource limits are real, the Python-level guards are not a security boundary,
and this is not equivalent to a container.
"""

import os
import signal
import subprocess
import sys
import tempfile
import threading

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sandbox_runner.py")

WALL_CLOCK_SECONDS = 10   # outer bound; the child also has a CPU-time rlimit
CPU_SECONDS = 5
MEMORY_BYTES = 256 * 1024 * 1024
OUTPUT_CHAR_LIMIT = 10_000  # what we keep, not what the program may print


class CodeRunnerError(Exception):
    """The harness itself failed -- not the student's code misbehaving.
    Only this should ever cause a submission to land in grading_failed."""


def normalise_output(text):
    """Trailing whitespace on each line, and wholly blank lines at either end,
    are ignored -- so a missing final newline doesn't fail an otherwise correct
    answer. Everything else must match exactly: leading indentation on every
    line (including the first), interior blank lines, case, and spacing within
    a line.

    Stripping the whole string instead would discard the first line's
    indentation while keeping every other line's, which is a rule nobody could
    predict.
    """
    if text is None:
        return ""

    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _truncate(text, limit=OUTPUT_CHAR_LIMIT):
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (truncated, {len(text)} characters total)"


def _kill_group(pid):
    """Kill the whole process group, so nothing the program spawned survives."""
    for target in (lambda: os.killpg(os.getpgid(pid), signal.SIGKILL), lambda: os.kill(pid, signal.SIGKILL)):
        try:
            target()
            return
        except (ProcessLookupError, PermissionError, OSError):
            continue


def _resident_bytes(pid):
    """Resident size of the process, or None if it has already gone.

    Darwin refuses to set RLIMIT_AS/DATA/RSS at all -- setrlimit raises
    "current limit exceeds maximum limit" even when lowering -- so on macOS
    the kernel will not cap memory for us and this poll is the only thing
    standing between a runaway allocation and the machine's RAM.
    """
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return int(out) * 1024 if out.isdigit() else None


def run_one(source_dir, stdin_text):
    """Runs the program once. Returns a dict describing what happened -- a
    crash, a timeout or a memory breach is a normal result here, not an
    exception."""
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": source_dir,
        "TMPDIR": source_dir,
        "SANDBOX_CPU_SECONDS": str(CPU_SECONDS),
        "SANDBOX_MEMORY_BYTES": str(MEMORY_BYTES),
        # No proxy variables, no PYTHONPATH, nothing inherited from the server.
        **({"SANDBOX_DEBUG": "1"} if os.environ.get("SANDBOX_DEBUG") else {}),
    }

    try:
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", RUNNER, "main.py"],
            cwd=source_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,  # own process group, so a kill takes children too
        )
    except (OSError, ValueError) as cause:
        raise CodeRunnerError(f"Could not start the sandbox: {cause}") from cause

    over_memory = threading.Event()
    finished = threading.Event()

    def watch_memory():
        while not finished.wait(0.1):
            used = _resident_bytes(process.pid)
            if used is not None and used > MEMORY_BYTES:
                over_memory.set()
                _kill_group(process.pid)
                return

    watcher = threading.Thread(target=watch_memory, daemon=True)
    watcher.start()

    timed_out = False
    try:
        stdout, stderr = process.communicate(input=stdin_text or "", timeout=WALL_CLOCK_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(process.pid)
        stdout, stderr = process.communicate()
    finally:
        finished.set()

    if over_memory.is_set():
        outcome = "memory"
    elif timed_out:
        outcome = "timeout"
    elif process.returncode == 0:
        outcome = "ok"
    else:
        outcome = "error"

    return {
        "outcome": outcome,
        "stdout": _truncate(stdout),
        "stderr": _truncate(stderr),
        "exit_code": process.returncode,
    }


def run_test_cases(source, cases):
    """Runs `source` against each case. Returns a list of result dicts in the
    same order, each with passed / outcome / expected / actual / detail.

    Raises CodeRunnerError only if the harness breaks. A program that crashes,
    hangs or prints the wrong thing simply fails its cases, which is a valid
    grade rather than a grading failure.
    """
    if not cases:
        return []

    try:
        with tempfile.TemporaryDirectory(prefix="grading_") as source_dir:
            path = os.path.join(source_dir, "main.py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(source)
            os.chmod(path, 0o444)

            results = []
            for case in cases:
                run = run_one(source_dir, case.stdin)
                expected = normalise_output(case.expected_output)
                actual = normalise_output(run["stdout"])

                if run["outcome"] == "timeout":
                    detail = f"Took longer than {WALL_CLOCK_SECONDS} seconds and was stopped."
                elif run["outcome"] == "memory":
                    detail = f"Used more than {MEMORY_BYTES // (1024 * 1024)}MB of memory and was stopped."
                elif run["outcome"] == "error":
                    last_line = (run["stderr"].strip().splitlines() or ["exited with an error"])[-1]
                    detail = f"The program stopped with an error: {last_line}"
                elif actual != expected:
                    detail = "Output did not match."
                else:
                    detail = "Passed."

                results.append({
                    "case_id": case.id,
                    "passed": run["outcome"] == "ok" and actual == expected,
                    "outcome": run["outcome"],
                    "expected": expected,
                    "actual": actual,
                    "stderr": run["stderr"],
                    "detail": detail,
                })
            return results
    except CodeRunnerError:
        raise
    except OSError as cause:
        raise CodeRunnerError(f"Could not prepare the sandbox: {cause}") from cause
