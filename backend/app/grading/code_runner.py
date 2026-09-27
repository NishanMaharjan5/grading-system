"""Runs a student's Python program against stdin/stdout test cases.

The submitted code is written to a throwaway directory and executed as a
separate process; it is never exec'd, eval'd or imported into the server.
Each case gets its own process, so one hanging test cannot affect the next.

Two independent layers now enforce containment, and only one of them is
trustworthy on its own:

* macOS Seatbelt, via `sandbox-exec` (_build_profile below), applied by the
  kernel before the student's code ever runs. This is real: proven in
  tests/test_code_grading.py by bypassing every Python-level guard with a raw
  ctypes syscall and confirming the kernel still refuses it.
* The resource limits and Python-level guards in _sandbox_runner.py, kept as
  defense-in-depth. Cheap to keep, and they catch things Seatbelt doesn't
  (CPU time, memory, output size).

macOS-only. If sandbox-exec is unavailable -- wrong platform, or a future
macOS removes it -- grading fails loudly into grading_failed rather than
silently running student code with no kernel-level containment. See
SANDBOX.md for what porting this to Linux would need (Landlock, seccomp, or
a container).
"""

import os
import platform
import signal
import subprocess
import sys
import tempfile
import threading

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sandbox_runner.py")

# Repo root: .../grading-system, four directories up from this file
# (app/grading/code_runner.py -> app/grading -> app -> backend -> grading-system).
# Everything sensitive -- .env, ml_models/, training_data/, the app's own
# source -- lives under here, so denying reads to this one subtree is what
# actually protects the secrets, rather than trying to allowlist every path a
# Python interpreter needs to read at startup (fragile and version-specific;
# see the note in _build_profile below).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# The system interpreter, not sys.executable: the venv lives under
# PROJECT_ROOT, which the sandbox profile denies reading, so venv python
# cannot even be exec'd once the profile applies. This must exist on any
# reasonably modern macOS (shipped by Xcode Command Line Tools).
SYSTEM_PYTHON = "/usr/bin/python3"
SANDBOX_EXEC = "/usr/bin/sandbox-exec"

# Classic high-value credential locations outside the project. Cheap to deny
# outright; not an attempt at a comprehensive home-directory lockdown, since
# the application's own secrets all live under PROJECT_ROOT regardless.
EXTRA_DENIED_READS = (".ssh", ".aws", ".netrc", ".gnupg")

WALL_CLOCK_SECONDS = 10   # outer bound; the child also has a CPU-time rlimit
CPU_SECONDS = 5
MEMORY_BYTES = 256 * 1024 * 1024
OUTPUT_CHAR_LIMIT = 10_000  # what we keep, not what the program may print


class CodeRunnerError(Exception):
    """The harness itself failed -- not the student's code misbehaving.
    Only this should ever cause a submission to land in grading_failed."""


def _sb_literal(path):
    """Escapes a path for use inside a double-quoted Seatbelt string literal.
    Our own paths (repo location, tempfile.TemporaryDirectory output) never
    contain quotes or backslashes, but escaping defensively costs nothing."""
    return path.replace("\\", "\\\\").replace('"', '\\"')


def _require_sandbox_exec():
    if platform.system() != "Darwin":
        raise CodeRunnerError(
            "Code grading requires macOS Seatbelt (sandbox-exec), which is not available on "
            f"{platform.system()}. See SANDBOX.md for what a Linux port would need."
        )
    if not os.path.exists(SANDBOX_EXEC):
        raise CodeRunnerError(f"{SANDBOX_EXEC} is not present on this system.")
    if not os.path.exists(SYSTEM_PYTHON):
        raise CodeRunnerError(f"{SYSTEM_PYTHON} is not present on this system.")


def _build_profile(source_dir):
    """One Seatbelt profile per submission, with that submission's scratch
    directory baked in as the sole place allowed to be written.

    Reads are denied for PROJECT_ROOT and a short list of credential
    directories, not for everything except source_dir. A true default-deny
    -- allow only source_dir and the interpreter's own files -- was tried and
    rejected: macOS's own /usr/bin/python3 is a dispatcher that re-execs into
    a versioned path under CommandLineTools' Python framework, and enumerating
    every path the interpreter needs to read to start up (dyld cache,
    framework internals, locale/encoding data) is exactly the kind of
    Apple-internal, version-specific fragility this project has avoided
    elsewhere. Since every actual secret lives under PROJECT_ROOT, denying
    that subtree achieves the real goal without depending on Apple's
    toolchain internals staying still.

    process-fork is denied outright: on this platform it also blocks
    posix_spawn (confirmed: a raw ctypes posix_spawn() call is refused the
    same as a raw fork()), which covers subprocess/os.system/os.fork at the
    kernel level without needing a process-exec allowlist -- which would hit
    the same dispatcher-chain fragility as the read side.
    """
    home = os.path.expanduser("~")
    extra_denies = "".join(
        f'(deny file-read* (subpath "{_sb_literal(os.path.join(home, name))}"))\n'
        for name in EXTRA_DENIED_READS
    )
    return f"""
(version 1)
(allow default)
(deny file-read* (subpath "{_sb_literal(PROJECT_ROOT)}"))
(allow file-read* (literal "{_sb_literal(RUNNER)}"))
{extra_denies}
(deny file-write*)
(allow file-write* (subpath "{_sb_literal(source_dir)}"))
(deny process-fork)
(deny network*)
"""


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
    standing between a runaway allocation and the machine's RAM. Seatbelt has
    no memory-capping primitive either; this gap is orthogonal to the
    file/network/process containment sandbox-exec adds.
    """
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return int(out) * 1024 if out.isdigit() else None


def run_one(source_dir, stdin_text):
    """Runs the program once, under sandbox-exec. Returns a dict describing
    what happened -- a crash, a timeout or a memory breach is a normal result
    here, not an exception."""
    _require_sandbox_exec()

    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": source_dir,
        "TMPDIR": source_dir,
        "SANDBOX_CPU_SECONDS": str(CPU_SECONDS),
        "SANDBOX_MEMORY_BYTES": str(MEMORY_BYTES),
        # No proxy variables, no PYTHONPATH, nothing inherited from the server.
        **({"SANDBOX_DEBUG": "1"} if os.environ.get("SANDBOX_DEBUG") else {}),
    }

    command = [
        SANDBOX_EXEC, "-p", _build_profile(source_dir),
        SYSTEM_PYTHON, "-I", "-B", RUNNER, "main.py",
    ]

    try:
        process = subprocess.Popen(
            command,
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
