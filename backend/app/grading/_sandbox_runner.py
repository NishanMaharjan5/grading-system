"""Runs one student program under whatever restrictions this process can impose
on itself, then hands control to the student's code.

Launched as:  python -I -B _sandbox_runner.py <student_file>

This process is itself launched by code_runner.py under a macOS Seatbelt
profile (sandbox-exec), which is the layer that actually holds against a
determined program -- it is kernel-enforced and cannot be bypassed from inside
this interpreter, ctypes included. Everything in this file is a second,
weaker layer kept for defense-in-depth and for the things Seatbelt does not
cover:

* Kernel-enforced resource limits (RLIMIT_*), set below. These genuinely
  hold on their own terms -- student code cannot lift them, because setrlimit
  can only lower a hard limit -- but they bound CPU/output/process-count, not
  file or network access, which is Seatbelt's job now.
* Python-level guards on sockets, file reads and process spawning. On their
  own these are a speed bump, not a wall: they live in the same interpreter
  as the student's code, so ctypes, importlib.reload or a direct syscall gets
  around them. Kept because they're cheap and they give a clearer error
  message than a raw kernel denial when they do catch something.

See SANDBOX.md for the full picture, including how the Seatbelt layer was
verified (a raw ctypes syscall bypass of everything in this file).
"""

import builtins
import os
import runpy
import sys
import traceback

CPU_SECONDS = int(os.environ.get("SANDBOX_CPU_SECONDS", "5"))
ADDRESS_SPACE_BYTES = int(os.environ.get("SANDBOX_MEMORY_BYTES", str(256 * 1024 * 1024)))
MAX_PROCESSES = 32


def apply_resource_limits():
    """Kernel-enforced, and irreversible downward. Returns {name: applied?} so
    the caller can see what actually took effect rather than assuming.

    Memory is the notable gap: Darwin refuses RLIMIT_AS, RLIMIT_DATA and
    RLIMIT_RSS outright -- setrlimit raises even when lowering them -- so on
    macOS nothing here caps memory and the parent process polls RSS instead.
    """
    applied = {}
    try:
        import resource
    except ImportError:  # not a Unix host
        return applied

    def limit(name, which, value):
        if which is None:
            applied[name] = False
            return
        try:
            soft, hard = resource.getrlimit(which)
            ceiling = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(which, (ceiling, ceiling))
            applied[name] = resource.getrlimit(which)[0] == ceiling
        except (ValueError, OSError, AttributeError):
            applied[name] = False

    limit("cpu", resource.RLIMIT_CPU, CPU_SECONDS)
    limit("address_space", getattr(resource, "RLIMIT_AS", None), ADDRESS_SPACE_BYTES)
    limit("file_writes", resource.RLIMIT_FSIZE, 0)   # no file may be written, at any size
    limit("core_dumps", resource.RLIMIT_CORE, 0)
    limit("processes", getattr(resource, "RLIMIT_NPROC", None), MAX_PROCESSES)

    if os.environ.get("SANDBOX_DEBUG"):
        print(f"[sandbox] limits applied: {applied}", file=sys.stderr)
    return applied


def _readable_roots():
    """Where reads stay allowed: the throwaway working directory, and the
    interpreter's own files so `import json` still works."""
    roots = [os.getcwd(), sys.base_prefix, sys.prefix]
    roots += [p for p in sys.path if p and os.path.isdir(p)]
    return tuple(os.path.realpath(r) for r in roots if r)


def install_python_guards():
    """Best-effort only. Documented as such -- see the module docstring."""
    allowed_roots = _readable_roots()
    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            raise PermissionError("Writing files is not permitted in this sandbox.")
        try:
            resolved = os.path.realpath(file)
        except TypeError:
            return real_open(file, mode, *args, **kwargs)  # fd or path-like we can't resolve
        if not resolved.startswith(allowed_roots):
            raise PermissionError(f"Reading {file!r} is not permitted in this sandbox.")
        return real_open(file, mode, *args, **kwargs)

    builtins.open = guarded_open

    def refuse(*_args, **_kwargs):
        raise PermissionError("Network access is not permitted in this sandbox.")

    try:
        import socket

        socket.socket = refuse
        socket.create_connection = refuse
        socket.create_server = refuse
        socket.getaddrinfo = refuse
    except ImportError:
        pass

    def refuse_process(*_args, **_kwargs):
        raise PermissionError("Starting other programs is not permitted in this sandbox.")

    os.system = refuse_process
    for name in ("popen", "execv", "execve", "execvp", "spawnv", "spawnve", "fork", "forkpty"):
        if hasattr(os, name):
            setattr(os, name, refuse_process)
    try:
        import subprocess

        subprocess.Popen = refuse_process
        subprocess.run = refuse_process
        subprocess.call = refuse_process
        subprocess.check_output = refuse_process
    except ImportError:
        pass


def main():
    if len(sys.argv) < 2:
        print("usage: _sandbox_runner.py <student_file>", file=sys.stderr)
        return 2

    target = sys.argv[1]
    apply_resource_limits()
    install_python_guards()

    try:
        runpy.run_path(target, run_name="__main__")
    except SystemExit as exit_request:
        return exit_request.code if isinstance(exit_request.code, int) else 0
    except BaseException:  # noqa: BLE001 -- the student's error is the result here
        # Trim our own frames so the student sees their traceback, not ours.
        exc_type, exc_value, exc_tb = sys.exc_info()
        frames = traceback.extract_tb(exc_tb)
        student_frames = [f for f in frames if os.path.realpath(f.filename) == os.path.realpath(target)]
        sys.stderr.write("Traceback (most recent call last):\n")
        sys.stderr.write("".join(traceback.format_list(student_frames or frames)))
        sys.stderr.write("".join(traceback.format_exception_only(exc_type, exc_value)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
