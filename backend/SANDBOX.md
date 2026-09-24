# Code submission sandbox — what it does and does not contain

Student Python is written to a throwaway directory and run as a separate
process (`app/grading/code_runner.py` launches `app/grading/_sandbox_runner.py`).
It is never `exec`'d, `eval`'d or imported into the Flask process.

**This is not a sandbox in the sense a container or a seccomp jail is.** It
raises the cost of misbehaving; it does not make it impossible. Everything
below was measured on this machine, not assumed — `tests/test_code_grading.py`
asserts each claim.

## Enforced by the kernel — a program cannot undo these

| Limit | Value | Verified |
|---|---|---|
| CPU time (`RLIMIT_CPU`) | 5s | Yes — an infinite loop dies at ~5s |
| Wall clock (parent kills the process group) | 10s | Yes |
| File writes (`RLIMIT_FSIZE` = 0) | none, any size | Yes — writes fail, no file appears |
| Core dumps (`RLIMIT_CORE`) | 0 | Yes |
| New processes (`RLIMIT_NPROC`) | 32 | Yes — `fork()` fails |
| Output captured per run | 10,000 chars kept | Yes |

`setrlimit` can only lower a limit, so student code cannot raise them back.

## Enforced by the parent process

**Memory, 256MB.** `RLIMIT_AS`, `RLIMIT_DATA` and `RLIMIT_RSS` cannot be set at
all on macOS — `setrlimit` raises `ValueError: current limit exceeds maximum
limit` even when *lowering* them. The kernel will not cap memory here, so the
parent polls the child's RSS every 100ms and kills the process group when it
goes over. Consequences worth knowing:

- There is a **~100ms window** in which a program can allocate beyond the cap.
  A fast allocator can briefly exceed 256MB before being killed.
- On Linux the `RLIMIT_AS` path would work and this poll would be redundant.

## Best-effort only — a determined program gets past these

`_sandbox_runner.py` monkeypatches the interpreter before handing over to the
student's code: `open()` is restricted to the working directory and the
interpreter's own files, `socket` is stubbed out, and `os.system`, `os.exec*`,
`os.fork` and `subprocess` are replaced with functions that raise.

These guards run **in the same interpreter as the student's code**, so they are
a speed bump, not a boundary. They stop careless or casually curious code.
They do not stop anyone who tries:

- `ctypes` to call libc directly, bypassing every Python-level patch
- `importlib.reload(socket)` to restore the real module
- reading `/proc`-style interfaces or raw syscalls
- anything at all in a language other than Python (not currently possible —
  only Python submissions are accepted)

## What is *not* contained at all

- **The filesystem is readable** wherever the guard is bypassed. The process
  runs as the same OS user as the server, so it has that user's permissions.
  Nothing but the Python-level `open()` guard stops a determined program from
  reading `backend/.env` — which holds `JWT_SECRET` and the database
  credentials.
- **The network is reachable** if the socket guard is bypassed.
- **No syscall filtering, no namespace isolation, no separate user.**

## If this needs to be safe rather than merely careful

In rough order of value for effort:

1. **Run the grader as a separate unprivileged OS user** with no read access to
   the project directory. Removes the `.env` exposure, which is the worst of
   the above, and needs no new infrastructure.
2. **Container per submission** (`docker run --network=none --read-only
   --memory=256m --pids-limit=64`). This is the standard answer and it makes
   most of this document unnecessary. It was ruled out for this project, and
   that ruling is what everything above is working around.
3. **`sandbox-exec`** on macOS (deprecated but functional) or **seccomp-bpf**
   on Linux for real syscall filtering.

## Demo-day judgement

For a demo where the submissions are the operator's own, this is fine. For real
students submitting real code on a machine holding real credentials, item 1 is
the minimum I would want in place first.
