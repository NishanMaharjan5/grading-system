# Code submission sandbox — what it does and does not contain

Student Python is written to a throwaway directory and run as a separate
process. `app/grading/code_runner.py` launches it under **macOS Seatbelt**
(`sandbox-exec`), and inside that, `app/grading/_sandbox_runner.py` applies
resource limits and Python-level guards before handing over to the student's
code. It is never `exec`'d, `eval`'d or imported into the Flask process.

Everything below was measured on this machine, not assumed. Each claim is
pinned by a test in `tests/test_code_grading.py`, named where it matters.

**This is still not a container.** It is kernel-enforced for the things that
matter most (reading secrets, the network, spawning processes), but it is one
macOS-specific mechanism rather than full isolation. See the limits at the end.

## Layer 1 — Seatbelt (`sandbox-exec`): kernel-enforced, the real boundary

`code_runner._build_profile()` generates one profile per submission:

| Rule | Effect |
|---|---|
| `(deny file-read* (subpath <repo root>))` | Nothing in the project is readable: `backend/.env` (`JWT_SECRET`, database credentials), the app's source, `ml_models/`, `training_data/`, the venv. |
| `(allow file-read* (literal <_sandbox_runner.py>))` | The one exception — the harness itself has to be readable to run. |
| `(deny file-read* …)` for `~/.ssh`, `~/.aws`, `~/.netrc`, `~/.gnupg` | Classic credential stores on the same machine. |
| `(deny file-write*)` then `(allow file-write* (subpath <scratch dir>))` | Nothing may be written anywhere except the submission's own scratch directory. |
| `(deny process-fork)` | No child processes. On macOS this also refuses `posix_spawn`, so `subprocess`, `os.system` and raw `fork()` are all covered. |
| `(deny network*)` | No network at all, including loopback. |

**How this was verified — and why the tests are built the way they are.**
Every probe calls libc directly through `ctypes`, so `open()`, `socket` and
`subprocess` are never touched and the Python-level guards in layer 3 never get
a chance to intervene. Any denial is therefore the kernel's. Three things make
the result trustworthy rather than merely green:

- **Positive controls** (`TestProbesCanDetectALeak`): the same probes, run with
  no sandbox, must actually read the file, connect, fork and spawn. A probe
  broken for its own reasons would otherwise print "denied" and the security
  tests would pass while proving nothing.
- **Assertions on `EPERM` specifically**, not just "denied". A missing file is
  `ENOENT`, Unix permissions give `EACCES`, `RLIMIT_NPROC` gives `EAGAIN`, no
  route gives `ENETUNREACH`. Only the Seatbelt profile answers `EPERM`. The
  network probe connects to a local listener the test controls, so it can't
  pass by accident on a machine that has no network.
- **The profile tested alone** (`TestSeatbeltProfileAlone`), without the
  rlimits or guards. Through the full runner, `RLIMIT_NPROC` answers `fork()`
  first (`EAGAIN`) on this machine, which would otherwise leave Seatbelt's own
  fork rule unproven and the protection quietly dependent on how many
  processes the user happens to be running.

The tests were also checked for teeth: replacing the profile with one that
allows everything makes all eight tests that depend on a Seatbelt rule fail.

**If Seatbelt is unavailable, grading stops.** On anything other than macOS,
or if `/usr/bin/sandbox-exec` or `/usr/bin/python3` is missing, the runner
raises and the submission lands in `grading_failed` for a teacher to grade by
hand. It never falls back to running student code under the weaker layers
alone (`TestFailsLoudlyWithoutTheSandbox`). A structural test also checks that
every run is launched through `sandbox-exec` with each deny rule present, so a
refactor can't drop the wrapper unnoticed.

### Read denial is a denylist, deliberately

Reads are denied for the repo and the credential directories — not "everything
except the scratch directory". A true default-deny was tried and rejected.
macOS's `/usr/bin/python3` is a dispatcher that re-execs into a versioned path
inside the Command Line Tools' Python framework. To start the interpreter at
all, the profile would have to allowlist every path it reads along the way
(dyld cache, framework internals, locale and encoding data). That list depends
on Apple's toolchain internals and would break on an Xcode or macOS update,
silently disabling all code grading. Every actual secret lives under the repo,
so denying the repo achieves the goal without that dependency.

The same fragility is why process spawning is blocked with `(deny
process-fork)` rather than a `process-exec` allowlist — the allowlist would have
to track that same dispatcher chain.

## Layer 2 — resource limits: kernel-enforced, but not about access

| Limit | Value | Enforced by |
|---|---|---|
| CPU time (`RLIMIT_CPU`) | 5s | kernel — an infinite loop dies at ~5s |
| Wall clock | 10s | parent kills the process group |
| File size (`RLIMIT_FSIZE`) | 0 | kernel — no regular file may grow |
| Core dumps (`RLIMIT_CORE`) | 0 | kernel |
| Processes (`RLIMIT_NPROC`) | 32 | kernel — per user, so on a busy machine this alone blocks every fork |
| Memory | 256MB | **parent**, polling RSS every 100ms (see below) |
| Output kept | 10,000 chars | parent |

`setrlimit` can only lower a limit, so student code cannot raise them back.

**Memory is the weak point.** macOS refuses `RLIMIT_AS`, `RLIMIT_DATA` and
`RLIMIT_RSS` outright — `setrlimit` raises even when *lowering* them — and
Seatbelt has no memory primitive either. The parent polls the child's RSS and
kills the group on breach, which leaves a **~100ms window** in which a fast
allocator can exceed the cap. On Linux, `RLIMIT_AS` would work.

## Layer 3 — Python-level guards: defense-in-depth only

`_sandbox_runner.py` still restricts `open()`, stubs `socket`, and replaces
`os.system`, `os.exec*`, `os.fork` and `subprocess`. These live in the same
interpreter as the student's code, so `ctypes` or `importlib.reload` gets
around them — the tests in layer 1 do exactly that. They stay because they're
cheap and give a clearer error message than a bare kernel denial when they do
catch something first.

## Behaviour changes from the previous version

- **Submissions run under the system Python** (`/usr/bin/python3`, 3.9.6 on this
  machine), not the app's venv (3.11). The venv lives inside the repo, which
  the profile makes unreadable, so it can't be exec'd. Student code therefore
  sees only the standard library of the system interpreter — not numpy,
  pandas, torch or anything else the server has installed. That is arguably a
  fix (student code shouldn't share the server's environment), but it means
  **code rubrics should be written against the standard library, and against
  Python 3.9 syntax.**

## Known limitations

- **macOS only, and `sandbox-exec` is deprecated.** Apple has discouraged it
  for years; it still works on this macOS, which the tests confirm on every
  run. A future macOS could remove it; the fail-closed check above means
  grading would stop rather than degrade.
- **File existence leaks, contents don't.** A denied path answers `EPERM` if
  it exists and `ENOENT` if it doesn't, so a program can confirm whether a
  guessed filename exists in the repo. It cannot read it, and it cannot list
  directories to find names to guess. Pinned by
  `test_denial_hides_contents_but_not_existence` and
  `test_profile_refuses_listing_the_repo`.
- **The credential-directory denies are untested on this machine**, because
  none of `~/.ssh`, `~/.aws`, `~/.netrc` or `~/.gnupg` exists here. The test
  skips and says so; it will run anywhere those exist.
- **Same OS user as the server.** Seatbelt closes the file and network gaps
  that made this dangerous, but anything the profile doesn't mention is still
  available with the server user's permissions. Default-allow means the
  profile is only as good as its deny list.
- **Memory**, as above.

## Porting to Linux

`sandbox-exec` has no Linux equivalent under the same name. The options, from
closest to most complete:

1. **Landlock** (kernel 5.13+) — unprivileged, path-based file access control;
   the closest match to the file rules here. Network rules need 6.7+.
2. **seccomp-bpf** — syscall filtering; the natural fit for denying `fork`,
   `execve` and `socket` outright, but not path-aware, so it pairs with
   Landlock rather than replacing it.
3. **A container** (`docker run --network=none --read-only --memory=256m
   --pids-limit=64`) — the standard answer, covering all of the above and
   memory too. It was ruled out for this project.

Until one of those exists, the runner refuses to grade on Linux rather than run
unsandboxed.
