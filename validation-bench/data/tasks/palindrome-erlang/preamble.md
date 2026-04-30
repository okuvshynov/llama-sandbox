You are an expert Erlang programmer. Implement the solution described below.
Submit your complete Erlang source code using the `submit` tool.
You will receive compilation and test results. Fix and resubmit if needed.

## Specification

Implement a byte-level palindrome detector in Erlang/OTP using only the
standard library.

The source file is named `solution.erl`. It must declare module `solution`
and export `main/0` (the runner invokes it via `erl -s solution main`,
which calls a zero-arity function — note: arity 0, not 1).

Compile command: `{compile_cmd}` (produces `solution.beam`).
Run command: `{run_cmd}`.

Your `main/0` must read all of stdin as raw bytes — the runner is
launched with `-noinput` so the BEAM I/O subsystem leaves stdin alone,
and `file:read_file("/dev/stdin")` returns `{ok, Binary}` with the full
byte sequence. Then print to stdout exactly `valid` (e.g.
`io:format("valid")`) if the byte sequence is a palindrome (equals its
reverse), or exactly `invalid` otherwise. After printing, call `halt(0)`
to exit the runtime — the BEAM does not auto-halt when `main/0` returns,
so without it the runtime idles until the per-test timeout fires and
the test counts as a failure. Surrounding whitespace is
allowed; anything else (debug output, mixed casing, multiple lines)
counts as a test failure. The process must also exit cleanly with status 0 — a correct
verdict followed by a crash, timeout, or non-zero exit is still a
failure.

The full definition follows.
