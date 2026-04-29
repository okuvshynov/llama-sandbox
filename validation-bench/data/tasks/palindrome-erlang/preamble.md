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
byte sequence. Then exit the runtime via `halt/1`: `halt(0)` if the byte
sequence is a palindrome (equals its reverse), `halt(1)` otherwise. Do
not print anything; only the exit code is checked.

The full definition follows.
