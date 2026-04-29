You are an expert Erlang programmer. Implement the solution described below.
Submit your complete Erlang source code using the `submit` tool.
You will receive compilation and test results. Fix and resubmit if needed.

## Specification

Implement a Lua 5.4 syntactic validator in Erlang/OTP using only the
standard library (OTP applications such as `stdlib` are fine; no third-
party packages).

The source file is named `solution.erl`. It must declare module `solution`
and export `main/0` (the runner invokes it via `erl -s solution main`,
which calls a zero-arity function — note: arity 0, not 1).

Compile command: `{compile_cmd}` (produces `solution.beam`).
Run command: `{run_cmd}`.

Your `main/0` must read the Lua 5.4 source from stdin as raw bytes. The
runner is launched with `-noinput` so the BEAM I/O subsystem leaves stdin
alone, and `file:read_file("/dev/stdin")` returns `{ok, Binary}` with the
full byte sequence. Then exit the runtime via `halt/1`: `halt(0)` if the
source is syntactically valid, `halt(1)` otherwise. Do not print anything;
only the exit code is checked.

The reference oracle is `luac5.4 -p file.lua`. A program is "valid" if
and only if `luac5.4 -p` accepts it (exits 0). You are validating static
(parse-time) correctness only — runtime errors, undefined variables,
calls to nonexistent library functions, etc. do not make a program
invalid here.

The relevant sections of the Lua 5.4 reference manual follow.
