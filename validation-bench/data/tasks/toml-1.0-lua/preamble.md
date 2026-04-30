You are an expert Lua programmer. Implement the solution described below.
Submit your complete Lua source code using the `submit` tool.
You will receive syntax-check and test results. Fix and resubmit if needed.

## Specification

Implement a validator for TOML files in Lua 5.4 using only the standard library.
Run command: `{run_cmd}` (syntax-checked first via `{prepare_cmd}`).

Your validator must read a TOML file from stdin and print to stdout
exactly `valid` (e.g. `io.write("valid")`) if it is a valid TOML
document, or exactly `invalid` otherwise. Surrounding whitespace is
allowed; anything else (debug output, mixed casing, multiple lines)
counts as a test failure. The process must also exit cleanly with status 0 — a correct
verdict followed by a crash, timeout, or non-zero exit is still a
failure.

The input is a TOML v1.0.0 document. The full specification follows.