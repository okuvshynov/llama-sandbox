You are an expert Lua programmer. Implement the solution described below.
Submit your complete Lua source code using the `submit` tool.
You will receive syntax-check and test results. Fix and resubmit if needed.

## Specification

Implement a validator for TOML files in Lua 5.4 using only the standard library.
Run command: `{run_cmd}` (syntax-checked first via `{prepare_cmd}`).

Your validator must read a TOML file from stdin.
If it is valid, exit with zero exit code (`os.exit(0)`).
If it is invalid, exit with non-zero exit code (`os.exit(1)`).

The input is a TOML v1.0.0 document. The full specification follows.