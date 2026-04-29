You are an expert Lua programmer. Implement the solution described below.
Submit your complete Lua source code using the `submit` tool.
You will receive syntax-check and test results. Fix and resubmit if needed.

## Specification

Implement a byte-level palindrome detector in Lua 5.4 using only the standard
library.
Run command: `{run_cmd}` (syntax-checked first via `{prepare_cmd}`).

Your program must read all of stdin as raw bytes.
If the byte sequence is a palindrome (equals its reverse), exit with zero
exit code (`os.exit(0)`).
Otherwise, exit with non-zero exit code (`os.exit(1)`).

The full definition follows.
