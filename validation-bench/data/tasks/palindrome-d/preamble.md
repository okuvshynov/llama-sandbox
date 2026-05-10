You are an expert D programmer. Implement the solution described below.
Submit your complete D source code using the `submit` tool.
You will receive compilation and test results. Fix and resubmit if needed.

## Specification

Implement a byte-level palindrome detector in D using only the standard
library (Phobos) — no external packages. The source file is named
`solution.d` and must compile as a standalone executable with a
`void main()` (or `int main()`) entry point.
Compile command: `{compile_cmd}`

Your program must read all of stdin as raw bytes and print to stdout
exactly `valid` if the byte sequence is a palindrome (equals its
reverse), or exactly `invalid` otherwise. Surrounding whitespace is
allowed; anything else (debug output, mixed casing, multiple lines)
counts as a test failure. The process must also exit cleanly with
status 0 — a correct verdict followed by a crash, timeout, or non-zero
exit is still a failure.

The full definition follows.
