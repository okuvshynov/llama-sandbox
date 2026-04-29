You are an expert Go programmer. Implement the solution described below.
Submit your complete Go source code using the `submit` tool.
You will receive compilation and test results. Fix and resubmit if needed.

## Specification

Implement a byte-level palindrome detector in Go using only the standard
library.

The source file is named `solution.go`. It must declare `package main`
and contain a `func main()` entry point. Compile command: `{compile_cmd}`.

Your program must read all of stdin as raw bytes (e.g. via
`io.ReadAll(os.Stdin)`). Exit with status 0 if the byte sequence is a
palindrome (equals its reverse); exit non-zero otherwise (e.g.
`os.Exit(1)`). Do not print anything; only the exit code is checked.

The full definition follows.
