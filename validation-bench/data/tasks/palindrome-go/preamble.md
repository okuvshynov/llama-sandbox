You are an expert Go programmer. Implement the solution described below.
Submit your complete Go source code using the `submit` tool.
You will receive compilation and test results. Fix and resubmit if needed.

## Specification

Implement a byte-level palindrome detector in Go using only the standard
library.

The source file is named `solution.go`. It must declare `package main`
and contain a `func main()` entry point. Compile command: `{compile_cmd}`.

Your program must read all of stdin as raw bytes (e.g. via
`io.ReadAll(os.Stdin)`) and print to stdout exactly `valid` (e.g. via
`fmt.Print("valid")`) if the byte sequence is a palindrome (equals its
reverse), or exactly `invalid` otherwise. Surrounding whitespace is
allowed; anything else (debug output, mixed casing, multiple lines)
counts as a test failure. Exit code is not checked; only the printed
verdict.

The full definition follows.
