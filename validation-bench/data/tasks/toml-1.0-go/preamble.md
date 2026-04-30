You are an expert Go programmer. Implement the solution described below.
Submit your complete Go source code using the `submit` tool.
You will receive compilation and test results. Fix and resubmit if needed.

## Specification

Implement a validator for TOML files in Go using only the standard
library — no third-party packages. The validator may import any package
under `https://pkg.go.dev/std` (`bytes`, `strings`, `regexp`,
`unicode/utf8`, `io`, `os`, `bufio`, etc.) but `golang.org/x/...`,
`gopkg.in/...`, and any external module are not available; the build
sandbox has no network access.

The source file is named `solution.go`. It must declare `package main`
and contain a `func main()` entry point. Compile command: `{compile_cmd}`.

Your validator must read a TOML file from stdin and print to stdout
exactly `valid` (e.g. via `fmt.Print("valid")`) if it is a valid TOML
document, or exactly `invalid` otherwise. Surrounding whitespace is
allowed; anything else (debug output, mixed casing, multiple lines)
counts as a test failure. The process must also exit cleanly with status 0 — a correct
verdict followed by a crash, timeout, or non-zero exit is still a
failure.

The input is a TOML v1.0.0 document. The full specification follows.
