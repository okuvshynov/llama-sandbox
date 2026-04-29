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

Your validator must read a TOML file from stdin. If it is valid, exit
with zero exit code; if invalid, exit with non-zero exit code (e.g.
`os.Exit(1)`). Do not print anything; only the exit code is checked.

The input is a TOML v1.0.0 document. The full specification follows.
