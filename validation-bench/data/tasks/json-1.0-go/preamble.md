You are an expert Go programmer. Implement the solution described below.
Submit your complete Go source code using the `submit` tool.
You will receive compilation and test results. Fix and resubmit if needed.

## Specification

Implement a validator for JSON texts in Go using only the standard
library — no third-party packages. The validator may import any package
under `https://pkg.go.dev/std` (`bytes`, `strings`, `unicode/utf8`,
`io`, `os`, `bufio`, etc.) but `golang.org/x/...`, `gopkg.in/...`, and
any external module are not available; the build sandbox has no network
access. **Do not use `encoding/json`** — the goal is to write a
syntactic validator from scratch, not to delegate to the stdlib JSON
parser.

The source file is named `solution.go`. It must declare `package main`
and contain a `func main()` entry point. Compile command: `{compile_cmd}`.

Your validator must read a JSON text from stdin and print to stdout
exactly `valid` (e.g. via `fmt.Print("valid")`) if it is a syntactically
valid JSON document, or exactly `invalid` otherwise. Surrounding
whitespace is allowed; anything else (debug output, mixed casing,
multiple lines) counts as a test failure. The process must also exit
cleanly with status 0 — a correct verdict followed by a crash, timeout,
or non-zero exit is still a failure.

The input is a JSON 1.0 document as specified by RFC 8259 / ECMA-404.
The full specification follows.
