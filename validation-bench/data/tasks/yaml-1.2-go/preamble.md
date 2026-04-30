You are an expert Go programmer. Implement the solution described below.
Submit your complete Go source code using the `submit` tool.
You will receive compilation and test results. Fix and resubmit if needed.

## Specification

Implement a YAML 1.2 syntactic validator in Go using only the standard
library — no third-party packages. The validator may import any package
under `https://pkg.go.dev/std` (`bytes`, `strings`, `regexp`,
`unicode/utf8`, `io`, `os`, `bufio`, etc.) but `golang.org/x/...`,
`gopkg.in/...`, and any external module are not available; the build
sandbox has no network access.

The source file is named `solution.go`. It must declare `package main`
and contain a `func main()` entry point. Compile command: `{compile_cmd}`.

Your validator must read a YAML 1.2 stream from stdin and print to
stdout exactly `valid` (e.g. via `fmt.Print("valid")`) if the stream
is syntactically valid, or exactly `invalid` otherwise. Surrounding
whitespace is allowed; anything else (debug output, mixed casing,
multiple lines) counts as a test failure. Exit code is not checked;
only the printed verdict. Validate static (parse-time) correctness only
— the validator does not need to construct or emit the parsed
representation.

The reference oracle is the YAML 1.2 specification proper (revision
1.2.2, embedded below). A program is "valid" iff it parses successfully
under the spec's production rules.

The full specification follows.
