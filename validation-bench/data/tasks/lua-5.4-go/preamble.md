You are an expert Go programmer. Implement the solution described below.
Submit your complete Go source code using the `submit` tool.
You will receive compilation and test results. Fix and resubmit if needed.

## Specification

Implement a Lua 5.4 syntactic validator in Go using only the standard
library — no third-party packages. The build sandbox has no network
access, so `golang.org/x/...`, `gopkg.in/...`, and any external module
will not resolve.

The source file is named `solution.go`. It must declare `package main`
and contain a `func main()` entry point. Compile command: `{compile_cmd}`.

Your validator must read a Lua 5.4 source from stdin and print to
stdout exactly `valid` (e.g. via `fmt.Print("valid")`) if the source
is syntactically valid, or exactly `invalid` otherwise. Surrounding
whitespace is allowed; anything else (debug output, mixed casing,
multiple lines) counts as a test failure. The process must also exit cleanly with status 0 — a correct
verdict followed by a crash, timeout, or non-zero exit is still a
failure.

The reference oracle is `luac5.4 -p file.lua`. A program is "valid" if
and only if `luac5.4 -p` accepts it (exits 0). You are validating static
(parse-time) correctness only — runtime errors, undefined variables,
calls to nonexistent library functions, etc. do not make a program
invalid here.

The relevant sections of the Lua 5.4 reference manual follow.
