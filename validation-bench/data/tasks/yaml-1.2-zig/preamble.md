You are an expert Zig programmer. Implement the solution described below.
Submit your complete Zig source code using the `submit` tool.
You will receive compilation and test results. Fix and resubmit if needed.

## Specification

Implement a YAML 1.2 syntactic validator in Zig using only the standard
library — no external packages. The source file is named `solution.zig`
and must compile as a standalone executable with a `pub fn main()` entry
point.
Compile command: `{compile_cmd}`

Your validator must read a YAML 1.2 stream from stdin and print to
stdout exactly `valid` if the stream is syntactically valid, or exactly
`invalid` otherwise. Surrounding whitespace is allowed; anything else
(debug output, mixed casing, multiple lines) counts as a test failure.
The process must also exit cleanly with status 0 — a correct verdict
followed by a crash, timeout, or non-zero exit is still a failure.
Validate static (parse-time) correctness only — the validator does not
need to construct or emit the parsed representation.

The reference oracle is the YAML 1.2 specification proper (revision
1.2.2, embedded below). A program is "valid" iff it parses successfully
under the spec's production rules.

The full specification follows.
