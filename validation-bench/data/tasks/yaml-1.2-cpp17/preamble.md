You are an expert C++ programmer. Implement the solution described below.
Submit your complete C++ source code using the `submit` tool.
You will receive compilation and test results. Fix and resubmit if needed.

## Specification

Implement a YAML 1.2 syntactic validator in C++17 using only the standard library.
Compiler command: `{compile_cmd}`

Your validator must read a YAML 1.2 stream from stdin and print to
stdout exactly `valid` (e.g. `std::cout << "valid"`) if the stream is
syntactically valid, or exactly `invalid` otherwise. Surrounding
whitespace is allowed; anything else (debug output, mixed casing,
multiple lines) counts as a test failure. The process must also exit cleanly with status 0 — a correct
verdict followed by a crash, timeout, or non-zero exit is still a
failure.

The reference oracle is the YAML 1.2 specification proper (revision 1.2.2,
embedded below). A program is "valid" iff it parses successfully under the
spec's production rules; you are validating static (parse-time) correctness
only — the validator does not need to construct or emit the parsed
representation.

The full specification follows.
