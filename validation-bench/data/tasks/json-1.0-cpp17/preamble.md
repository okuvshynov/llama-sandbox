You are an expert C++ programmer. Implement the solution described below.
Submit your complete C++ source code using the `submit` tool.
You will receive compilation and test results. Fix and resubmit if needed.

## Specification

Implement a validator for JSON texts in C++17 using only the standard
library.
Compiler command: `{compile_cmd}`

Your validator must read a JSON text from stdin and print to stdout
exactly `valid` (e.g. `std::cout << "valid"`) if it is a syntactically
valid JSON document, or exactly `invalid` otherwise. Surrounding
whitespace is allowed; anything else (debug output, mixed casing,
multiple lines) counts as a test failure. The process must also exit
cleanly with status 0 — a correct verdict followed by a crash, timeout,
or non-zero exit is still a failure.

The input is a JSON 1.0 document as specified by RFC 8259 / ECMA-404.
The full specification follows.
