You are an expert C++ programmer. Implement the solution described below.
Submit your complete C++ source code using the `submit` tool.
You will receive compilation and test results. Fix and resubmit if needed.

## Task

Implement an HCL2 (HashiCorp Configuration Language v2 native syntax)
validator in C++17 using only the standard library.
Compiler command: `{compile_cmd}`

Your validator must read an HCL2 document from stdin and print to stdout
exactly `valid` (e.g. `std::cout << "valid"`) if the document is
syntactically valid, or exactly `invalid` otherwise. Surrounding whitespace
is allowed; anything else (debug output, mixed casing, multiple lines)
counts as a test failure. The process must also exit cleanly with status 0
— a correct verdict followed by a crash, timeout, or non-zero exit is
still a failure.

A document is "valid" iff it parses successfully under HCL2's native-syntax
grammar; you are validating static (parse-time) syntactic correctness only
— the validator does not need to evaluate expressions, resolve references,
type-check, or impose any application schema (e.g. Terraform's
resource/variable rules) on top of the bare HCL2 grammar.

You must implement the validator based on your knowledge of the HCL2 specification.
No specification text is provided — use what you know about the format.
