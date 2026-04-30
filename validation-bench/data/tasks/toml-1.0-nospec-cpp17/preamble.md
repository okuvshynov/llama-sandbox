You are an expert C++ programmer. Implement the solution described below.
Submit your complete C++ source code using the `submit` tool.
You will receive compilation and test results. Fix and resubmit if needed.

## Task

Implement a validator for TOML v1.0.0 files in C++17 using only the standard library.
Compiler command: `{compile_cmd}`

Your validator must read a TOML file from stdin and print to stdout
exactly `valid` (e.g. `std::cout << "valid"`) if it is a valid TOML
document, or exactly `invalid` otherwise. Surrounding whitespace is
allowed; anything else (debug output, mixed casing, multiple lines)
counts as a test failure. Exit code is not checked; only the printed
verdict.

You must implement the validator based on your knowledge of the TOML v1.0.0 specification.
No specification text is provided — use what you know about the format.
