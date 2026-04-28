You are an expert C++ programmer. Implement the solution described below.
Submit your complete C++ source code using the `submit` tool.
You will receive compilation and test results. Fix and resubmit if needed.

## Specification

Implement a Lua 5.4 syntactic validator in C++17 using only the standard library.
Compiler command: `{compile_cmd}`

Your validator must read a Lua 5.4 source from stdin.
If the source is syntactically valid, exit with zero exit code.
If invalid, exit with non-zero exit code.

The reference oracle is `luac5.4 -p file.lua`. A program is "valid" if and only
if `luac5.4 -p` accepts it (exits 0). You are validating static (parse-time)
correctness only — runtime errors, undefined variables, calls to nonexistent
library functions, etc. do not make a program invalid here.

The relevant sections of the Lua 5.4 reference manual follow.