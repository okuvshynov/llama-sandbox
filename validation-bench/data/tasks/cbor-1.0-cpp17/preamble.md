You are an expert C++ programmer. Implement the solution described below.
Submit your complete C++ source code using the `submit` tool.
You will receive compilation and test results. Fix and resubmit if needed.

## Specification

Implement a CBOR (Concise Binary Object Representation, RFC 8949)
**well-formedness** validator in C++17 using only the standard library.
Compiler command: `{compile_cmd}`

Your validator must read CBOR-encoded **binary** bytes from stdin and
print to stdout exactly `valid` (e.g. `std::cout << "valid"`) if the
input is well-formed CBOR per RFC 8949 §3, or exactly `invalid`
otherwise. Surrounding whitespace is allowed; anything else (debug
output, mixed casing, multiple lines) counts as a test failure. The
process must also exit cleanly with status 0 — a correct verdict
followed by a crash, timeout, or non-zero exit is still a failure.

The input is binary, not text — use `std::cin.read()` after
`std::ios::sync_with_stdio(false)` (or read directly from
`std::freopen(nullptr, "rb", stdin)` and `fread`) to consume raw
bytes including embedded NULs. **Do not** use `std::getline` or any
text-mode reader, which will mangle 0x0a / 0x0d bytes.

A document is "well-formed" iff exactly one CBOR data item can be
parsed from the bytes per RFC 8949 §3, with no truncation and no
trailing bytes. You are NOT being asked to enforce the stricter
"validity" rules in §5.3.1 (no duplicate map keys, valid UTF-8 in
text strings, valid date strings for tag 0, etc.) — those are
explicitly out of scope and the test corpus doesn't probe them.

The reference oracle is the `Wellformed` function in
`github.com/fxamacker/cbor/v2`, which scans bytes without allocating
decoded values; a document is "valid" iff that function returns no
error.

The full RFC 8949 specification follows.
