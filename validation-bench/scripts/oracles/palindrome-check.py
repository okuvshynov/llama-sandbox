#!/usr/bin/env python3
"""Reference oracle for the palindrome spec.

Reads the file given as argv[1] in binary mode and exits 0 if its contents
are a byte-level palindrome (data == data[::-1]), non-zero otherwise.
Used by setup.sh's generate_corpus_spec to re-derive labels at corpus
materialization time, so a checked-in corpus file in valid/ that fails
this check (or vice versa) aborts setup with a clear MISMATCH report.
"""

import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <file>", file=sys.stderr)
        return 2
    with open(sys.argv[1], "rb") as f:
        data = f.read()
    return 0 if data == data[::-1] else 1


if __name__ == "__main__":
    sys.exit(main())
