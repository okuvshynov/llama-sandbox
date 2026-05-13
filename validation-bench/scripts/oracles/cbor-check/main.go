// Reference oracle for the cbor-1.0 spec.
//
// Reads the file given as argv[1] as raw bytes and exits 0 if its
// contents are well-formed CBOR per RFC 8949, non-zero otherwise.
//
// "Well-formed" here means RFC 8949 §3 — the bytes parse as exactly
// one CBOR data item with no trailing garbage and no truncation. We
// deliberately use the weaker "well-formed" bar (not the stricter
// "valid" defined in §5.3.1, which adds constraints like "no
// duplicate map keys" and "valid UTF-8 in text strings") because:
//
//   1. Well-formedness is unambiguous — every conformant decoder
//      agrees on what parses; "validity" varies by parser strictness.
//   2. Most practical CBOR decoders default to well-formedness only
//      and surface validity violations as warnings or via a separate
//      strict-mode flag.
//   3. Test-corpus design is much cleaner — invalid cases can target
//      the encoding rules directly (truncation, reserved AI values,
//      bad break codes) without litigating UTF-8 edge cases.
//
// The Wellformed function from github.com/fxamacker/cbor/v2 implements
// exactly this check — it scans bytes without allocating decoded
// values, so it's a pure framing/structure validator.
package main

import (
	"fmt"
	"os"

	"github.com/fxamacker/cbor/v2"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintf(os.Stderr, "usage: %s <file>\n", os.Args[0])
		os.Exit(2)
	}
	data, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "read: %v\n", err)
		os.Exit(2)
	}
	if err := cbor.Wellformed(data); err != nil {
		os.Exit(1)
	}
	os.Exit(0)
}
