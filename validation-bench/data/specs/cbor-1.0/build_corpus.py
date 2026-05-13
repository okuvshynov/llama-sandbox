#!/usr/bin/env python3
"""Materialize the cbor-1.0 hand-curated corpus from the inline TESTS
table below into data/specs/cbor-1.0/corpus/{valid,invalid}/...,
verifying each file's intended verdict against the reference oracle.

Run from the validation-bench root:

    python data/specs/cbor-1.0/build_corpus.py [--check-only]

Same generator pattern as data/specs/hcl-2/build_corpus.py — but the
inputs are *raw bytes* (CBOR is a binary format), so the inline table
uses bytes.fromhex(...) literals rather than triple-quoted text. The
materialized files are .cbor binaries.

This script does NOT produce tests/ or tests.jsonl — those are derived
artifacts (gitignored) that setup.sh's `generate_corpus_spec` materializes
from corpus/ after running its own oracle pass.

The "validity" bar tested here is "well-formed CBOR per RFC 8949 §3":
the bytes must encode exactly one CBOR data item, no truncation, no
trailing garbage. Stricter "validity" checks defined in RFC 8949 §5.3.1
(no duplicate map keys, valid UTF-8 in text strings, valid date strings
for tag 0, etc.) are intentionally NOT enforced — they vary across
parsers and would muddy the corpus with judgment calls.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORACLE = HERE.parent.parent.parent / "scripts" / "oracles" / "cbor-check" / "cbor-check"


def hx(s: str) -> bytes:
    """Hex-string → bytes shorthand. Spaces and underscores allowed."""
    return bytes.fromhex(s.replace(" ", "").replace("_", ""))


# Each entry: (subpath_under_tests/, expected_verdict, content_bytes).
TESTS: list[tuple[str, str, bytes]] = [
    # ============================================================
    # VALID — Major type 0 (unsigned integers)
    # ============================================================
    ("uint/0.cbor",                    "valid", hx("00")),                    # 0
    ("uint/1.cbor",                    "valid", hx("01")),                    # 1
    ("uint/10.cbor",                   "valid", hx("0a")),                    # 10
    ("uint/23-max-single-byte.cbor",   "valid", hx("17")),                    # 23
    ("uint/24-1byte-arg.cbor",         "valid", hx("18 18")),                 # 24
    ("uint/100.cbor",                  "valid", hx("18 64")),                 # 100
    ("uint/255-max-1byte.cbor",        "valid", hx("18 ff")),                 # 255
    ("uint/256-2byte-arg.cbor",        "valid", hx("19 01 00")),              # 256
    ("uint/65535-max-2byte.cbor",      "valid", hx("19 ff ff")),              # 65535
    ("uint/65536-4byte-arg.cbor",      "valid", hx("1a 00 01 00 00")),        # 65536
    ("uint/uint32-max.cbor",           "valid", hx("1a ff ff ff ff")),
    ("uint/2pow32-8byte-arg.cbor",     "valid", hx("1b 00 00 00 01 00 00 00 00")),
    ("uint/uint64-max.cbor",           "valid", hx("1b ff ff ff ff ff ff ff ff")),

    # ============================================================
    # VALID — Major type 1 (negative integers; encoded as -(n+1))
    # ============================================================
    ("nint/-1.cbor",                   "valid", hx("20")),
    ("nint/-10.cbor",                  "valid", hx("29")),
    ("nint/-24-max-single-byte.cbor",  "valid", hx("37")),
    ("nint/-25-1byte-arg.cbor",        "valid", hx("38 18")),
    ("nint/-100.cbor",                 "valid", hx("38 63")),
    ("nint/-256.cbor",                 "valid", hx("38 ff")),
    ("nint/-257-2byte-arg.cbor",       "valid", hx("39 01 00")),
    ("nint/-65537-4byte-arg.cbor",     "valid", hx("3a 00 01 00 00")),
    ("nint/-2pow32m1-8byte-arg.cbor",  "valid", hx("3b 00 00 00 01 00 00 00 00")),

    # ============================================================
    # VALID — Major type 2 (byte strings)
    # ============================================================
    ("bstr/empty.cbor",                "valid", hx("40")),
    ("bstr/1-byte.cbor",               "valid", hx("41 00")),
    ("bstr/4-bytes.cbor",              "valid", hx("44 01 02 03 04")),
    ("bstr/23-bytes.cbor",             "valid", hx("57") + bytes(range(23))),
    ("bstr/24-bytes-1byte-len.cbor",   "valid", hx("58 18") + bytes(range(24))),
    ("bstr/255-bytes.cbor",            "valid", hx("58 ff") + (b"\xab" * 255)),
    ("bstr/256-bytes-2byte-len.cbor",  "valid", hx("59 01 00") + (b"\xab" * 256)),
    ("bstr/indef-empty.cbor",          "valid", hx("5f ff")),                 # indefinite, no chunks
    ("bstr/indef-2-chunks.cbor",       "valid", hx("5f 42 01 02 43 03 04 05 ff")),
    ("bstr/indef-empty-chunk.cbor",    "valid", hx("5f 40 41 42 ff")),

    # ============================================================
    # VALID — Major type 3 (text strings; UTF-8)
    # ============================================================
    ("tstr/empty.cbor",                "valid", hx("60")),
    ("tstr/a.cbor",                    "valid", hx("61 61")),                  # "a"
    ("tstr/utf8-2byte.cbor",           "valid", hx("62 c3 a9")),               # "é"
    ("tstr/hello.cbor",                "valid", hx("65") + b"hello"),
    ("tstr/23-chars.cbor",             "valid", hx("77") + (b"x" * 23)),
    ("tstr/24-chars-1byte-len.cbor",   "valid", hx("78 18") + (b"x" * 24)),
    ("tstr/256-chars-2byte-len.cbor",  "valid", hx("79 01 00") + (b"x" * 256)),
    ("tstr/indef-2-chunks.cbor",       "valid", hx("7f 65") + b"hello" + hx("65") + b"world" + hx("ff")),

    # ============================================================
    # VALID — Major type 4 (arrays)
    # ============================================================
    ("array/empty.cbor",               "valid", hx("80")),
    ("array/single.cbor",              "valid", hx("81 01")),
    ("array/3-ints.cbor",              "valid", hx("83 01 02 03")),
    ("array/23-ints.cbor",             "valid", hx("97") + bytes(range(1, 24))),
    ("array/24-ints-1byte-len.cbor",   "valid", hx("98 18") + bytes(range(24))),
    ("array/nested.cbor",              "valid", hx("82 82 01 02 83 03 04 05")),  # [[1,2],[3,4,5]]
    ("array/heterogeneous.cbor",       "valid", hx("83 01 64") + b"text" + hx("f5")),  # [1, "text", true]
    ("array/indef-empty.cbor",         "valid", hx("9f ff")),
    ("array/indef-3-ints.cbor",        "valid", hx("9f 01 02 03 ff")),
    ("array/indef-nested.cbor",        "valid", hx("9f 9f 01 02 ff 83 03 04 05 ff")),

    # ============================================================
    # VALID — Major type 5 (maps)
    # ============================================================
    ("map/empty.cbor",                 "valid", hx("a0")),
    ("map/single-pair.cbor",           "valid", hx("a1 01 02")),               # {1: 2}
    ("map/2-string-keys.cbor",         "valid",
     hx("a2 61 61 01 61 62 02")),                                              # {"a":1, "b":2}
    ("map/string-to-string.cbor",      "valid",
     hx("a1 65") + b"hello" + hx("65") + b"world"),                            # {"hello":"world"}
    ("map/nested.cbor",                "valid", hx("a1 61 61 a1 61 62 01")),   # {"a":{"b":1}}
    ("map/array-value.cbor",           "valid", hx("a1 61 78 83 01 02 03")),   # {"x":[1,2,3]}
    ("map/indef-empty.cbor",           "valid", hx("bf ff")),
    ("map/indef-2-pairs.cbor",         "valid", hx("bf 61 61 01 61 62 02 ff")),

    # ============================================================
    # VALID — Major type 6 (tagged items)
    # ============================================================
    ("tag/0-datetime.cbor",            "valid",
     hx("c0 74") + b"2013-03-21T20:04:00Z"),                                   # tag 0 + 20-char text string
    ("tag/1-epoch-uint.cbor",          "valid", hx("c1 1a 51 4b 67 b0")),     # tag 1 + uint 1363896240
    ("tag/2-bignum.cbor",              "valid",
     hx("c2 49 01 00 00 00 00 00 00 00 00")),                                  # tag 2 + 9-byte bstr (2^64)
    ("tag/24-encoded-cbor.cbor",       "valid", hx("d8 18 45 64") + b"IETF"),  # tag 24 + 5-byte bstr containing CBOR text
    ("tag/32-uri.cbor",                "valid",
     hx("d8 20 76") + b"http://www.example.com"),                              # tag 32 + URI
    ("tag/55799-self-describe.cbor",   "valid", hx("d9 d9 f7 00")),           # CBOR magic + uint 0
    ("tag/2byte-arg.cbor",             "valid", hx("d9 01 00 00")),           # tag 256 + uint 0
    ("tag/4byte-arg.cbor",             "valid", hx("da 00 01 00 00 00")),     # tag 65536 + uint 0
    ("tag/8byte-arg.cbor",             "valid",
     hx("db 00 00 00 01 00 00 00 00 00")),

    # ============================================================
    # VALID — Major type 7 (simple values + floats)
    # ============================================================
    ("simple/false.cbor",              "valid", hx("f4")),
    ("simple/true.cbor",               "valid", hx("f5")),
    ("simple/null.cbor",               "valid", hx("f6")),
    ("simple/undefined.cbor",          "valid", hx("f7")),
    ("simple/value-0.cbor",            "valid", hx("e0")),                     # simple value 0
    ("simple/value-19.cbor",           "valid", hx("f3")),                     # simple value 19 (max in main range)
    ("simple/value-32.cbor",           "valid", hx("f8 20")),                  # simple value 32 (1-byte arg)
    ("simple/value-255.cbor",          "valid", hx("f8 ff")),
    ("float/half-pos-zero.cbor",       "valid", hx("f9 00 00")),
    ("float/half-neg-zero.cbor",       "valid", hx("f9 80 00")),
    ("float/half-1.0.cbor",            "valid", hx("f9 3c 00")),
    ("float/half-pos-inf.cbor",        "valid", hx("f9 7c 00")),
    ("float/half-nan.cbor",            "valid", hx("f9 7e 00")),
    ("float/single-100000.0.cbor",     "valid", hx("fa 47 c3 50 00")),
    ("float/double-pi.cbor",           "valid", hx("fb 40 09 21 fb 54 44 2d 18")),
    ("float/double-1e300.cbor",        "valid", hx("fb 7e 37 e4 3c 88 00 75 9c")),

    # ============================================================
    # VALID — realistic / RFC appendix examples
    # ============================================================
    ("realistic/coap-link-format.cbor", "valid",
     hx("a2 61 61 01 61 62 82 02 03")),                                        # {"a":1, "b":[2,3]}
    ("realistic/array-of-objects.cbor", "valid",
     hx("82 a1 61 61 01 a1 61 62 02")),                                        # [{"a":1}, {"b":2}]
    ("realistic/all-types.cbor",       "valid",
     hx("87 00 20 60 80 a0 f4 f6")),                                           # [0, -1, "", [], {}, false, null]
    ("realistic/deep-nesting.cbor",    "valid",
     hx("81 81 81 81 81 81 81 00")),                                           # 7 levels deep

    # ============================================================
    # INVALID — truncation
    # ============================================================
    ("truncation/empty-input.cbor",    "invalid", b""),                        # 0 bytes = no data item
    ("truncation/uint-1byte-missing.cbor",  "invalid", hx("18")),
    ("truncation/uint-2byte-partial.cbor",  "invalid", hx("19 01")),
    ("truncation/uint-4byte-partial.cbor",  "invalid", hx("1a 00 00 00")),
    ("truncation/uint-8byte-partial.cbor",  "invalid", hx("1b 00 00 00 00")),
    ("truncation/bstr-len-only.cbor",  "invalid", hx("44")),                   # claims 4 bytes, gives 0
    ("truncation/bstr-partial.cbor",   "invalid", hx("44 01 02")),             # claims 4, gives 2
    ("truncation/tstr-partial.cbor",   "invalid", hx("65") + b"hi"),           # claims 5 chars, gives 2
    ("truncation/array-short.cbor",    "invalid", hx("83 01 02")),             # claims 3, has 2
    ("truncation/array-empty-decl.cbor", "invalid", hx("83")),                 # claims 3, has 0
    ("truncation/map-half-pair.cbor",  "invalid", hx("a2 61 61 01 61 62")),    # claims 2 pairs, has 1.5
    ("truncation/map-no-value.cbor",   "invalid", hx("a1 01")),                # claims 1 pair, has key only
    ("truncation/tag-no-content.cbor", "invalid", hx("c0")),
    ("truncation/tag1byte-arg-missing.cbor", "invalid", hx("d8")),
    ("truncation/simple-1byte-arg-missing.cbor", "invalid", hx("f8")),
    ("truncation/half-float-partial.cbor", "invalid", hx("f9 00")),
    ("truncation/double-float-partial.cbor", "invalid", hx("fb 40 09 21")),

    # ============================================================
    # INVALID — reserved additional info (AI 28..30 = reserved)
    # ============================================================
    ("reserved/major0-ai28.cbor",      "invalid", hx("1c")),
    ("reserved/major0-ai29.cbor",      "invalid", hx("1d")),
    ("reserved/major0-ai30.cbor",      "invalid", hx("1e")),
    ("reserved/major1-ai28.cbor",      "invalid", hx("3c")),
    ("reserved/major2-ai28.cbor",      "invalid", hx("5c")),
    ("reserved/major3-ai28.cbor",      "invalid", hx("7c")),
    ("reserved/major4-ai28.cbor",      "invalid", hx("9c")),
    ("reserved/major5-ai28.cbor",      "invalid", hx("bc")),
    ("reserved/major6-ai28.cbor",      "invalid", hx("dc")),
    ("reserved/major7-ai28.cbor",      "invalid", hx("fc")),                   # AI 28-30 in major 7 are reserved

    # ============================================================
    # INVALID — bad break (0xff outside indefinite-length context)
    # ============================================================
    ("break/lone.cbor",                "invalid", hx("ff")),
    ("break/in-fixed-array.cbor",      "invalid", hx("83 01 ff 03")),
    ("break/in-fixed-map.cbor",        "invalid", hx("a1 01 ff")),
    ("break/at-toplevel-after-data.cbor", "invalid", hx("00 ff")),

    # ============================================================
    # INVALID — indefinite-length type/structure issues
    # ============================================================
    ("indef/map-odd-items.cbor",       "invalid", hx("bf 61 61 01 61 62 ff")),  # odd count: key without value
    ("indef/bstr-with-tstr-chunk.cbor", "invalid", hx("5f 60 ff")),             # indef bstr containing tstr chunk
    ("indef/bstr-with-int-chunk.cbor", "invalid", hx("5f 00 ff")),              # chunk must be byte-string
    ("indef/tstr-with-bstr-chunk.cbor", "invalid", hx("7f 40 ff")),
    ("indef/bstr-nested-indef-chunk.cbor", "invalid", hx("5f 5f 41 00 ff ff")), # chunks themselves must be definite-length

    # ============================================================
    # INVALID — trailing data (more than one CBOR item)
    # ============================================================
    ("trailing/two-uints.cbor",        "invalid", hx("00 00")),
    ("trailing/uint-then-byte.cbor",   "invalid", hx("01 ff")),
    ("trailing/array-then-int.cbor",   "invalid", hx("83 01 02 03 04")),
    ("trailing/tag-then-extra.cbor",   "invalid", hx("c0 60 00")),
]


def materialize(check_only: bool) -> list[str]:
    """Write each TESTS entry to disk under corpus/<verdict>/<subpath>
    (overwriting any existing file) and run the oracle on each to confirm
    intended verdict matches the parser's. Returns the list of drift
    reports (oracle disagreed with the file's directory placement)."""
    corpus_dir = HERE / "corpus"
    if not check_only:
        if corpus_dir.exists():
            shutil.rmtree(corpus_dir)
        (corpus_dir / "valid").mkdir(parents=True)
        (corpus_dir / "invalid").mkdir(parents=True)

    drift: list[str] = []
    n_valid = n_invalid = 0
    for subpath, verdict, content in TESTS:
        if verdict not in ("valid", "invalid"):
            raise ValueError(f"bad verdict {verdict!r} for {subpath}")
        target = corpus_dir / verdict / subpath
        if not check_only:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        if not target.exists():
            drift.append(f"missing: {target.relative_to(HERE)}")
            continue
        result = subprocess.run([str(ORACLE), str(target)],
                                capture_output=True)
        oracle_says = "valid" if result.returncode == 0 else "invalid"
        if oracle_says != verdict:
            drift.append(
                f"{target.relative_to(HERE)}: file in corpus/{verdict}/ "
                f"but oracle returned {oracle_says} (rc={result.returncode}). "
                f"stderr: {result.stderr.decode().strip()[:200]}"
            )
            continue
        if verdict == "valid":
            n_valid += 1
        else:
            n_invalid += 1

    print(f"corpus: {n_valid + n_invalid} tests "
          f"({n_valid} valid, {n_invalid} invalid)")
    return drift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="Verify on-disk corpus without (re)writing files")
    args = parser.parse_args()

    if not ORACLE.exists():
        print(f"oracle not built at {ORACLE}", file=sys.stderr)
        print("Build it with: (cd scripts/oracles/cbor-check && go build -o cbor-check .)",
              file=sys.stderr)
        return 2

    drift = materialize(check_only=args.check_only)
    if drift:
        print(f"DRIFT ({len(drift)} files):")
        for d in drift:
            print(f"  {d}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
