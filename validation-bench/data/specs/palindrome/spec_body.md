# Byte-level palindrome detection

A sequence of bytes is a **palindrome** if and only if reversing the byte
sequence yields the same sequence.

This is a **byte-level** check:

- No case folding. `"Aa"` is *not* a palindrome — the first byte (`'A'`,
  0x41) differs from the last byte (`'a'`, 0x61).
- No whitespace stripping. Trailing newlines, spaces, and other whitespace
  are part of the input. `"abba\n"` is *not* a palindrome — the leading
  byte `'a'` (0x61) differs from the trailing newline (0x0a).
- No Unicode-aware character segmentation. Multi-byte UTF-8 sequences
  are compared as raw bytes; a string that *looks* symmetric in characters
  may not be byte-symmetric.

## Edge cases

- The **empty input** (zero bytes) is a palindrome (it trivially equals
  its reverse).
- Any **single-byte input** is a palindrome.

## Examples

| input bytes (escaped)      | valid? |
|----------------------------|--------|
| `""`                       | yes    |
| `"a"`                      | yes    |
| `"abba"`                   | yes    |
| `"racecar"`                | yes    |
| `"12321"`                  | yes    |
| `"a\na"`                   | yes    |
| `"abc"`                    | no     |
| `"Aa"`                     | no     |
| `"abba\n"`                 | no     |
| `"abba x"`                 | no     |
