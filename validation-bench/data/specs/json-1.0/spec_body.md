# JSON Data Interchange Syntax (RFC 8259 / ECMA-404)

JSON ("JavaScript Object Notation") is a lightweight text-based
data interchange format. A JSON text is a sequence of Unicode
code points (encoded as UTF-8 in practice) forming exactly one
JSON value, optionally surrounded by whitespace.

This task validates the **syntactic** correctness of a JSON text
according to the grammar below. Semantic concerns (e.g. duplicate
object keys, number precision, deeply nested structures beyond an
implementation's limit) are not in scope; the implementation must
accept exactly the texts described by the grammar.

## Grammar (ABNF)

```
JSON-text = ws value ws

begin-array     = ws %x5B ws  ; [ left square bracket
begin-object    = ws %x7B ws  ; { left curly bracket
end-array       = ws %x5D ws  ; ] right square bracket
end-object      = ws %x7D ws  ; } right curly bracket
name-separator  = ws %x3A ws  ; : colon
value-separator = ws %x2C ws  ; , comma

ws = *(
        %x20 /                ; Space
        %x09 /                ; Horizontal tab
        %x0A /                ; Line feed or New line
        %x0D )                ; Carriage return

value = false / null / true / object / array / number / string

false = %x66.61.6c.73.65      ; false
null  = %x6e.75.6c.6c          ; null
true  = %x74.72.75.65          ; true

object = begin-object [ member *( value-separator member ) ] end-object

member = string name-separator value

array = begin-array [ value *( value-separator value ) ] end-array

number = [ minus ] int [ frac ] [ exp ]

decimal-point = %x2E          ; .
digit1-9 = %x31-39             ; 1-9
e = %x65 / %x45                ; e E
exp = e [ minus / plus ] 1*DIGIT
frac = decimal-point 1*DIGIT
int = zero / ( digit1-9 *DIGIT )
minus = %x2D                   ; -
plus = %x2B                    ; +
zero = %x30                    ; 0

string = quotation-mark *char quotation-mark

char = unescaped /
       escape (
           %x22 /              ; "    quotation mark   U+0022
           %x5C /              ; \    reverse solidus  U+005C
           %x2F /              ; /    solidus          U+002F
           %x62 /              ; b    backspace        U+0008
           %x66 /              ; f    form feed        U+000C
           %x6E /              ; n    line feed        U+000A
           %x72 /              ; r    carriage return  U+000D
           %x74 /              ; t    tab              U+0009
           %x75 4HEXDIG )      ; uXXXX                 U+XXXX

escape = %x5C                  ; \
quotation-mark = %x22          ; "
unescaped = %x20-21 / %x23-5B / %x5D-10FFFF
```

## Key rules that trip up casual implementations

- **Empty input is invalid.** A JSON text must contain exactly one
  value (possibly surrounded by whitespace). Zero-byte input is not
  a valid JSON document.
- **Leading zeros are invalid** in numbers (`01`, `007` are invalid;
  `0`, `0.1`, `0e1` are valid). Only the literal `0` may be used
  for zero; subsequent digits require a leading `1`-`9`.
- **Leading `+` is invalid** on numbers (`+1` is invalid; `1` and
  `-1` are valid).
- **Trailing comma is invalid** in arrays (`[1,2,]`) and objects
  (`{"a":1,}`). The grammar does not allow it.
- **Comments are not part of JSON.** Lines starting with `//` or
  `/* ... */` blocks are invalid.
- **Strings must use double quotes** (`"`). Single-quoted strings
  (`'foo'`) are invalid.
- **Control characters U+0000..U+001F** (except via `\u` escape)
  are invalid inside strings. A literal newline in a string body
  is invalid.
- **Invalid escape sequences** (`\x`, `\a`, etc. — anything not in
  the char production above) make the string invalid.
- **`\u` escapes must have exactly four hex digits**. `\u12` is
  invalid; `ÿ` is valid.
- **Object members and array elements must be separated by exactly
  one comma**, with no comma after the last element.

## Examples

| input                              | valid? | reason                          |
|------------------------------------|--------|---------------------------------|
| `{}`                               | yes    | empty object                    |
| `[]`                               | yes    | empty array                     |
| `null`                             | yes    | null value                      |
| `"hello"`                          | yes    | string value                    |
| `42`                               | yes    | number value                    |
| `-0.5e+10`                         | yes    | full number production          |
| `{"a": 1, "b": [true, false]}`     | yes    | nested object + array           |
| `   42   `                         | yes    | whitespace around value         |
| ``                                 | no     | empty input — no value          |
| `{"a": 1,}`                        | no     | trailing comma                  |
| `[1, 2,]`                          | no     | trailing comma                  |
| `01`                               | no     | leading zero                    |
| `+1`                               | no     | leading `+`                     |
| `.5`                               | no     | missing integer part            |
| `'foo'`                            | no     | single quotes                   |
| `// comment\n42`                   | no     | comments not allowed            |
| `{a: 1}`                           | no     | unquoted key                    |
| `{"a": 1, "a": 2}`                 | yes    | duplicate keys are syntactically valid |
| `"\u00"`                           | no     | `\u` needs 4 hex digits         |
| `"hello`                           | no     | unterminated string             |
| `[1, , 2]`                         | no     | empty array slot                |
