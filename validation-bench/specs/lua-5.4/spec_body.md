# 3 – The Language

This section describes the lexis, the syntax, and the semantics of Lua.
Language constructs are explained using extended BNF notation, where {a}
means 0 or more a's, and [a] means an optional a. Non-terminals appear like
`non-terminal`, keywords like **kword**, and other terminal symbols like '**=**'.

## 3.1 – Lexical Conventions

Lua is free-form and ignores spaces and comments between lexical elements
(tokens), except as delimiters. The language recognizes standard ASCII
whitespace: space, form feed, newline, carriage return, horizontal tab, and
vertical tab.

Names (identifiers) may be any string of Latin letters, Arabic-Indic digits,
and underscores, not beginning with a digit and not being a reserved word.

Reserved keywords:

```
and       break     do        else      elseif    end
false     for       function  goto      if        in
local     nil       not       or        repeat    return
then      true      until     while
```

Lua is case-sensitive; `and` is reserved but `And` and `AND` are valid names.
By convention, avoid names starting with underscore followed by uppercase
letters.

Other token symbols:

```
+     -     *     /     %     ^     #
&     ~     |     <<    >>    //
==    ~=    <=    >=    <     >     =
(     )     {     }     [     ]     ::
;     :     ,     .     ..    ...
```

### Short Literal Strings

Short literal strings delimit with matching single or double quotes and
support C-like escape sequences: '`\a`' (bell), '`\b`' (backspace),
'`\f`' (form feed), '`\n`' (newline), '`\r`' (carriage return), '`\t`'
(horizontal tab), '`\v`' (vertical tab), '`\\`' (backslash), '`\"`' (double
quote), and '`\'`' (single quote).

A backslash followed by a line break produces a newline in the string. The
escape '`\z`' skips following whitespace including line breaks, useful for
breaking long strings across lines without adding unwanted characters.

Bytes in short strings can be specified numerically via '`\xXX`' (exactly two
hexadecimal digits) or '`\ddd`' (up to three decimal digits). UTF-8 encoding
uses '`\u{XXX}`' with mandatory braces, where XXX represents the Unicode code
point (any value less than 2^31).

### Long Literal Strings

Long literal strings use long brackets. An opening long bracket of level *n*
comprises an opening square bracket, *n* equal signs, and another opening
square bracket; a closing long bracket mirrors this structure. A long literal
starts with an opening bracket of any level and ends at the first closing
bracket of the same level.

Long literals can contain any text except a closing bracket of their level,
may span multiple lines, do not interpret escape sequences, and ignore
brackets of other levels. End-of-line sequences convert to simple newlines.
A newline immediately following the opening bracket is not included in the
string.

### Numeric Constants

A numeric constant accepts an optional fractional part and decimal exponent
(marked by 'e' or 'E'). Hexadecimal constants start with `0x` or `0X` and
accept optional fractional parts and binary exponents (marked by 'p' or 'P').

Constants with a radix point or exponent denote floats. Otherwise, if the
value fits in an integer or is hexadecimal, it denotes an integer; decimal
numerals that overflow denote floats. Hexadecimal numerals without radix
points or exponents always denote integers, wrapping on overflow.

### Comments

Comments start with double hyphen (`--`) outside strings. If the text
immediately following is not an opening long bracket, it is a short comment
running to line end. Otherwise, it is a long comment running to the matching
closing long bracket.

## 3.2 – Variables

Variables store values. Three kinds exist: global variables, local variables,
and table fields.

```
var ::= Name
```

Names denote identifiers. Any variable name is assumed global unless
explicitly declared local. Local variables are lexically scoped and can be
freely accessed by functions defined within their scope.

Before first assignment, a variable value is **nil**.

Square brackets index tables:

```
var ::= prefixexp '[' exp ']'
```

The syntax `var.Name` is syntactic sugar for `var["Name"]`:

```
var ::= prefixexp '.' Name
```

Access to global variable `x` equals `_ENV.x`. Due to compilation, `_ENV`
itself is never global.

## 3.3 – Statements

Lua supports conventional statements: blocks, assignments, control structures,
function calls, and variable declarations.

### 3.3.1 – Blocks

A block is a sequential list of statements:

```
block ::= {stat}
```

Empty statements separate statements with semicolons:

```
stat ::= ';'
```

Both function calls and assignments can start with open parentheses, creating
potential ambiguity. The parser interprets such constructions as function
calls. Precede statements starting with parentheses with semicolons to avoid
ambiguity:

```
;(print or io.write)('done')
```

Explicit block delimiters produce single statements:

```
stat ::= 'do' block 'end'
```

Explicit blocks control variable scope and add return statements mid-block.

### 3.3.2 – Chunks

The compilation unit is a chunk — syntactically just a block:

```
chunk ::= block
```

Lua handles chunks as bodies of anonymous functions with variable arguments.
Chunks define local variables, receive arguments, and return values. The
anonymous function compiles within scope of external local variable `_ENV`.

The resulting function always has `_ENV` as its sole external variable, even
if unused.

### 3.3.3 – Assignment

Lua allows multiple assignments with comma-separated variable and expression
lists:

```
stat ::= varlist '=' explist
varlist ::= var {',' var}
explist ::= exp {',' exp}
```

Before assignment, the values list adjusts to the variables list length.

If a variable is both assigned and read in a multiple assignment, all reads
use the pre-assignment value.

### 3.3.4 – Control Structures

Control structures **if**, **while**, and **repeat** have conventional
meanings:

```
stat ::= 'while' exp 'do' block 'end'
stat ::= 'repeat' block 'until' exp
stat ::= 'if' exp 'then' block {'elseif' exp 'then' block} ['else' block] 'end'
```

A **for** statement exists in two forms.

Condition expressions can return any value. Both **false** and **nil** test
false; all other values test true. Zero and empty strings test true.

In **repeat**–**until** loops, the block ends after the **until** condition,
allowing condition references to loop-declared locals.

The **goto** statement transfers control to a label:

```
stat ::= 'goto' Name
stat ::= label
label ::= '::' Name '::'
```

Labels are visible throughout their defining block except inside nested
functions. Gotos jump to visible labels without entering local variable
scopes. Labels with identical names should not coexist in visible scopes.

The **break** statement terminates **while**, **repeat**, or **for** loops:

```
stat ::= 'break'
```

**break** ends the innermost enclosing loop.

The **return** statement returns values from functions or chunks (anonymous
functions):

```
stat ::= 'return' [explist] [';']
```

Functions return multiple values. **return** statements must be the last
statement in blocks; explicit inner blocks enable mid-block returns via
`do return end`.

### 3.3.5 – For Statement

The **for** statement has two forms: numerical and generic.

Numerical:

```
stat ::= 'for' Name '=' exp ',' exp [',' exp] 'do' block 'end'
```

Generic:

```
stat ::= 'for' namelist 'in' explist 'do' block 'end'
namelist ::= Name {',' Name}
```

The identifier defines a loop-local control variable.

### 3.3.6 – Function Calls as Statements

Function calls execute as statements for side-effects:

```
stat ::= functioncall
```

### 3.3.7 – Local Declarations

Local variables declare anywhere within blocks with optional initialization:

```
stat ::= 'local' attnamelist ['=' explist]
attnamelist ::= Name attrib {',' Name attrib}
```

Variable names postfix with optional attributes in angle brackets:

```
attrib ::= ['<' Name '>']
```

Two attributes exist: `const` declares constant variables (unassignable
after initialization) and `close` declares to-be-closed variables.

A list can contain at most one to-be-closed variable.

Chunks are blocks, so local variables declare in chunks outside explicit
blocks.

## 3.4 – Expressions

Basic expressions:

```
exp ::= prefixexp
exp ::= nil | false | true
exp ::= Numeral
exp ::= LiteralString
exp ::= functiondef
exp ::= tableconstructor
exp ::= '...'
exp ::= exp binop exp
exp ::= unop exp
prefixexp ::= var | functioncall | '(' exp ')'
```

Vararg expressions (`...`) only appear directly inside variadic functions.

### Operators

Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `^`, unary `-`.
Bitwise: `&`, `|`, `~` (binary xor), `>>`, `<<`, unary `~`.
Relational: `==`, `~=`, `<`, `>`, `<=`, `>=`.
Logical: **and**, **or**, **not**.
Concatenation: `..`.
Length: unary prefix `#`.

Operator precedence (lowest to highest):

```
or
and
<     >     <=    >=    ~=    ==
|
~
&
<<    >>
..
+     -
*     /     //    %
unary operators (not   #     -     ~)
^
```

Concatenation (`..`) and exponentiation (`^`) are right associative; all
other binary operators are left associative.

### Table Constructors

```
tableconstructor ::= '{' [fieldlist] '}'
fieldlist ::= field {fieldsep field} [fieldsep]
field ::= '[' exp ']' '=' exp | Name '=' exp | exp
fieldsep ::= ',' | ';'
```

### Function Calls

```
functioncall ::= prefixexp args
functioncall ::= prefixexp ':' Name args
args ::= '(' [explist] ')'
args ::= tableconstructor
args ::= LiteralString
```

`v:name(args)` is syntactic sugar for `v.name(v, args)` (with v evaluated
once). `f{fields}` is syntactic sugar for `f({fields})`. `f"string"` (or
`f'string'` or `f[[string]]`) is syntactic sugar for `f("string")`.

### Function Definitions

```
functiondef ::= 'function' funcbody
funcbody ::= '(' [parlist] ')' block 'end'
stat ::= 'function' funcname funcbody
stat ::= 'local' 'function' Name funcbody
funcname ::= Name {'.' Name} [':' Name]
parlist ::= namelist [',' '...'] | '...'
```

Colon syntax in `funcname` adds an implicit `self` parameter:
`function t:f(params) body end` is equivalent to
`t.f = function (self, params) body end`.

## 3.5 – Visibility Rules

Lua is lexically scoped. Local variable scope begins at the first statement
after declaration and lasts until the last non-void statement of the
innermost containing block. (Void statements are labels and empty
statements.)

In declarations like `local x = x`, the new `x` isn't in scope yet, so the
second `x` references the outside variable.

Each **local** statement execution defines new locals.

---

# 9 – The Complete Syntax of Lua

```
chunk ::= block

block ::= {stat} [retstat]

stat ::= ';' |
     varlist '=' explist |
     functioncall |
     label |
     break |
     goto Name |
     do block end |
     while exp do block end |
     repeat block until exp |
     if exp then block {elseif exp then block} [else block] end |
     for Name '=' exp ',' exp [',' exp] do block end |
     for namelist in explist do block end |
     function funcname funcbody |
     local function Name funcbody |
     local attnamelist ['=' explist]

attnamelist ::= Name attrib {',' Name attrib}

attrib ::= ['<' Name '>']

retstat ::= return [explist] [';']

label ::= '::' Name '::'

funcname ::= Name {'.' Name} [':' Name]

varlist ::= var {',' var}

var ::= Name | prefixexp '[' exp ']' | prefixexp '.' Name

namelist ::= Name {',' Name}

explist ::= exp {',' exp}

exp ::= nil | false | true | Numeral | LiteralString | '...' | functiondef |
     tableconstructor | prefixexp | exp binop exp | unop exp

prefixexp ::= var | functioncall | '(' exp ')'

functioncall ::= prefixexp args | prefixexp ':' Name args

args ::= '(' [explist] ')' | tableconstructor | LiteralString

functiondef ::= function funcbody

funcbody ::= '(' [parlist] ')' block end

parlist ::= namelist [',' '...'] | '...'

tableconstructor ::= '{' [fieldlist] '}'

fieldlist ::= field {fieldsep field} [fieldsep]

field ::= '[' exp ']' '=' exp | Name '=' exp | exp

fieldsep ::= ',' | ';'

binop ::= '+' | '-' | '*' | '/' | '//' | '%' | '^' |
     '&' | '|' | '~' | '<<' | '>>' | '..' |
     '<' | '<=' | '>' | '>=' | '==' | '~=' | and | or

unop ::= '-' | not | '#' | '~'
```
