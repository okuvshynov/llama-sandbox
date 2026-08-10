#!/usr/bin/env python3
"""Measure nano-chat's tokenizer against llama.cpp's, token for token.

The tokenizer is the one part of nano-glm that is *not* covered by the logit
gate, and deliberately so: a tokenizer bug would otherwise present as a
numerics failure and send you looking in the wrong place (PLAN.md step 7). So
it gets its own check, and the check is a comparison against an independent
implementation rather than a set of hand-written expectations.

    python tokenizer_check.py [-m MODEL] [--corpus DIR] [--verbose]

Both sides tokenize the same UTF-8 text with no special-token parsing and no
BOS, which is the only mode where "the same input" is unambiguous:

    nano-chat --raw --dry-run      our byte-level BPE + the glm4 pre-tokenizer
    llama-tokenize --ids --no-bos --no-parse-special

Cases are the five corpus prompts plus a set chosen to hit the parts of the
pre-tokenizer regex that English prose never reaches: CJK (no spaces, so every
split is a category decision), combining accents, emoji outside the BMP,
digit-run chunking at \\p{N}{1,3}, contractions, and the whitespace
alternatives, which are the two that need backtracking and so are the two most
likely to be subtly wrong.

Needs llama-tokenize, which is not built by default:

    cmake --build <llama.cpp>/build --target llama-tokenize
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXE = ".exe" if os.name == "nt" else ""

DEFAULT_MODEL = os.environ.get(
    "NANO_MODEL", r"D:\llms\UD-Q6_K\GLM-5.2-UD-Q6_K-00001-of-00014.gguf")
DEFAULT_LLAMA = os.environ.get("LLAMA_CPP_DIR", r"C:\Users\oleksandr\Desktop\llama.cpp")

# Text the corpus does not contain. Each line names what it is probing.
CASES = [
    ("ascii",           "The quick brown fox jumps over the lazy dog."),
    ("contractions",    "I'm sure it's theirs, they've won, we'll see, don't you'd rather?"),
    ("digits short",    "1 22 333 4444 55555 666666"),
    ("digits mixed",    "Order 66 in 1999 cost $1,234.56 (about 78%)."),
    ("cjk",             "\u4eba\u5de5\u667a\u80fd\u6a21\u578b\u7684\u53c2\u6570\u91cf\u975e\u5e38\u5927\u3002"),
    ("cjk + ascii",     "GLM-5.2 \u662f\u4e00\u4e2a MoE \u6a21\u578b\uff0c\u6709 256 \u4e2a\u4e13\u5bb6\u3002"),
    ("japanese",        "\u3053\u3093\u306b\u3061\u306f\u3001\u4e16\u754c\u3002\u30c8\u30fc\u30af\u30ca\u30a4\u30b6\u30fc"),
    ("korean",          "\uc548\ub155\ud558\uc138\uc694 \uc138\uacc4"),
    ("cyrillic",        "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! \u041a\u0430\u043a \u0434\u0435\u043b\u0430?"),
    ("greek + math",    "\u03b1\u03b2\u03b3 \u2211 x\u1d62 \u2248 \u222b f(x)dx, \u2200\u03b5>0"),
    ("accents",         "na\u00efve fa\u00e7ade cr\u00e8me br\u00fbl\u00e9e \u00fcber Stra\u00dfe"),
    ("combining",       "e\u0301te\u0301 a\u0300 co\u0302te\u0301 nai\u0308f"),
    ("emoji",           "ok \U0001f600 \U0001f469\u200d\U0001f4bb \U0001f1fa\U0001f1f8 done"),
    ("whitespace runs", "a  b   c\td\n\ne\n\n\nf   "),
    ("newline heavy",   "line1\n  line2\n\n\tline3\r\nline4"),
    ("leading space",   "   leading and trailing   "),
    ("punct runs",      "wait... really?! ---> [x](y) {a:1} <<>>"),
    ("code",            "if (x <= 3 && y != 'a') { return v[i]->f(); } // done\n"),
    ("urls",            "see https://example.com/a_b-c?d=1&e=2#f for more"),
    ("empty-ish",       " "),
    ("single char",     "a"),
    ("only punct",      "!!!"),
    ("mixed script",    "Tokyo \u6771\u4eac Moscow \u041c\u043e\u0441\u043a\u0432\u0430 2024\u5e74"),
]


def ours(nano_chat, model, text):
    # Via -f for the same reason as llama-tokenize below: a file has no code
    # page and no length limit. (nano-chat handles UTF-8 argv correctly now,
    # but the check should not depend on that being true.)
    tmp = _tmpfile(text)
    r = subprocess.run([nano_chat, "-m", model, "-f", tmp, "--raw", "--dry-run"],
                       capture_output=True)
    if r.returncode != 0:
        raise SystemExit("nano-chat failed: %s" % r.stderr.decode("utf-8", "replace")[-500:])
    out = r.stdout.decode("utf-8", "replace").strip()
    return [int(x) for x in out.split(",")] if out else []


def _tmpfile(text):
    tmp = os.path.join(HERE, "results", "_tokcheck.txt")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(text.encode("utf-8"))
    return tmp


def theirs(llama_tokenize, model, text):
    # Through a file, not -p: argv would mangle newlines and non-ASCII on the
    # way in, and then the two sides would not be tokenizing the same bytes —
    # the one thing this script must not get wrong.
    tmp = _tmpfile(text)
    r = subprocess.run([llama_tokenize, "-m", model, "-f", tmp, "--ids",
                        "--no-bos", "--no-parse-special"],
                       capture_output=True)
    if r.returncode != 0:
        raise SystemExit("llama-tokenize failed: %s" % r.stderr.decode("utf-8", "replace")[-500:])
    out = r.stdout.decode("utf-8", "replace")
    lo, hi = out.rfind("["), out.rfind("]")
    if lo < 0 or hi < 0:
        raise SystemExit("could not find an id list in llama-tokenize output:\n" + out[-500:])
    body = out[lo + 1:hi].strip()
    return [int(x) for x in body.split(",")] if body else []


def first_diff(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("--build", default=os.path.join(HERE, "build", "bin"))
    ap.add_argument("--llama-cpp", default=DEFAULT_LLAMA)
    ap.add_argument("--corpus", default=os.path.join(REPO, "logit-kld", "prompts"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    nano_chat = os.path.abspath(os.path.join(args.build, "nano-chat" + EXE))
    llama_tok = os.path.abspath(os.path.join(args.llama_cpp, "build", "bin",
                                             "llama-tokenize" + EXE))
    for p, what in ((nano_chat, "nano-chat"), (llama_tok, "llama-tokenize")):
        if not os.path.exists(p):
            raise SystemExit("%s not found at %s" % (what, p))

    cases = list(CASES)
    if os.path.isdir(args.corpus):
        for name in sorted(os.listdir(args.corpus)):
            if name.endswith(".txt"):
                with open(os.path.join(args.corpus, name), encoding="utf-8") as f:
                    cases.append(("corpus/" + name, f.read()))

    print("=" * 72)
    print("tokenizer check: nano-chat vs llama-tokenize")
    print("  model  %s" % os.path.basename(args.model))
    print("  cases  %d" % len(cases))
    print()

    n_tok = n_bad_tok = 0
    failures = []
    for name, text in cases:
        a = ours(nano_chat, args.model, text)
        b = theirs(llama_tok, args.model, text)
        n_tok += len(b)
        if a == b:
            print("  ok    %-16s %4d tokens" % (name, len(b)))
            continue
        d = first_diff(a, b)
        n_bad_tok += max(len(a), len(b)) - d
        failures.append(name)
        print("  FAIL  %-16s ours %d, llama.cpp %d, first differs at %d"
              % (name, len(a), len(b), d))
        if args.verbose:
            print("        ours      %s" % a[max(0, d - 3):d + 6])
            print("        llama.cpp %s" % b[max(0, d - 3):d + 6])

    print()
    print("%d/%d cases agree; %d of %d tokens differ (%.3f%%)"
          % (len(cases) - len(failures), len(cases), n_bad_tok, n_tok,
             100.0 * n_bad_tok / max(1, n_tok)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
