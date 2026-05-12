#!/usr/bin/env python3
"""Materialize the hcl-2 hand-curated corpus from the inline TESTS table
below into data/specs/hcl-2/corpus/{valid,invalid}/..., and verify each
file's intended label against the reference oracle.

Run from the validation-bench root:

    python data/specs/hcl-2/build_corpus.py [--check-only]

By default this wipes data/specs/hcl-2/corpus/ and rewrites every test
file from the inline TESTS table, then runs the oracle on each and
aborts (with diff report) if any file's label disagrees with its
directory placement. With --check-only it skips writing and just
verifies that every existing file matches its intended label — useful
in CI / pre-commit hooks to catch drift after a parser bump.

This script does NOT produce tests/ or tests.jsonl — those are derived
artifacts (gitignored) that setup.sh's `generate_corpus_spec` materializes
from corpus/ after running its own oracle pass. The inline TESTS table
here is the single source of truth for the design; corpus/ is a
materialized form that's checked in (matching the lua-5.4 / palindrome
hand-curated pattern); tests/ + tests.jsonl is the harness-readable form
generated at setup time.

Why a generator script (vs. just hand-editing corpus/ files like
lua-5.4 does): HCL has no canonical upstream test-suite repo (toml-test,
yaml-test-suite, JSONTestSuite all exist; an "hcl-test" doesn't). With
~150 small files all using a similar shape, a single inline TESTS table
makes design intent reviewable at a glance and adding a test is a
one-line edit.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORACLE = HERE.parent.parent.parent / "scripts" / "oracles" / "hcl-check" / "hcl-check"

# Each entry: (subpath_under_tests/, expected_verdict, content).
# The subpath is "<category>/<name>.hcl"; the file is materialized at
# tests/<expected>/<subpath>. The category prefix in the subpath is for
# human readability only — the oracle doesn't see it.
TESTS: list[tuple[str, str, str]] = [
    # ============================================================
    # VALID — attributes
    # ============================================================
    ("attributes/string-01.hcl", "valid", 'foo = "bar"\n'),
    ("attributes/string-empty.hcl", "valid", 'foo = ""\n'),
    ("attributes/number-int.hcl", "valid", 'count = 42\n'),
    ("attributes/number-float.hcl", "valid", 'pi = 3.14159\n'),
    ("attributes/number-negative.hcl", "valid", 'temp = -273.15\n'),
    ("attributes/number-zero.hcl", "valid", 'zero = 0\n'),
    ("attributes/number-scientific.hcl", "valid", 'avogadro = 6.022e23\n'),
    ("attributes/bool-true.hcl", "valid", 'enabled = true\n'),
    ("attributes/bool-false.hcl", "valid", 'enabled = false\n'),
    ("attributes/null.hcl", "valid", 'unset = null\n'),
    ("attributes/multiple.hcl", "valid",
     'a = 1\nb = "two"\nc = true\nd = null\n'),
    ("attributes/identifier-with-underscore.hcl", "valid",
     'snake_case_name = 1\n'),
    ("attributes/identifier-with-digit.hcl", "valid",
     'attr2 = "ok"\n'),

    # ============================================================
    # VALID — strings & escapes
    # ============================================================
    ("strings/escape-newline.hcl", "valid", 'msg = "line1\\nline2"\n'),
    ("strings/escape-tab.hcl", "valid", 'msg = "col1\\tcol2"\n'),
    ("strings/escape-quote.hcl", "valid", 'msg = "say \\"hi\\""\n'),
    ("strings/escape-backslash.hcl", "valid", 'path = "C:\\\\Users"\n'),
    ("strings/escape-unicode-bmp.hcl", "valid", 'snowman = "\\u2603"\n'),
    ("strings/escape-unicode-astral.hcl", "valid",
     'face = "\\U0001F600"\n'),
    ("strings/unicode-literal.hcl", "valid", 'greeting = "héllo wörld"\n'),

    ("strings/heredoc-basic.hcl", "valid",
     'msg = <<EOT\nhello\nworld\nEOT\n'),
    ("strings/heredoc-indented.hcl", "valid",
     'msg = <<-EOT\n    hello\n    world\n    EOT\n'),
    ("strings/heredoc-with-interp.hcl", "valid",
     'msg = <<EOT\nhello ${name}\nEOT\n'),
    ("strings/heredoc-empty.hcl", "valid", 'msg = <<EOT\nEOT\n'),

    # ============================================================
    # VALID — collections
    # ============================================================
    ("collections/tuple-empty.hcl", "valid", 'xs = []\n'),
    ("collections/tuple-numbers.hcl", "valid", 'xs = [1, 2, 3]\n'),
    ("collections/tuple-mixed.hcl", "valid",
     'xs = [1, "two", true, null]\n'),
    ("collections/tuple-nested.hcl", "valid",
     'matrix = [[1, 2], [3, 4]]\n'),
    ("collections/tuple-trailing-comma.hcl", "valid",
     'xs = [1, 2, 3,]\n'),
    ("collections/object-empty.hcl", "valid", 'm = {}\n'),
    ("collections/object-attrs.hcl", "valid",
     'm = {a = 1, b = 2}\n'),
    ("collections/object-newline-sep.hcl", "valid",
     'm = {\n  a = 1\n  b = 2\n}\n'),
    ("collections/object-quoted-keys.hcl", "valid",
     'm = {"a-b" = 1, "with space" = 2}\n'),
    ("collections/object-nested.hcl", "valid",
     'm = {outer = {inner = "value"}}\n'),
    ("collections/object-trailing-comma.hcl", "valid",
     'm = {a = 1, b = 2,}\n'),

    # ============================================================
    # VALID — blocks
    # ============================================================
    ("blocks/empty.hcl", "valid", 'service {}\n'),
    ("blocks/empty-newlines.hcl", "valid", 'service {\n}\n'),
    ("blocks/with-attrs.hcl", "valid",
     'service {\n  port = 8080\n  enabled = true\n}\n'),
    ("blocks/one-label.hcl", "valid",
     'resource "aws_instance" {\n  ami = "ami-123"\n}\n'),
    ("blocks/two-labels.hcl", "valid",
     'resource "aws_instance" "web" {\n  ami = "ami-123"\n}\n'),
    ("blocks/three-labels.hcl", "valid",
     'data "external" "named" "v2" {\n  arg = 1\n}\n'),
    ("blocks/nested.hcl", "valid",
     'outer {\n  inner {\n    leaf = 1\n  }\n}\n'),
    ("blocks/sibling.hcl", "valid",
     'a {\n  x = 1\n}\n\nb {\n  y = 2\n}\n'),
    ("blocks/mixed-attrs-blocks.hcl", "valid",
     'app {\n  name = "x"\n  config {\n    debug = true\n  }\n  port = 80\n}\n'),
    ("blocks/repeated.hcl", "valid",
     'tag {\n  key = "a"\n}\n\ntag {\n  key = "b"\n}\n'),

    # ============================================================
    # VALID — comments
    # ============================================================
    ("comments/hash-line.hcl", "valid",
     '# top comment\nfoo = 1\n'),
    ("comments/double-slash-line.hcl", "valid",
     '// top comment\nfoo = 1\n'),
    ("comments/inline-after-attr.hcl", "valid",
     'foo = 1 # trailing\n'),
    ("comments/block.hcl", "valid",
     '/* multi\n   line\n   comment */\nfoo = 1\n'),
    ("comments/block-inline.hcl", "valid",
     'foo = /* inline */ 1\n'),
    ("comments/empty-line-only.hcl", "valid", '#\nfoo = 1\n'),
    ("comments/many.hcl", "valid",
     '# one\n// two\n/* three */\nfoo = 1 // four\n'),

    # ============================================================
    # VALID — expressions
    # ============================================================
    ("expressions/arith-add.hcl", "valid", 'sum = 1 + 2\n'),
    ("expressions/arith-mixed.hcl", "valid",
     'x = 1 + 2 * 3 - 4 / 2 % 3\n'),
    ("expressions/arith-parens.hcl", "valid", 'x = (1 + 2) * 3\n'),
    ("expressions/unary-minus.hcl", "valid", 'x = -y\n'),
    ("expressions/unary-not.hcl", "valid", 'x = !y\n'),
    ("expressions/comparison.hcl", "valid",
     'eq = a == b\nlt = a < b\nge = a >= b\nne = a != b\n'),
    ("expressions/logical.hcl", "valid",
     'and = a && b\nor = a || b\n'),
    ("expressions/conditional.hcl", "valid",
     'x = a > 0 ? "pos" : "nonpos"\n'),
    ("expressions/conditional-nested.hcl", "valid",
     'x = a > 0 ? (b > 0 ? "++" : "+-") : (b > 0 ? "-+" : "--")\n'),
    ("expressions/precedence.hcl", "valid",
     'x = a && b || c && d\n'),
    ("expressions/concat-strings.hcl", "valid",
     'x = "hello " + "world"\n'),

    # ============================================================
    # VALID — templates
    # ============================================================
    ("templates/interp-simple.hcl", "valid",
     'msg = "hello ${name}"\n'),
    ("templates/interp-expr.hcl", "valid",
     'msg = "sum is ${a + b}"\n'),
    ("templates/interp-traversal.hcl", "valid",
     'msg = "value is ${obj.field}"\n'),
    ("templates/interp-nested.hcl", "valid",
     'msg = "outer ${"inner ${var}"}"\n'),
    ("templates/escaped-dollar.hcl", "valid",
     'msg = "literal $${not_interp}"\n'),
    ("templates/if-directive.hcl", "valid",
     'msg = "%{ if cond }yes%{ endif }"\n'),
    ("templates/if-else.hcl", "valid",
     'msg = "%{ if cond }yes%{ else }no%{ endif }"\n'),
    ("templates/for-directive.hcl", "valid",
     'msg = "%{ for x in xs }${x}, %{ endfor }"\n'),
    ("templates/strip-markers.hcl", "valid",
     'msg = <<EOT\n%{~ if cond ~}\nyes\n%{~ endif ~}\nEOT\n'),

    # ============================================================
    # VALID — traversals
    # ============================================================
    ("traversals/bare.hcl", "valid", 'x = foo\n'),
    ("traversals/attribute.hcl", "valid", 'x = foo.bar\n'),
    ("traversals/chained-attr.hcl", "valid",
     'x = foo.bar.baz.qux\n'),
    ("traversals/index-int.hcl", "valid", 'x = foo[0]\n'),
    ("traversals/index-string.hcl", "valid", 'x = foo["key"]\n'),
    ("traversals/index-expr.hcl", "valid", 'x = foo[i + 1]\n'),
    ("traversals/mixed-attr-index.hcl", "valid",
     'x = foo.bar[0].baz\n'),
    ("traversals/full-splat.hcl", "valid", 'x = foo[*].bar\n'),
    ("traversals/attr-splat.hcl", "valid", 'x = foo.*.bar\n'),

    # ============================================================
    # VALID — for expressions
    # ============================================================
    ("for/tuple-comp.hcl", "valid",
     'xs = [for x in ys : x + 1]\n'),
    ("for/tuple-comp-with-if.hcl", "valid",
     'xs = [for x in ys : x + 1 if x > 0]\n'),
    ("for/object-comp.hcl", "valid",
     'm = {for k, v in src : k => v + 1}\n'),
    ("for/object-comp-grouping.hcl", "valid",
     'm = {for k, v in src : k => v...}\n'),
    ("for/nested.hcl", "valid",
     'x = [for r in rows : [for c in r : c * 2]]\n'),
    ("for/with-index.hcl", "valid",
     'xs = [for i, v in src : "${i}=${v}"]\n'),

    # ============================================================
    # VALID — function calls
    # ============================================================
    ("functions/no-args.hcl", "valid", 'x = uuid()\n'),
    ("functions/one-arg.hcl", "valid",
     'x = upper("hello")\n'),
    ("functions/many-args.hcl", "valid",
     'x = format("%s-%d", name, n)\n'),
    ("functions/expand.hcl", "valid",
     'x = max(items...)\n'),
    ("functions/nested.hcl", "valid",
     'x = upper(trim(input, " "))\n'),
    ("functions/in-expr.hcl", "valid",
     'x = length(items) + 1\n'),

    # ============================================================
    # VALID — realistic mixes
    # ============================================================
    ("realistic/terraform-resource.hcl", "valid",
     'resource "aws_instance" "web" {\n'
     '  ami           = "ami-0123456789abcdef0"\n'
     '  instance_type = var.instance_type\n'
     '  count         = length(var.subnets)\n'
     '  tags = {\n'
     '    Name        = "web-${count.index}"\n'
     '    Environment = local.env\n'
     '  }\n'
     '}\n'),
    ("realistic/terraform-variable.hcl", "valid",
     'variable "region" {\n'
     '  type        = string\n'
     '  default     = "us-east-1"\n'
     '  description = "AWS region for resources"\n'
     '}\n'),
    ("realistic/terraform-locals.hcl", "valid",
     'locals {\n'
     '  env       = terraform.workspace\n'
     '  base_tags = {Owner = "team", Env = local.env}\n'
     '  subnets   = [for az in var.azs : "subnet-${az}"]\n'
     '}\n'),
    ("realistic/terraform-output.hcl", "valid",
     'output "instance_ips" {\n'
     '  value       = aws_instance.web[*].public_ip\n'
     '  description = "Public IPs of web instances"\n'
     '  sensitive   = false\n'
     '}\n'),
    ("realistic/packer-template.hcl", "valid",
     'source "amazon-ebs" "linux" {\n'
     '  region        = "us-east-1"\n'
     '  source_ami    = data.amazon-ami.linux.id\n'
     '  instance_type = "t3.micro"\n'
     '  ssh_username  = "ec2-user"\n'
     '  ami_name      = "myami-${formatdate("YYYY-MM-DD", timestamp())}"\n'
     '}\n'
     '\n'
     'build {\n'
     '  sources = ["source.amazon-ebs.linux"]\n'
     '}\n'),
    ("realistic/empty-file.hcl", "valid", ""),
    ("realistic/whitespace-only.hcl", "valid", "\n\n   \n\t\n\n"),
    ("realistic/comment-only.hcl", "valid",
     '# only comments here\n// also this\n/* and this */\n'),

    # ============================================================
    # INVALID — strings
    # ============================================================
    ("strings/unterminated-quoted.hcl", "invalid",
     'foo = "bar\n'),
    ("strings/unterminated-heredoc.hcl", "invalid",
     'foo = <<EOT\nbody but no terminator\n'),
    ("strings/heredoc-bad-terminator.hcl", "invalid",
     'foo = <<EOT\nbody\n  EOT (with garbage)\n'),
    ("strings/raw-newline-in-quoted.hcl", "invalid",
     'foo = "line1\nline2"\n'),
    ("strings/lone-backslash.hcl", "invalid",
     'foo = "trailing \\"\n'),

    # ============================================================
    # INVALID — collections
    # ============================================================
    ("collections/tuple-missing-bracket.hcl", "invalid",
     'xs = [1, 2, 3\n'),
    ("collections/tuple-double-comma.hcl", "invalid",
     'xs = [1,, 2]\n'),
    ("collections/tuple-leading-comma.hcl", "invalid",
     'xs = [, 1, 2]\n'),
    ("collections/object-missing-brace.hcl", "invalid",
     'm = {a = 1\n'),
    ("collections/object-missing-equals.hcl", "invalid",
     'm = {a 1, b 2}\n'),
    ("collections/object-empty-key.hcl", "invalid",
     'm = {= "no_key"}\n'),
    ("collections/tuple-bad-separator.hcl", "invalid",
     'xs = [1; 2; 3]\n'),

    # ============================================================
    # INVALID — blocks
    # ============================================================
    ("blocks/unclosed.hcl", "invalid",
     'service {\n  port = 8080\n'),
    ("blocks/extra-close.hcl", "invalid",
     'service {\n  port = 8080\n}\n}\n'),
    ("blocks/bare-expr-between-attrs.hcl", "invalid",
     'service {\n  port = 8080\n  1 + 2\n  enabled = true\n}\n'),
    ("blocks/bad-label-syntax.hcl", "invalid",
     'resource = "aws_instance" "web" {\n  ami = "x"\n}\n'),
    ("blocks/missing-equals.hcl", "invalid",
     'service {\n  port 8080\n}\n'),
    ("blocks/lone-equals.hcl", "invalid", 'foo =\n'),
    ("blocks/lone-rhs.hcl", "invalid", '= 1\n'),

    # ============================================================
    # INVALID — expressions
    # ============================================================
    ("expressions/dangling-add.hcl", "invalid", 'x = 1 +\n'),
    ("expressions/dangling-mul.hcl", "invalid", 'x = 1 *\n'),
    ("expressions/dangling-comparison.hcl", "invalid",
     'x = a ==\n'),
    ("expressions/double-operator.hcl", "invalid", 'x = 1 + + 2\n'),
    ("expressions/unmatched-paren-open.hcl", "invalid",
     'x = (1 + 2\n'),
    ("expressions/unmatched-paren-close.hcl", "invalid",
     'x = 1 + 2)\n'),
    ("expressions/empty-parens.hcl", "invalid", 'x = ()\n'),
    ("expressions/conditional-no-false.hcl", "invalid",
     'x = a > 0 ? 1\n'),
    ("expressions/conditional-no-question.hcl", "invalid",
     'x = a > 0 : 1 : 2\n'),
    ("expressions/bare-operator.hcl", "invalid", 'x = +\n'),
    ("expressions/bad-number.hcl", "invalid", 'x = 1.2.3\n'),

    # ============================================================
    # INVALID — templates
    # ============================================================
    ("templates/unterminated-interp.hcl", "invalid",
     'msg = "hello ${name"\n'),
    ("templates/empty-interp.hcl", "invalid",
     'msg = "x ${} y"\n'),
    ("templates/unterminated-directive.hcl", "invalid",
     'msg = "%{ if cond }body"\n'),
    ("templates/missing-endif.hcl", "invalid",
     'msg = "%{ if a }x%{ if b }y%{ endif }"\n'),
    ("templates/bad-directive-name.hcl", "invalid",
     'msg = "%{ wat foo }x%{ endwat }"\n'),
    ("templates/lone-percent-brace.hcl", "invalid",
     'msg = "say %{"\n'),

    # ============================================================
    # INVALID — traversals
    # ============================================================
    ("traversals/dangling-dot.hcl", "invalid", 'x = foo.\n'),
    ("traversals/empty-index.hcl", "invalid", 'x = foo[]\n'),
    ("traversals/unmatched-index.hcl", "invalid", 'x = foo[0\n'),
    ("traversals/double-dot.hcl", "invalid", 'x = foo..bar\n'),
    ("traversals/index-no-expr.hcl", "invalid", 'x = foo[,]\n'),

    # ============================================================
    # INVALID — for-expressions
    # ============================================================
    ("for/missing-in.hcl", "invalid",
     'xs = [for x ys : x]\n'),
    ("for/missing-colon.hcl", "invalid",
     'xs = [for x in ys x]\n'),
    ("for/object-missing-arrow.hcl", "invalid",
     'm = {for k, v in src : k v}\n'),

    # ============================================================
    # INVALID — comments / misc
    # ============================================================
    ("misc/unterminated-block-comment.hcl", "invalid",
     'foo = 1\n/* never closes\n'),
    ("misc/garbage-tokens.hcl", "invalid", '@#$%^&*\n'),
    ("misc/lone-keyword.hcl", "invalid", 'true\n'),
    ("misc/bare-string.hcl", "invalid", '"just a string"\n'),
    ("misc/identifier-leading-digit.hcl", "invalid", '2foo = 1\n'),
    ("misc/duplicate-equals.hcl", "invalid", 'foo == 1\n'),
]


def materialize(check_only: bool) -> list[str]:
    """Write each TESTS entry to disk under corpus/<verdict>/<subpath>
    and run the oracle on each to confirm intended verdict matches reality.
    Returns the list of drift reports (oracle disagreed with placement).

    When check_only=True, skip the write step but still verify each
    expected file exists and the oracle agrees with its placement.
    """
    corpus_dir = HERE / "corpus"
    if not check_only:
        # Wipe the tree so removed entries don't leave stale files behind.
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
            target.write_text(content)
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
        print("Build it with: (cd scripts/oracles/hcl-check && go build -o hcl-check .)",
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
