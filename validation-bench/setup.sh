#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/toml-lang/toml-test.git"
PINNED_COMMIT="0ee318ae97ae5dec5f74aeccafbdc75f435580e2"  # Release 2.1
CACHE_DIR=".cache/toml-test"

cd "$(dirname "$0")"

# Clone or update cached repo
if [ -d "$CACHE_DIR/.git" ]; then
    echo "toml-test already cloned in $CACHE_DIR"
    current=$(git -C "$CACHE_DIR" rev-parse HEAD)
    if [ "$current" != "$PINNED_COMMIT" ]; then
        echo "Checking out pinned commit $PINNED_COMMIT ..."
        git -C "$CACHE_DIR" fetch origin
        git -C "$CACHE_DIR" checkout "$PINNED_COMMIT"
    fi
else
    echo "Cloning toml-test ..."
    mkdir -p "$(dirname "$CACHE_DIR")"
    git clone "$REPO_URL" "$CACHE_DIR"
    git -C "$CACHE_DIR" checkout "$PINNED_COMMIT"
fi

# Generate tests.jsonl + tests/ symlink under specs/<spec>/. Tests are a
# property of the spec, not the (spec, env) cell — every env that consumes
# a spec reads from the same location, so generation runs once per spec.
generate_upstream_spec() {
    local spec="$1"        # e.g. toml-1.0
    local file_list="$2"   # e.g. files-toml-1.0.0 inside the upstream tests/ dir
    local spec_dir="data/specs/$spec"

    echo "Setting up $spec ..."

    # Symlink tests/ -> cached toml-test/tests/
    local link="$spec_dir/tests"
    local target="../../../$CACHE_DIR/tests"
    rm -rf "$link"
    ln -s "$target" "$link"

    # Generate tests.jsonl from the file list
    local list_file="$CACHE_DIR/tests/$file_list"
    local out="$spec_dir/tests.jsonl"
    python3 -c "
import json, sys

with open('$list_file') as f:
    lines = [l.strip() for l in f if l.strip()]

tests = []
for path in lines:
    # Skip .json expected-output files (only want .toml inputs)
    if not path.endswith('.toml'):
        continue
    if path.startswith('valid/'):
        expected = 'valid'
    elif path.startswith('invalid/'):
        expected = 'invalid'
    else:
        continue
    test_id = path.removesuffix('.toml')
    label = test_id.replace('/', ': ', 1)
    tests.append({
        'id': test_id,
        'input_file': 'tests/' + path,
        'expected': expected,
        'label': label,
    })

with open('$out', 'w') as f:
    for t in tests:
        f.write(json.dumps(t, ensure_ascii=False) + '\n')

print(f'  {len(tests)} test cases written to $out')
"
}

generate_upstream_spec "toml-1.0"        "files-toml-1.0.0"
generate_upstream_spec "toml-1.0-nospec" "files-toml-1.0.0"
generate_upstream_spec "toml-1.1"        "files-toml-1.1.0"
generate_upstream_spec "toml-1.1-nospec" "files-toml-1.1.0"

# Hand-curated corpus specs: tests/ is materialized from specs/<spec>/corpus/,
# and labels are re-derived locally by running an oracle on each file. Setup
# fails loudly if a corpus file's directory class disagrees with the oracle's
# verdict — protects against label drift in checked-in test data.
generate_corpus_spec() {
    local spec="$1"          # e.g. lua-5.4
    local oracle_cmd="$2"    # e.g. "luac5.4 -p"
    local extension="$3"     # e.g. "lua"
    local spec_dir="data/specs/$spec"
    local corpus_dir="$spec_dir/corpus"
    local tests_dir="$spec_dir/tests"
    local out="$spec_dir/tests.jsonl"

    echo "Setting up $spec ..."

    if [ ! -d "$corpus_dir" ]; then
        echo "  ERROR: missing corpus dir: $corpus_dir" >&2
        return 1
    fi

    local oracle_bin
    oracle_bin="$(echo "$oracle_cmd" | awk '{print $1}')"
    if ! command -v "$oracle_bin" >/dev/null 2>&1; then
        echo "  ERROR: oracle '$oracle_bin' not found on PATH; install it first" >&2
        return 1
    fi

    rm -rf "$tests_dir"
    mkdir -p "$tests_dir/valid" "$tests_dir/invalid"

    : > "$out"
    local n_total=0 n_mismatch=0
    while IFS= read -r src; do
        local rel="${src#"$corpus_dir/"}"
        local class="${rel%%/*}"
        if [ "$class" != "valid" ] && [ "$class" != "invalid" ]; then
            continue
        fi
        local name="${rel#*/}"
        name="${name%.$extension}"
        local dest="$tests_dir/$class/$name.$extension"
        mkdir -p "$(dirname "$dest")"
        cp "$src" "$dest"

        local actual
        if $oracle_cmd "$src" >/dev/null 2>&1; then
            actual="valid"
        else
            actual="invalid"
        fi

        if [ "$actual" != "$class" ]; then
            echo "  MISMATCH: $rel is in corpus/$class/ but oracle says '$actual'" >&2
            n_mismatch=$((n_mismatch + 1))
            continue
        fi

        local test_id="$class/$name"
        local label="$class: $name"
        printf '{"id": "%s", "input_file": "tests/%s.%s", "expected": "%s", "label": "%s"}\n' \
            "$test_id" "$test_id" "$extension" "$class" "$label" >> "$out"
        n_total=$((n_total + 1))
    done < <(find "$corpus_dir" -type f -name "*.$extension" | sort)

    if [ "$n_mismatch" -gt 0 ]; then
        echo "  $n_mismatch mismatch(es); aborting" >&2
        return 1
    fi
    echo "  $n_total test cases written to $out (oracle: $oracle_cmd)"
}

generate_corpus_spec "lua-5.4"    "luac5.4 -p"                              "lua"
generate_corpus_spec "palindrome" "python3 scripts/oracles/palindrome-check.py" "txt"

# yaml-test-suite uses a per-directory test layout (data branch): each test
# is a directory containing in.yaml plus auxiliary files; presence of an
# `error` file means the input is invalid. This shape doesn't match toml-test
# (flat file list with valid/invalid prefix) or the hand-curated corpus
# pattern, so it gets its own generator.
YAML_REPO_URL="https://github.com/yaml/yaml-test-suite.git"
YAML_PINNED_COMMIT="6ad3d2c62885d82fc349026c136ef560838fdf3d"  # data branch HEAD as of 2022-01-17
YAML_CACHE_DIR=".cache/yaml-test-suite"

generate_yaml_test_suite_spec() {
    local spec="$1"        # e.g. yaml-1.2
    local spec_dir="data/specs/$spec"

    echo "Setting up $spec ..."

    if [ -d "$YAML_CACHE_DIR/.git" ]; then
        local current
        current=$(git -C "$YAML_CACHE_DIR" rev-parse HEAD)
        if [ "$current" != "$YAML_PINNED_COMMIT" ]; then
            echo "  Updating $YAML_CACHE_DIR to $YAML_PINNED_COMMIT ..."
            git -C "$YAML_CACHE_DIR" fetch origin data
            git -C "$YAML_CACHE_DIR" checkout "$YAML_PINNED_COMMIT"
        fi
    else
        echo "  Cloning yaml-test-suite (data branch) ..."
        mkdir -p "$(dirname "$YAML_CACHE_DIR")"
        git clone --branch data --single-branch "$YAML_REPO_URL" "$YAML_CACHE_DIR"
        git -C "$YAML_CACHE_DIR" checkout "$YAML_PINNED_COMMIT"
    fi

    # Symlink tests/ -> the cache root so input_file paths in tests.jsonl
    # resolve uniformly via spec_dir/tests/<test_id>/in.yaml.
    local link="$spec_dir/tests"
    local target="../../../$YAML_CACHE_DIR"
    rm -rf "$link"
    ln -s "$target" "$link"

    local out="$spec_dir/tests.jsonl"
    python3 -c "
import json, os, sys

cache = '$YAML_CACHE_DIR'

# Tests symlinked under tags/1.3-err and tags/1.3-mod are errors-only-in-1.3
# or modified-in-1.3; they reflect YAML 1.3 semantics, not 1.2. The 'data'
# branch is pre-1.3, so those test IDs still appear in the top-level tree;
# we filter them out so labels match YAML 1.2 ground truth.
skip_ids = set()
for tag_subdir in ('tags/1.3-err', 'tags/1.3-mod'):
    p = os.path.join(cache, tag_subdir)
    if not os.path.isdir(p):
        continue
    # Each entry under tags/<category>/ is a symlink whose name is the test ID.
    for entry in os.listdir(p):
        skip_ids.add(entry)

# Walk every directory that contains an in.yaml. Some test IDs have variant
# subdirs (e.g. Y79Y/000..010), each a separate test case.
def walk():
    for root, dirs, files in os.walk(cache):
        # Skip git internals and the tags/ categorization layer.
        rel = os.path.relpath(root, cache)
        if rel == '.':
            dirs[:] = [d for d in dirs if d not in ('.git', 'tags')]
            continue
        if 'in.yaml' in files:
            yield root

skipped_1p3 = 0
tests = []
for test_dir in sorted(walk()):
    rel = os.path.relpath(test_dir, cache)
    # rel is either '<id>' or '<id>/<variant>' (variant is digits like 00, 000)
    parts = rel.split('/')
    test_id_root = parts[0]
    if test_id_root in skip_ids:
        skipped_1p3 += 1
        continue
    test_id = rel  # use the full relative path as the id; variants get '<id>/<n>'
    is_invalid = os.path.exists(os.path.join(test_dir, 'error'))
    expected = 'invalid' if is_invalid else 'valid'

    # Human-readable label: prefer the === file's first line if present.
    name_path = os.path.join(test_dir, '===')
    if os.path.exists(name_path):
        with open(name_path) as f:
            name = f.readline().rstrip('\\n').strip()
    else:
        name = test_id
    label = f'{expected}: {test_id} ({name})' if name else f'{expected}: {test_id}'

    tests.append({
        'id': test_id,
        'input_file': f'tests/{rel}/in.yaml',
        'expected': expected,
        'label': label,
    })

with open('$out', 'w') as f:
    for t in tests:
        f.write(json.dumps(t, ensure_ascii=False) + '\n')

n_valid = sum(1 for t in tests if t['expected'] == 'valid')
n_invalid = sum(1 for t in tests if t['expected'] == 'invalid')
print(f'  {len(tests)} test cases written to $out '
      f'({n_valid} valid, {n_invalid} invalid; '
      f'skipped {skipped_1p3} that are 1.3-only)')
"
}

generate_yaml_test_suite_spec "yaml-1.2"

echo "Done."
