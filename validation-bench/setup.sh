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

# Generate tests.jsonl and symlink test data for each TOML task
generate_task() {
    local task="$1"        # e.g. toml-1.0-cpp
    local file_list="$2"   # e.g. files-toml-1.0.0
    local task_dir="tasks/$task"

    echo "Setting up $task ..."

    # Symlink tests/ -> cached toml-test/tests/
    local link="$task_dir/tests"
    local target="../../$CACHE_DIR/tests"
    rm -f "$link"
    ln -s "$target" "$link"

    # Generate tests.jsonl from the file list
    local list_file="$CACHE_DIR/tests/$file_list"
    local out="$task_dir/tests.jsonl"
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

generate_task "toml-1.0-cpp" "files-toml-1.0.0"
generate_task "toml-1.1-cpp" "files-toml-1.1.0"

echo "Done."
