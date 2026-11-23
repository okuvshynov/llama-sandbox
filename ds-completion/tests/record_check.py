#!/usr/bin/env python3
"""One run of a program whose output's hash is known (README step 190; for `scripts/check.py complete`).

  python3 tests/record_check.py fnv1a64:<hash> COMMAND...        exit code 1 if the hash differs

Runs COMMAND with --record FILE added (docs/record.md) and compares the record's output.hash,
the FNV-1a of the generated token ids, with the one given.
"""
import json
import os
import subprocess
import sys
import tempfile

want, cmd = sys.argv[1], [os.path.expanduser(c) for c in sys.argv[2:]]
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs('data/check', exist_ok=True)
rec = tempfile.NamedTemporaryFile(prefix='record_check-', suffix='.json', dir='data/check', delete=False).name
rc = subprocess.run(cmd + ['--record', rec], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
try:
    r = json.load(open(rec))
except Exception:
    r = None
got = r['output']['hash'] if r else 'no record'
ok = rc == 0 and got == want
print(f'{"ok    " if ok else "FAIL  "}{got}' + ('' if ok else f', expected {want} (exit code {rc}; the record: {rec})') + (f'; {r["counters"]["gen_tokens"]} tokens, prompt {r["counters"].get("prompt_tps", 0):.1f} t/s, generation {r["counters"]["gen_tps"]:.2f} t/s' if r else ''))
if ok:
    os.remove(rec)
sys.exit(0 if ok else 1)
