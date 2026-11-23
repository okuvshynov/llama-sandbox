#!/usr/bin/env python3
"""The checks of this repository as a few fixed runs (docs/testing.md, "The runs"; README step 186).

  scripts/check.py unit        no model files, under a minute
  scripts/check.py default     unit, then the checks with the model: after every change
  scripts/check.py default --store    the same, and what the checks wrote becomes what is expected

default has two halves after unit. The arithmetic (README step 187): the logits of the runtime
and of the CPU reference path on golden token sequences (12, 202 and 1,102 tokens; tests/e2e/),
each file's sha256 against tests/e2e/expected.txt. And what stands around the arithmetic in the
product (step 188; tests/outputs_check.py): one engine with the fastest settings takes the
held-out prompts, sampled ones and a second turn, then one fresh interactive session; every
reply's token ids against tests/outputs/expected.json.
The runs speed and complete are to come (docs/plan.md, "The testing process").
--store is for a change that is meant to move the logits or the outputs (a toolchain, the
numerics): only after the difference was compared with the reference (ref/compare_logits.py).
It prints old and new; a greedy reply that is not the project's reference is never stored.
What says where a difference is (ds-test's per-layer table, traces) is not part of any run: a
failed check names the command to run next.

A run is a fixed list: there is nothing to choose. It first builds everything (make, make HIP=1:
a stale program is a wrong verdict), then runs each check and prints one line per check:

  ok    <name>  <seconds>
  FAIL  <name>  <seconds>   and where its output is, and what to run next

and "all checks passed" or "FAILED" at the end. Exit code 1 if the build or a check failed.
Every check's output is kept in data/check/<run>-<time>/ (data/ is not in git).
"""
import os
import subprocess
import sys
import time

import hashlib

HOME = os.path.expanduser('~')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_REF = HOME + '/venvs/ds-ref/bin/python'               # torch: the reference's own top-k (ref/check_topk.py)
ENCODING = HOME + '/llms/ds-flash/encoding/encoding.py'  # the model's chat format, which the server test renders with

MODEL = HOME + '/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf'
F8 = HOME + '/llms/ds-1017-fp8/dense-fp8.gguf'
DRAFT = HOME + '/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf'
EXPECTED = 'tests/e2e/expected.txt'  # lines: <key> <sha256>


def logits(path, name):
    """ds-test --e2e on a cut golden set (ref/trim_golden.py): the logits of every step, as bytes"""
    rt = path == 'runtime'
    key = f'{path}/{name}'
    cmd = ['bin-hip/ds-test' if rt else 'bin/ds-test', '-m', MODEL, '-g', f'tests/e2e/{name}'] + (['--runtime', F8] if rt else []) + ['--e2e', '{out}']
    full = f'data/golden/{name}'
    then = (f'how far from the reference: {PY_REF.replace(HOME, "~")} ref/compare_logits.py data/ref/{name}.pt <the file>; '
            f'which layer: {cmd[0]} -m $M -g {full}' + (' --runtime $F8' if rt else '') + ' against the same from the last good commit')
    return (f'logits, {"the runtime" if rt else "the CPU reference path"}, {name}', cmd, [MODEL] + ([F8] if rt else []), then, key)


# (name, command, the files outside the repository it needs, what to run next if it fails[, the key in
# EXPECTED of the file the command writes at {out}])
UNIT = [
    ('fakequant: the rounding ops against the reference\'s, bit for bit', ['bin/test-fakequant'], [],
     'its output names the first values that differ; the functions are in src/fakequant_graph.cpp'),
    ('sampler: temperature, top-p, one seed one sequence', ['bin/test-sampler'], [],
     'its output names the check; src/sampler.cpp'),
    ('top-k: the selection kernels against torch and the reference', [PY_REF, 'ref/check_topk.py'], [PY_REF],
     'its output names the case; hip/kernels.hip (k_topk, k_candidate_blocks)'),
    ('kernels: M tokens per launch against one, bit for bit', ['bin-hip/bench-kernels', '--bitwise'], [],
     'its output names the kernel; hip/kernels.hip'),
    ('server: ds_server.py against a fake engine', [sys.executable, 'server/test_server.py'], [ENCODING],
     'its output names the check; server/ds_server.py'),
]
DEFAULT = UNIT + [
    logits('runtime', 'hello-fp32'), logits('runtime', 'readme200'), logits('runtime', 'readme1100'),
    logits('cpu', 'hello-fp32'), logits('cpu', 'readme200'),
    ('outputs: the product in one load, and one session', [sys.executable, 'tests/outputs_check.py'], [MODEL, F8, DRAFT],
     'its output names the reply and the first token that differs, and says how to narrow it down'),
]
RUNS = {'unit': UNIT, 'default': DEFAULT}


def expected():
    return dict(l.split() for l in open(EXPECTED) if l.strip() and not l.startswith('#')) if os.path.exists(EXPECTED) else {}


def sclk():
    out = []
    for c in range(4):
        try:
            star = [l for l in open(f'/sys/class/drm/card{c}/device/pp_dpm_sclk') if '*' in l]
            out.append(star[0].split(':')[0].strip() if star else '?')
        except OSError:
            out.append('-')
    return ','.join(out)


def main():
    store = '--store' in sys.argv[2:]
    if len(sys.argv) < 2 or sys.argv[1] not in RUNS or [a for a in sys.argv[2:] if a != '--store']:
        print(__doc__.split('\n\n')[1], file=sys.stderr)
        return 2
    run = sys.argv[1]
    os.chdir(ROOT)
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', '.']).returncode != 0
    out = f'data/check/{run}-{time.strftime("%Y%m%d-%H%M%S")}'
    os.makedirs(out)
    print(f'# scripts/check.py {run}; commit {commit}{"+changes" if dirty else ""}; sclk {sclk()}; {time.strftime("%Y-%m-%d %H:%M")}; outputs in {out}/', flush=True)

    t_all = time.time()
    for name, cmd in (('make', ['make', '-j16']), ('make HIP=1', ['make', '-j16', 'HIP=1'])):
        t0 = time.time()
        log = f'{out}/build-{"hip" if "HIP=1" in cmd else "cpu"}.txt'
        rc = subprocess.run(cmd, stdout=open(log, 'w'), stderr=subprocess.STDOUT).returncode
        print(f'{"ok  " if rc == 0 else "FAIL"}  build: {name}  {time.time() - t0:.1f} s' + ('' if rc == 0 else f'   see {log}'), flush=True)
        if rc != 0:
            print('FAILED (the build: no check was run)')
            return 1

    failed = 0
    want, stored = expected(), {}
    for i, (name, cmd, needs, then, *key) in enumerate(RUNS[run]):
        log = f'{out}/{i + 1:02d}-{name.split(":")[0].replace(" ", "-").replace(",", "")}.txt'
        made = log[:-4] + '.f32'
        cmd = [c.replace('{out}', made) for c in cmd] + (['--store'] if store and cmd[-1].endswith('outputs_check.py') else [])
        missing = [f for f in needs if not os.path.exists(f)]
        t0 = time.time()
        if missing:  # a check that cannot run has not passed
            open(log, 'w').write('missing: ' + ', '.join(missing) + '\n')
            rc = 1
        else:
            with open(log, 'w') as f:
                f.write('# ' + ' '.join(c.replace(HOME, '~') for c in cmd) + '\n')
                f.flush()
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        why = "missing: " + ", ".join(missing) if missing else "exit code " + str(rc)
        if rc == 0 and key:  # the file the check wrote, against the expected hash
            got = hashlib.sha256(open(made, 'rb').read()).hexdigest()
            stored[key[0]] = got
            if want.get(key[0]) != got and not store:
                rc, why = 1, f'sha256 {got[:16]}..., expected {(want.get(key[0]) or "nothing stored")[:16]}... ({EXPECTED}); the file is kept: {made}'
            elif want.get(key[0]) != got:
                print(f'      stored: {key[0]} {(want.get(key[0]) or "-")[:16]}... -> {got[:16]}...')
            if rc == 0:
                os.remove(made)  # (kept only when it is not what was expected)
        print(f'{"ok  " if rc == 0 else "FAIL"}  {name}  {time.time() - t0:.1f} s', flush=True)
        if rc != 0:
            failed += 1
            print(f'      {why}')
            print(f'      output: {log}')
            print(f'      next: {then}', flush=True)
    if store and not failed and stored:
        want.update(stored)
        with open(EXPECTED, 'w') as f:
            f.write('# The sha256 of what each check of scripts/check.py writes: ds-test --e2e logits, [steps][n_vocab] f32,\n'
                    '# 16 threads, this toolchain (ROCm 6.3.4, the compilers of docs/reproduce.md). Written by check.py --store.\n')
            for k in sorted(want):
                f.write(f'{k} {want[k]}\n')
        print(f'      {EXPECTED} written: {len(want)} entries')
    print(f'{"FAILED: " + str(failed) + " of " + str(len(RUNS[run])) + " checks" if failed else "all checks passed"} ({time.time() - t_all:.0f} s)')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
