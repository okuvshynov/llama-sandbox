#!/usr/bin/env python3
"""The checks of this repository as a few fixed runs (docs/testing.md, "The runs"; README step 186).

  scripts/check.py unit        no model files, under a minute: after every change

To come (docs/plan.md, "The testing process"): default (the model, one load, 5-10 minutes: after
every change), speed (the held-out prompts' tokens per second), complete (everything, overnight).
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

HOME = os.path.expanduser('~')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_REF = HOME + '/venvs/ds-ref/bin/python'               # torch: the reference's own top-k (ref/check_topk.py)
ENCODING = HOME + '/llms/ds-flash/encoding/encoding.py'  # the model's chat format, which the server test renders with

# (name, command, the files outside the repository it needs, what to run next if it fails)
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
RUNS = {'unit': UNIT}


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
    if len(sys.argv) != 2 or sys.argv[1] not in RUNS:
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
    for i, (name, cmd, needs, then) in enumerate(RUNS[run]):
        log = f'{out}/{i + 1:02d}-{name.split(":")[0].replace(" ", "-")}.txt'
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
        print(f'{"ok  " if rc == 0 else "FAIL"}  {name}  {time.time() - t0:.1f} s', flush=True)
        if rc != 0:
            failed += 1
            print(f'      {"missing: " + ", ".join(missing) if missing else "exit code " + str(rc)}; output: {log}')
            print(f'      next: {then}', flush=True)
    print(f'{"FAILED: " + str(failed) + " of " + str(len(RUNS[run])) + " checks" if failed else "all checks passed"} ({time.time() - t_all:.0f} s)')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
