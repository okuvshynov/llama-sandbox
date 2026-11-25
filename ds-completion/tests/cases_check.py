#!/usr/bin/env python3
"""Replies of the product against stored token ids, as a table of cases; one engine per
configuration (README step 205).

  python3 tests/cases_check.py default|complete [--store] [--out DIR]        exit code 1 if a case fails

A case is a line of CASES: a run (default or complete; complete takes default's too), a
configuration of tests/engine.py, a key, the request's fields and a prompt. All cases of one
configuration go to one engine, one after another: the engine starts once (~13 s of load and the
registration of the expert tensors, 40-285 s) where a process per case started once per case.
With --canon a reply does not depend on what the engine answered before.

A case's reply must be the token ids stored under its key in tests/data/cases/expected.json. Ids
are stored, not a hash: a case that fails says which token differs first. Where a case names a
reference (the stored record of a single run of the same prompt and settings in a fresh process),
the ids' hash must also be that record's output.hash: this is what ties the replies of one long
engine to the single runs, and --store refuses a reply that is not its reference's.
An engine that dies takes its case with it, not the rest: it is started again for the next case.
An engine must also exit with code 0 when its input ends.

To add a check: a line in CASES (and, for settings a request cannot carry, a configuration in
tests/engine.py), then --store once and a look at what was stored.
"""
import json
import os
import sys
import time

import engine
from engine import ROOT

EXPECTED = 'tests/data/cases/expected.json'
S1 = 'temp=1 top_p=0.95 seed=1'


def record_hash(path):
    """a reference: output.hash of a stored --record file"""
    return lambda: json.load(open(path))['output']['hash']


# (run, configuration, key, the request's fields after n=, prompt, reference or None)
#   prompt: ('heldout', name): the held-out prompt of that name, rendered, with its own n
CASES = []
# The eight held-out prompts sampled with one seed, with speculation and the cache: what
# results/sampler/spec-heldout.matrix ran as eight processes (steps 150-151). Its eight greedy runs are
# tests/outputs_check.py's (default) and the speed check's.
for _name in engine.heldout():
    CASES.append(('complete', 'full', f'full/s1/{_name}', S1, ('heldout', _name),
                  record_hash(f'results/sampler/step151/spec-heldout/s1-{_name}-1.json')))


def main():
    a = sys.argv[1:]
    store = '--store' in a
    if not a or a[0] not in ('default', 'complete'):
        print(__doc__.split('\n\n')[1], file=sys.stderr)
        return 2
    runs = ('default', 'complete') if a[0] == 'complete' else ('default',)
    os.chdir(ROOT)
    out = a[a.index('--out') + 1] if '--out' in a else f'data/check/cases-{a[0]}-{time.strftime("%Y%m%d-%H%M%S")}'
    os.makedirs(out, exist_ok=True)
    cases = [c for c in CASES if c[0] in runs]
    prompts = engine.heldout()
    want = json.load(open(EXPECTED)) if os.path.exists(EXPECTED) else {}
    got, failed = {}, 0

    def check(ok, what):
        nonlocal failed
        failed += not ok
        print(('ok    ' if ok else 'FAIL  ') + what, flush=True)

    for config in dict.fromkeys(c[1] for c in cases):  # (in the order of their first case)
        eng, starts = None, 0
        for _, _, key, fields, prompt, ref in [c for c in cases if c[1] == config]:
            if eng is None or not eng.alive():
                if starts == 3:
                    check(False, f'{key}: not run: the engine "{config}" was started three times')
                    continue
                starts += 1
                eng = engine.Engine(config, f'{out}/{config}-{starts}.err')
                print(f'# engine "{config}"{" again" if starts > 1 else ""}: {eng.command()}\n# its own lines: {out}/{config}-{starts}.err', flush=True)
                print(f'# {"ready" if eng.ready else "NOT ready"} after {eng.load_s:.0f} s', flush=True)
                if not eng.ready:
                    check(False, f'{key}: the engine "{config}" did not start')
                    continue
            kind, name = prompt
            assert kind == 'heldout', prompt
            mode, n, text = prompts[name]
            r = eng.request(f'gen n={n} {fields}'.rstrip(), engine.render(mode, text))
            if r is None:
                check(False, f'{key}: no reply: the engine said "{eng.said}"' + ('' if eng.alive() else f' and is gone (exit code {eng.p.wait()})'))
                continue
            h = engine.fnv1a64_ids(r.ids)
            what = f'{key:24s} {len(r.ids):3d} tokens {h}; {len(r.ids) / max(float(r.end["gen_s"]), 1e-9):5.1f} t/s'
            if ref is not None and h != ref():  # (not stored either: --store keeps the old ids)
                w = want.get(key) or []
                k = next((i for i, (x, y) in enumerate(zip(r.ids, w)) if x != y), min(len(r.ids), len(w)))
                check(False, f'{what}: not the single run\'s {ref()}' + (f'; it differs from the stored ids at token {k} of {len(w)}' if w else ''))
                continue
            got[key] = r.ids
            w = want.get(key)
            if w == r.ids or (store and w is None):
                check(True, what + ('' if ref is None else ', the single run\'s'))
            elif w is None:
                check(False, f'{what}: nothing is expected for "{key}" in {EXPECTED} (--store writes it)')
            else:
                k = next((i for i, (x, y) in enumerate(zip(r.ids, w)) if x != y), min(len(r.ids), len(w)))
                check(store, f'{what}: {"replaces" if store else "differs from"} what is expected at token {k} of {len(w)} '
                             f'(got {r.ids[k] if k < len(r.ids) else "the end"}, expected {w[k] if k < len(w) else "the end"})')
        if eng is not None and eng.alive():
            rc = eng.close()
            check(rc == 0, f'the engine "{config}" exits with code {rc} when its input ends')

    if store and not failed:
        os.makedirs(os.path.dirname(EXPECTED), exist_ok=True)
        new = dict(want, **got)  # (the cases of the other run stay)
        changed = [k for k in got if want.get(k) != got[k]]
        with open(EXPECTED, 'w') as f:
            f.write('{\n' + ',\n'.join(f'  {json.dumps(k)}: {json.dumps(new[k])}' for k in sorted(new)) + '\n}\n')
        print(f'# {EXPECTED} written: {len(new)} replies, {len(changed)} new or changed ({", ".join(changed) or "none"})')
    elif failed:
        json.dump(got, open(f'{out}/got.json', 'w'))
        print(f'# the replies\' ids as the engine gave them: {out}/got.json')
        print('# next: the case alone in a fresh process (bin-hip/ds-completion with the configuration\'s arguments and -p, QUICKSTART "Run");\n'
              '#   if that gives the stored ids, the difference is in what the engine kept from the requests before it')
    print('FAILED' if failed else 'all checks passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
