#!/usr/bin/env python3
"""Replies of the product against stored token ids, as a table of cases; one engine per
configuration (README steps 205, 206).

  python3 tests/cases_check.py default|complete [--store] [--only KEY-PREFIX] [--out DIR]        exit code 1 if a case fails

The logits checks (ds-test --e2e) cover the arithmetic. This covers what stands around it in the
product: prompt chunks, speculative decoding with the draft, the adaptive expert cache, sampling,
a context kept between requests. With --canon none of them may change a token.

A case is a line of CASES: a run (default or complete; complete takes default's too), a
configuration of tests/engine.py, a key, a prompt, the request's fields. All cases of one
configuration go to one engine, one after another, in the table's order: the engine starts once
(~13 s of load and the registration of the expert tensors, 40-285 s) where a process per case
started once per case. With --canon a reply does not depend on what the engine answered before.

A case's reply must be the token ids stored under its key in tests/data/cases/expected.json. Ids
are stored, not a hash: a case that fails says which token differs first. A case can also name
  - a reference: a hash the ids must have, from outside this check's stored ids: the project's
    reference for a greedy held-out reply (results/record/reference/heldout-output.tsv), the
    stored record of a single run of the same prompt and settings in a fresh process, or
    another case's reply in this run. This is what ties the replies of one long engine to the
    single runs, and --store refuses a reply that is not its reference's;
  - conditions on the reply's "prompt" and "end" lines (stop=length, how=continue, kept>0).
An engine that dies takes its case with it, not the rest: it is started again for the next case.
An engine must also exit with code 0 when its input ends.

Speeds are printed: one process per configuration, each reply after the others. They compare with
this check's own earlier runs, not with the documents' numbers.

To add a check: a line in CASES (and, for settings a request cannot carry, a configuration in
tests/engine.py), then --store once and a look at what was stored
(--only full/single: the cases whose keys start so, e.g. the new ones alone).
"""
import collections
import json
import os
import re
import sys
import time

import engine
import sessions_check
from engine import ROOT

EXPECTED = 'tests/data/cases/expected.json'
S1 = 'temp=1 top_p=0.95 seed=1'
GREEDY_TSV = 'results/record/reference/heldout-output.tsv'  # the project's reference: plain greedy decoding on a fixed placement

Case = collections.namedtuple('Case', 'run config key prompt fields ref expect text')


def case(run, config, key, prompt, fields='', ref=None, expect=(), text=None):
    """prompt: ('heldout', name[, n]): that held-out prompt, rendered, with its own n or the one given;
               ('chat', user text, n): one user turn, rendered as --chat does;
               ('after', key, user text, n): the prompt of the case `key`, its reply in this run, and a next user turn;
               ('file', path, n): the file's text after BOS, as -f gives it to the engine.
    fields: the request's fields after n= ("temp=1 seed=1", "keep=0").
    ref: a reference, a function (want, got) -> (the hash the reply's ids must have, whose it is); see below.
    expect: conditions on the fields of the reply's "prompt" and "end" lines: 'stop=length', 'kept>0', 'how=continue'.
    text: a check of the reply's bytes, a function (bytes) -> (bool, what they must be); see session_reply."""
    return Case(run, config, key, prompt, fields, ref, tuple(expect), text)


# References: what ties a reply of the long engine to something outside this check's own stored ids.
def record_hash(path):
    """output.hash of a stored --record file: a single run of the same prompt and settings in a fresh process"""
    return lambda want, got: (json.load(open(path))['output']['hash'], f'the single run\'s ({path})')


def greedy_reference(name):
    """the output_hash of the project's reference for a held-out prompt"""
    def f(want, got):
        for l in open(GREEDY_TSV):
            r = l.rstrip('\n').split('\t')
            if len(r) >= 5 and r[0] == name:
                return r[4], f'the reference\'s ({GREEDY_TSV})'
        return None, f'the reference\'s ({GREEDY_TSV}, which has no "{name}")'
    return f


def same_as(key, first=None):
    """the reply of another case in this run (its first `first` ids)"""
    return lambda want, got: (engine.fnv1a64_ids(got[key][:first]) if key in got else None,
                              (f'the first {first} ids of ' if first else '') + f'the reply of "{key}" in this run')


def session_reply(stored, k):
    """the reply's bytes are reply k of a stored interactive session (tests/sessions_check.py runs the sessions)"""
    return lambda new: (sessions_check.is_reply(new, open(stored, 'rb').read(), k), f'reply {k} of the session {stored}')


CASES = []
# default: what stands around the arithmetic in the product, on the fastest settings (step 188, until
# step 206 tests/outputs_check.py's own engine): the eight held-out prompts greedy, two of them sampled,
# a short n, and a second turn that goes on from the kept context.
for _name in engine.heldout():
    CASES.append(case('default', 'full', f'full/greedy/{_name}', ('heldout', _name), ref=greedy_reference(_name)))
# The held-out prompts sampled with one seed: what results/sampler/spec-heldout.matrix ran as eight
# processes (steps 150-151). Two in default, the other six in complete.
for _name in engine.heldout():
    CASES.append(case('default' if _name in ('tcp-udp', 'cpp-matrix') else 'complete', 'full', f'full/s1/{_name}', ('heldout', _name), S1,
                      ref=record_hash(f'results/sampler/step151/spec-heldout/s1-{_name}-1.json')))
CASES += [
    case('default', 'full', 'full/n5/geometry', ('heldout', 'geometry', 5), ref=same_as('full/greedy/geometry', 5), expect=['stop=length']),
    # a second turn goes on from the context of the first; the same request on an empty context gives the same reply
    case('default', 'full', 'full/turn/first', ('chat', 'Name three primary colors, one line.', 4000)),
    case('default', 'full', 'full/turn/second', ('after', 'full/turn/first', 'And three secondary ones?', 4000), expect=['how=continue', 'kept>0']),
    case('default', 'full', 'full/turn/second-empty', ('after', 'full/turn/first', 'And three secondary ones?', 4000), 'keep=0',
         ref=same_as('full/turn/second'), expect=['how=empty', 'kept=0']),
]
# complete: sampling across the loops (steps 149-151; until step 207 two matrices of 33 processes,
# results/sampler/plain.matrix and results/sampler/step151/spec.matrix, on three prompts). One seed
# must give one reply in the plain loop, on another placement, with the cache, with speculation and with
# fixed draft lengths; another seed and top-p 1 (the sampler's other path) are checked in both loops.
# Each reply's reference is the matrix's stored record: a single run with the flags on the command line.
S2, S1P1 = 'temp=1 top_p=0.95 seed=2', 'temp=1 seed=1'
SPEC, PLAIN = 'results/sampler/step151/spec', 'results/sampler/plain'
for _config, _key, _fields, _record in [
        ('full', 's2', S2, SPEC + '/s2'), ('full', 's1-p1', S1P1, SPEC + '/s1-p1'),
        ('spec-n5', 's1', S1, SPEC + '/s1-n5'),
        ('plain', 'greedy', '', None), ('plain', 's1', S1, PLAIN + '/s1'), ('plain', 's2', S2, PLAIN + '/s2'), ('plain', 's1-p1', S1P1, PLAIN + '/s1-p1'),
        ('plain-oracle', 's1', S1, PLAIN + '/s1-oracle'), ('plain-cache', 's1', S1, PLAIN + '/s1-cache')]:
    for _name in ('tcp-udp', 'cpp-matrix', 'geometry'):
        CASES.append(case('complete', _config, f'{_config}/{_key}/{_name}', ('heldout', _name), _fields,
                          ref=record_hash(f'{_record}-{_name}-1.json') if _record else greedy_reference(_name)))
# complete: a session's reply k is the reply to the conversation before it as one prompt (steps 108-109;
# until step 208 ten processes with -f, tests/sessions_check.py's second half). The prompts are the stored
# conversations, raw; the replies' bytes must be those the six sessions printed, which that check still runs.
_T = 'results/chat/think'
for _tag, _file, _n, _stored, _k in (
        [('chat200', 'results/chat/whole.txt', 200, 'results/chat/session.out', 3), ('chat12', 'results/chat/whole-n12.txt', 12, 'results/chat/session-n12.out', 3)]
        + [(t, f'{_T}/{t}-ref{k}.txt', n, f'{_T}/{t}.out', k) for k in (2, 3) for t, n in (('th', 600), ('th20', 20), ('th45', 45), ('long', 2500))]):
    CASES.append(case('complete', 'full', f'full/single/{_tag}-{_k}', ('file', _file, _n), text=session_reply(_stored, _k)))
assert len({c.key for c in CASES}) == len(CASES), 'two cases with one key'
assert all(c.config in engine.CONFIGS for c in CASES)


def holds(cond, fields):
    """'stop=length', 'kept>0' on a reply's fields -> (bool, what the reply had)"""
    name, op, value = re.match(r'(\w+)(=|>)(.*)$', cond).groups()
    have = fields.get(name)
    if have is None:
        return False, f'{name} absent'
    return (have == value if op == '=' else float(have) > float(value)), f'{name}={have}'


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


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
    only = a[a.index('--only') + 1] if '--only' in a else ''
    cases = [c for c in CASES if c.run in runs and c.key.startswith(only)]
    prompts = engine.heldout()
    want = json.load(open(EXPECTED)) if os.path.exists(EXPECTED) else {}
    got, said, failed = {}, {}, 0  # said: {key: (the prompt's text, the reply's text)}, for the cases that go on from one

    def check(ok, what):
        nonlocal failed
        failed += not ok
        print(('ok    ' if ok else 'FAIL  ') + what, flush=True)

    for config in dict.fromkeys(c.config for c in cases):  # (in the order of their first case)
        eng, starts, tps, t0 = None, 0, [], time.time()
        for c in [c for c in cases if c.config == config]:
            key = c.key
            if c.prompt[0] == 'heldout':
                mode, n, text = prompts[c.prompt[1]]
                n, text = (c.prompt[2] if len(c.prompt) > 2 else n), engine.render(mode, text)
            elif c.prompt[0] == 'file':
                n, text = c.prompt[2], engine.BOS + open(c.prompt[1], encoding='utf-8').read()
            elif c.prompt[0] == 'chat':
                n, text = c.prompt[2], engine.render('chat', c.prompt[1])
            else:
                _, before, user, n = c.prompt
                if before not in said:
                    check(False, f'{key}: not run: it goes on from "{before}", which gave no reply')
                    continue
                text = said[before][0] + said[before][1] + engine.EOS + '<｜User｜>' + user + '<｜Assistant｜></think>'
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
            r = eng.request(f'gen n={n} {c.fields}'.rstrip(), text)
            if r is None:
                check(False, f'{key}: no reply: the engine said "{eng.said}"' + ('' if eng.alive() else f' and is gone (exit code {eng.p.wait()})'))
                continue
            said[key] = (text, r.text.decode('utf-8', errors='ignore'))
            h = engine.fnv1a64_ids(r.ids)
            speed = len(r.ids) / max(float(r.end['gen_s']), 1e-9)
            if len(r.ids) >= 90:  # (a speed over a few tokens says nothing)
                tps.append(speed)
            what = f'{key:26s} {len(r.ids):3d} tokens {h}; {speed:5.1f} t/s'
            w = want.get(key)
            whose = ''
            if c.ref is not None:
                ref, whose = c.ref(want, got)
                if h != ref:  # (not stored either: --store keeps the old ids)
                    check(False, f'{what}: not {ref}, {whose}' + (f'; it differs from the stored ids at token {first_difference(r.ids, w)} of {len(w)}' if w else ''))
                    continue
            if c.text is not None:
                ok, whose_text = c.text(r.text)
                if not ok:
                    open(f'{out}/{key.replace("/", "-")}.out', 'wb').write(r.text)
                    check(False, f'{what}: its bytes are not {whose_text}; what it gave: {out}/{key.replace("/", "-")}.out')
                    continue
            got[key] = r.ids
            if c.text is not None:
                what += f'; the bytes of {whose_text}'
            conds = [holds(e, dict(r.prompt, **r.end)) for e in c.expect]
            if not all(ok for ok, _ in conds):
                check(False, f'{what}: expected {", ".join(c.expect)}; the reply has {", ".join(h for _, h in conds)}')
                continue
            what += ''.join(f', {h}' for _, h in conds) + (f'; {whose.split(" (")[0]}' if whose else '')  # (where the reference is: on a failure)
            if w == r.ids or (store and w is None):
                check(True, what)
            elif w is None:
                check(False, f'{what}: nothing is expected for "{key}" in {EXPECTED} (--store writes it)')
            else:
                k = first_difference(r.ids, w)
                check(store, f'{what}: {"replaces" if store else "differs from"} what is expected at token {k} of {len(w)} '
                             f'(got {r.ids[k] if k < len(r.ids) else "the end"}, expected {w[k] if k < len(w) else "the end"})')
        if eng is not None and eng.alive():
            rc = eng.close()
            check(rc == 0, f'the engine "{config}" exits with code {rc} when its input ends')
        if tps:
            print(f'# engine "{config}": the {len(tps)} replies of 90 tokens and more: mean {sum(tps) / len(tps):.1f} t/s ({min(tps):.1f} to {max(tps):.1f}); '
                  f'{time.time() - t0:.0f} s with {starts} start{"s" if starts > 1 else ""}. Each reply after the others in one process:\n'
                  f'#   it compares with this check\'s earlier runs, not with the documents\' numbers (tests/speed_check.py measures those)', flush=True)

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
        print('# next: the logits checks of the same run (scripts/check.py default). If they pass, the arithmetic is right and the difference\n'
              '#   is in the loop: the case alone in a fresh process (bin-hip/ds-completion with the configuration\'s arguments and -p,\n'
              '#   QUICKSTART "Run"). If that gives the stored ids, the difference is in what the engine kept from the requests before it;\n'
              '#   if not, the same without --dspark --spec, then without --cache-loads, to see which part changes the reply.')
    print('FAILED' if failed else 'all checks passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
