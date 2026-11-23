#!/usr/bin/env python3
"""The product's outputs against stored token ids, with as few loads of the model as it takes
(README step 188): the second half of `scripts/check.py default`.

  python3 tests/outputs_check.py [--store]        exit code 1 if a check fails

The logits checks (ds-test --e2e) cover the arithmetic. This covers what stands around it in
the product: prompt chunks, speculative decoding with the draft, the adaptive expert cache,
sampling, a context kept between requests, an interactive session. With --canon none of them
may change a token.

One engine with the fastest settings and --requests (one load, ~140 s) takes in turn:
  - the eight held-out prompts, greedy;
  - two of them sampled (temperature 1, top-p 0.95, seed 1);
  - one of them with n=5: the first five ids of its full reply;
  - a second turn that goes on from the kept context, and the same request on an empty context
    (keep=0): the engine must say how it started each, and the two replies must be equal.
Then one fresh process, an interactive session (-i) with thinking and three questions, with the
placement and speculation but without the expert cache: the session's bytes against
results/chat/think/th20.out. Without the cache because its start is what takes long: the model
loads in ~13 s, and registering the 289 GB of expert tensors for the cache's loads takes ~105 s
more, which --cache-wait waits for at the start and any run with --cache-loads at its exit.

Every reply's token ids must be those in tests/outputs/expected.json. Ids are stored, not a hash,
so that a failed check can say which token differs first. The eight greedy replies are tied to
the project's reference: their fnv1a64 must be the output_hash of
results/record/reference/heldout-output.tsv (plain greedy decoding on a fixed placement).
--store writes what the engine gave as the new expectation, and refuses if a greedy reply is
not the reference's.

Speeds are printed for the held-out prompts: one process, the cache adapting from prompt to
prompt. They compare with this check's own earlier runs, not with the documents' numbers.
"""
import json
import os
import subprocess
import sys
import time

HOME = os.path.expanduser('~')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = HOME + '/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf'
F8 = HOME + '/llms/ds-1017-fp8/dense-fp8.gguf'
D = HOME + '/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf'
EXPECTED = 'tests/outputs/expected.json'
BOS, EOS = '<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>'
S1 = 'temp=1 top_p=0.95 seed=1'
SAMPLED = ['tcp-udp', 'cpp-matrix']
SESSION_Q, SESSION_OUT = 'results/chat/think/th20.q', 'results/chat/think/th20.out'


def render(mode, user):
    if mode == 'think':
        return (BOS + '<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n'
                + '<｜User｜>' + user + '<｜Assistant｜><think>')
    return BOS + '<｜User｜>' + user + '<｜Assistant｜></think>'


def fnv1a64_ids(ids):
    h = 0xcbf29ce484222325
    for i in ids:
        for b in range(4):
            h = ((h ^ ((i >> (8 * b)) & 0xff)) * 0x100000001b3) & 0xffffffffffffffff
    return 'fnv1a64:%016x' % h


def main():
    store = '--store' in sys.argv[1:]
    os.chdir(ROOT)
    prompts = {}
    for l in open('results/spec-heldout/prompts.txt', encoding='utf-8'):
        name, mode, n, text = l.rstrip('\n').split('|', 3)
        prompts[name] = (mode, int(n), text)
    reference = {}
    for l in open('results/record/reference/heldout-output.tsv'):
        r = l.rstrip('\n').split('\t')
        if len(r) >= 5 and r[0] in prompts:
            reference[r[0]] = r[4]
    want = json.load(open(EXPECTED)) if os.path.exists(EXPECTED) else {}
    got = {}
    failed = 0

    def check(ok, what):
        nonlocal failed
        failed += not ok
        print(('ok    ' if ok else 'FAIL  ') + what, flush=True)

    def same(key, ids, what):
        """ids against the expected ones under key; a difference is located"""
        got[key] = ids
        w = want.get(key)
        if w == ids or (store and w is None):
            check(True, what)
            return
        if w is None:
            check(False, f'{what}: nothing is expected for "{key}" in {EXPECTED} (--store writes it)')
            return
        k = next((i for i, (a, b) in enumerate(zip(ids, w)) if a != b), min(len(ids), len(w)))
        check(store, f'{what}: {"differs from" if not store else "replaces"} what is expected at token {k} of {len(w)} '
                     f'(got {ids[k] if k < len(ids) else "the end"}, expected {w[k] if k < len(w) else "the end"}; {len(ids)} tokens)')

    cmd = ['bin-hip/ds-completion', '-m', M, '--rt', F8, '--placement', 'placements/study30.txt', '--canon',
           '--cache-loads', '800', '--cache-period', '8', '--cache-wait', '--dspark', D, '--spec', '--requests']
    print('# engine: ' + ' '.join(c.replace(HOME, '~') for c in cmd), flush=True)
    t0 = time.time()
    eng = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    line = eng.stdout.readline().decode().strip()
    print(f'# "{line}" after {time.time() - t0:.0f} s', flush=True)
    if line != 'ready':
        print('FAIL  the engine did not start (no "ready" line): run the command above by hand')
        return 1

    def request(head, text):
        """-> (ids, text bytes, the prompt line's fields, the end line's fields); None if the engine refused or left"""
        body = text.encode('utf-8')
        eng.stdin.write(f'{head} bytes={len(body)}\n'.encode() + body)
        eng.stdin.flush()
        ids, out, pr = [], b'', {}
        while True:
            w = eng.stdout.readline().decode().split()
            if not w or w[0] == 'error':
                print(f'      the engine answered "{" ".join(w) or "(closed)"}" to "{head}"', flush=True)
                return None
            if w[0] == 'prompt':
                pr = dict(f.split('=') for f in w[1:])
            elif w[0] == 'tok':
                ids.append(int(w[1]))
                out += bytes.fromhex(w[2]) if len(w) > 2 else b''
            elif w[0] == 'end':
                return ids, out, pr, dict(f.split('=') for f in w[1:])

    tps = []
    for name, (mode, n, text) in prompts.items():  # the held-out prompts, greedy
        r = request(f'gen n={n}', render(mode, text))
        if not r:
            check(False, f'{name}, greedy: no reply')
            continue
        ids, out, pr, end = r
        h = fnv1a64_ids(ids)
        tps.append(len(ids) / float(end['gen_s']))
        if h != reference.get(name):  # (a failed check: --store then writes nothing)
            got[f'greedy/{name}'] = ids
            w = want.get(f'greedy/{name}') or []
            k = next((i for i, (x, y) in enumerate(zip(ids, w)) if x != y), min(len(ids), len(w)))
            check(False, f'{name}, greedy: {h} is not the reference\'s {reference.get(name)} (results/record/reference/heldout-output.tsv)'
                         + (f'; it differs at token {k} of {len(w)}' if w else ''))
        else:
            same(f'greedy/{name}', ids, f'{name:12s} greedy  {len(ids):3d} tokens {h}, the reference\'s; {tps[-1]:5.1f} t/s')
    for name in SAMPLED:
        mode, n, text = prompts[name]
        r = request(f'gen n={n} {S1}', render(mode, text))
        same(f'sampled/{name}', r[0], f'{name:12s} sampled {len(r[0]):3d} tokens {fnv1a64_ids(r[0])}') if r else check(False, f'{name}, sampled: no reply')
    mode, n, text = prompts['geometry']
    r = request('gen n=5', render(mode, text))
    check(bool(r) and r[0] == got.get('greedy/geometry', [])[:5] and r[3]['stop'] == 'length', 'geometry with n=5: the first five ids of its full reply, stop=length')
    # a second turn: it goes on from the context of the first; the same request on an empty context
    p1 = render('chat', 'Name three primary colors, one line.')
    r1 = request('gen n=4000', p1)
    if r1:
        p2 = p1 + r1[1].decode('utf-8', errors='ignore') + EOS + '<｜User｜>And three secondary ones?<｜Assistant｜></think>'
        r2, r3 = request('gen n=4000', p2), request('gen n=4000 keep=0', p2)
        same('turn/first', r1[0], f'a chat turn: {len(r1[0])} tokens, stop={r1[3]["stop"]}')
        if r2 and r3:
            check(r2[2].get('how') == 'continue' and int(r2[3]['kept']) > 0 and r3[2].get('how') == 'empty' and r3[3]['kept'] == '0',
                  f'the second turn goes on from the context (how={r2[2].get("how")}, kept {r2[3]["kept"]} of {r2[3]["prompt_tokens"]} prompt tokens); with keep=0 it starts empty (how={r3[2].get("how")})')
            check(r2[0] == r3[0] and len(r2[0]) > 0, f'the two replies are equal ({len(r2[0])} tokens)')
            same('turn/second', r2[0], 'the second turn\'s reply')
        else:
            check(False, 'the second turn: no reply')
    else:
        check(False, 'a chat turn: no reply')
    eng.stdin.close()
    rc = eng.wait()
    check(rc == 0, f'the engine exits with code {rc} when its input ends')
    if tps:
        print(f'# held-out greedy in this process: mean {sum(tps) / len(tps):.1f} t/s ({min(tps):.1f} to {max(tps):.1f}); {time.time() - t0:.0f} s with the load', flush=True)

    # a fresh process: an interactive session with thinking, three turns
    t1 = time.time()
    sess = ['bin-hip/ds-completion', '-m', M, '--rt', F8, '--placement', 'placements/study30.txt', '--canon', '--dspark', D, '--spec',
            '--chat', '-i', '--think', '--think-show', 'all', '--effort', '75', '-n', '20']
    s = subprocess.run(sess, stdin=open(SESSION_Q), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    ok = s.returncode == 0 and s.stdout == open(SESSION_OUT, 'rb').read()
    check(ok, f'a session of three turns with thinking (-i, a fresh process; {SESSION_Q}): the bytes of {SESSION_OUT}; {time.time() - t1:.0f} s with its load')
    if not ok:
        open('data/outputs_check-session.out', 'wb').write(s.stdout)
        print(f'      exit code {s.returncode}; what it printed: data/outputs_check-session.out (cmp with {SESSION_OUT}); the command:')
        print('      ' + ' '.join(c.replace(HOME, '~') for c in sess) + f' < {SESSION_Q}')

    if store and not failed:
        os.makedirs(os.path.dirname(EXPECTED), exist_ok=True)
        changed = [k for k in got if want.get(k) != got[k]]
        with open(EXPECTED, 'w') as f:
            f.write('{\n' + ',\n'.join(f'  {json.dumps(k)}: {json.dumps(got[k])}' for k in sorted(got)) + '\n}\n')
        print(f'# {EXPECTED} written: {len(got)} replies, {len(changed)} new or changed ({", ".join(changed) or "none"})')
    elif failed:  # what the engine gave, to look at
        os.makedirs('data', exist_ok=True)
        json.dump(got, open('data/outputs_check-got.json', 'w'))
        print('# the replies\' ids as the engine gave them: data/outputs_check-got.json')
        print('# next: the logits checks of the same run (scripts/check.py default). If they pass, the arithmetic is right and the\n'
              '#   difference is in the loop: run the prompt named above alone (bin-hip/ds-completion ... -p, QUICKSTART "Run"),\n'
              '#   then without --dspark --spec, then without --cache-loads, to see which part changes its reply.')
    print('FAILED' if failed else 'all checks passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
