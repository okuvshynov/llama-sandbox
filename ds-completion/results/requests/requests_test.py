#!/usr/bin/env python3
"""README step 152: the engine's --requests mode against single runs of the same prompts.

  results/requests/requests_test.py [--no-spec] > results/requests/requests_test.txt

Starts one engine with --requests (the fastest single-machine settings, as
results/sampler/spec-heldout.matrix; --no-spec: the plain loop) and sends held-out prompts one
after another, greedy and sampled, rendered here as the engine's --chat / --think render them.
Every reply's token ids must hash to the stored value of a single run of that prompt:
greedy: results/record/reference/heldout-output.tsv; temperature 1, top-p 0.95, seed 1:
results/sampler/spec-heldout/s1-<prompt>-1.json. Also: a prompt again after others, a short
n (a prefix of the full reply, stop=length), requests with mistakes (an error line, the
session goes on), cancels by SIGUSR1 (step 154): during a reply, during a prompt, between
requests, and prompts that go on from the context (step 158): kept=N, and the reply of the
same request on an empty context (keep=0); conversations that share a system prompt
(step 160): a saved point at its end (saved=N), returned to by the later ones; the engine's
account of its choice of context (step 164): ctx, common and how in the "prompt" line, and its
"# context:" lines on stderr. Exit code 1 if a check fails.
"""
import json
import os
import subprocess
import re
import sys
import tempfile
import time

HOME = os.path.expanduser('~')
M = HOME + '/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf'
F8 = HOME + '/llms/ds-1017-fp8/dense-fp8.gguf'
D = HOME + '/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf'
BOS = '<｜begin▁of▁sentence｜>'
S1 = 'temp=1 top_p=0.95 seed=1'


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
    spec = '--no-spec' not in sys.argv[1:]
    prompts = {}
    for l in open('results/spec-heldout/prompts.txt', encoding='utf-8'):
        name, mode, n, text = l.rstrip('\n').split('|', 3)
        prompts[name] = (mode, int(n), text)
    greedy = {}
    for l in open('results/record/reference/heldout-output.tsv'):
        r = l.rstrip('\n').split('\t')
        if len(r) >= 5:
            greedy[r[0]] = r[4]
    sampled = {p: json.load(open(f'results/sampler/spec-heldout/s1-{p}-1.json'))['output']['hash'] for p in prompts}

    cmd = ['bin-hip/ds-completion', '-m', M, '--rt', F8, '--placement', 'placements/study30.txt', '--canon',
           '--cache-loads', '800', '--cache-period', '8', '--cache-wait', '--requests']
    if spec:
        cmd += ['--dspark', D, '--spec']
    print('# engine: ' + ' '.join(c.replace(HOME, '~') for c in cmd), flush=True)
    t0 = time.time()
    err = tempfile.NamedTemporaryFile(prefix='requests_test-', suffix='.err')  # the engine's own lines (step 164 reads them)
    eng = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err)
    line = eng.stdout.readline().decode().strip()
    print(f'# "{line}" after {time.time() - t0:.0f} s', flush=True)
    if line != 'ready':
        print('FAIL  no ready line')
        return 1

    def request(head, text):
        """-> ('error', message) or (ids, bytes, the end line's fields)"""
        body = text.encode('utf-8')
        eng.stdin.write((head.replace('BYTES', str(len(body))) + '\n').encode() + body)
        eng.stdin.flush()
        ids, out = [], b''
        while True:
            w = eng.stdout.readline().decode().split()
            if not w:
                return 'closed', ''
            if w[0] == 'error':
                return 'error', ' '.join(w[1:])
            if w[0] == 'prompt':  # step 164: the engine's choice of context, one entry per accepted request
                seen.append(dict(f.split('=') for f in w[1:]))
            if w[0] == 'tok':
                ids.append(int(w[1]))
                out += bytes.fromhex(w[2]) if len(w) > 2 else b''
            elif w[0] == 'end':
                return ids, out, dict(f.split('=') for f in w[1:])

    failed = 0
    seen = []  # the "prompt" lines' fields, in order

    def check(ok, what):
        nonlocal failed
        failed += not ok
        print(('ok    ' if ok else 'FAIL  ') + what, flush=True)

    full = {}
    # (prompt, sampler fields, the stored hash's table)
    for name, how in [('tcp-udp', ''), ('cpp-matrix', S1), ('zh-travel', S1), ('tcp-udp', ''), ('geometry', S1),
                      ('geometry', ''), ('summary-json', ''), ('cpp-matrix', S1), ('py-lru', '')]:
        mode, n, text = prompts[name]
        r = request(f'gen n={n} {how} bytes=BYTES', render(mode, text))
        if r[0] in ('error', 'closed'):
            check(False, f'{name}: {r}')
            continue
        ids, out, end = r
        want = (sampled if how else greedy)[name]
        got = fnv1a64_ids(ids)
        full[name, how] = ids
        try:
            out.decode('utf-8')
            utf8 = 'valid UTF-8'
        except UnicodeDecodeError:
            utf8 = 'NOT valid UTF-8'
        check(got == want and int(end['gen_tokens']) == len(ids) and utf8 == 'valid UTF-8',
              f"{name:12s} {'sampled' if how else 'greedy ':7s} {len(ids):3d} tokens, stop={end['stop']:6s} {got} "
              f"{'= the single run' if got == want else 'is not the single run: ' + want}; {len(out)} bytes, {utf8}; "
              f"prompt {end['prompt_tokens']} tokens {float(end['prompt_s']):.2f} s, generation {len(ids) / float(end['gen_s']):.2f} t/s")
    # special tokens are written out in a reply: the reasoning's end in the think prompt's
    mode, n, text = prompts['geometry']
    ids, out, end = request(f'gen n=5 bytes=BYTES', render(mode, text))
    check(ids == full['geometry', ''][:5] and end['stop'] == 'length', f'n=5: the first 5 ids of the full reply, stop={end["stop"]}')
    ids, out, end = request(f'gen n=4000 bytes=BYTES', render('chat', 'Reply with the single word: yes'))
    check(end['stop'] == 'eos' and len(ids) < 50, f'a short reply ends by itself: {len(ids)} tokens, stop={end["stop"]}, text {out.decode()!r}')
    ids, out, end = request(f'gen n=4000 bytes=BYTES', render('think', 'What is 2 + 3? Reply with the number.'))
    check('</think>' in out.decode() and end['stop'] == 'eos', f'a think reply carries </think> as text: {len(ids)} tokens, stop={end["stop"]}, its end {out.decode()[-40:]!r}')
    for head, text in [('gen n=0 bytes=BYTES', 'x'), ('gen n=10 temp=-1 bytes=BYTES', 'x'), ('gen n=10 top_p=0 temp=1 bytes=BYTES', 'x'),
                       ('gen n=10 bytes=BYTES', '')]:
        r = request(head, text)
        check(r[0] == 'error', f'"{head}" with {len(text)} bytes: {r[0]} {r[1] if r[0] == "error" else ""}')
    mode, n, text = prompts['tcp-udp']
    r = request(f'gen n={n} bytes=BYTES', render(mode, text))
    check(r[0] not in ('error', 'closed') and fnv1a64_ids(r[0]) == greedy['tcp-udp'], 'tcp-udp greedy after the errors: the single run\'s reply')
    # a cancel (README step 154): SIGUSR1 two seconds into a long reply, and again during a prompt
    import signal
    import threading
    mode, n, text = prompts['py-lru']
    for label, n_req, body, delay in [('a reply of up to 4000 tokens, the signal after 2 s', 4000, render(mode, text), 2.0),
                                      ('a prompt of ~3000 tokens, the signal after 1 s', 300, render('chat', 'Summarize: ' + 'The quick brown fox jumps over the lazy dog. ' * 300), 1.0)]:
        t = threading.Timer(delay, lambda: eng.send_signal(signal.SIGUSR1))
        t0 = time.time()
        t.start()
        ids, out, end = request(f'gen n={n_req} bytes=BYTES', body)
        t.join()
        check(end['stop'] == 'cancel' and len(ids) < n_req and len(ids) == int(end['gen_tokens']),
              f'{label}: stop={end["stop"]} after {len(ids)} tokens, {time.time() - t0:.1f} s (prompt {end["prompt_tokens"]} tokens, {float(end["prompt_s"]):.1f} s)')
        if label.startswith('a reply'):
            check(ids == full['py-lru', ''][:len(ids)], f'its {len(ids)} tokens are the first of the full reply')
    for name, how in [('py-lru', ''), ('cpp-matrix', S1)]:
        mode, n, text = prompts[name]
        r = request(f'gen n={n} {how} bytes=BYTES', render(mode, text))
        want = (sampled if how else greedy)[name]
        check(r[0] not in ('error', 'closed') and fnv1a64_ids(r[0]) == want, f'{name} {"sampled" if how else "greedy"} after the cancels: the single run\'s reply')
    eng.send_signal(signal.SIGUSR1)  # with no request running: forgotten
    time.sleep(0.5)
    mode, n, text = prompts['tcp-udp']
    r = request(f'gen n={n} bytes=BYTES', render(mode, text))
    check(r[0] not in ('error', 'closed') and r[2]['stop'] == 'length' and fnv1a64_ids(r[0]) == greedy['tcp-udp'], 'a signal between requests does not touch the next one')
    # README step 158: a prompt that goes on from the context runs only its new tokens (kept=N), and
    # the reply is the one the same request gets on an empty context (keep=0)
    EOS, FOX = '<｜end▁of▁sentence｜>', 'The quick brown fox jumps over the lazy dog. '

    def turn2(first, more):
        return first + EOS + '<｜User｜>' + more + '<｜Assistant｜></think>'

    for label, how, n1, p1, more, n2 in [
            ('a short chat', '', 4000, render('chat', 'Name three primary colors, one line.'), 'And three secondary ones?', 4000),
            ('the same, sampled', S1, 4000, render('chat', 'Name three primary colors, one line.'), 'And three secondary ones?', 4000),
            ('a reply with reasoning, cut by n', '', 150, render(*prompts['geometry'][::2]), 'Go on.', 100),
            ('a 3000-token prompt, 1000 more', '', 30, render('chat', 'Summarize: ' + FOX * 300), 'Now this: ' + FOX * 100, 30),
            ('a long reply, sampled', S1, prompts['cpp-matrix'][1], render(*prompts['cpp-matrix'][::2]), 'Now the same in C.', 200)]:
        ids1, out1, end1 = request(f'gen n={n1} {how} bytes=BYTES', p1)
        p2 = turn2(p1 + out1.decode('utf-8', errors='ignore'), more)
        ids2, out2, end2 = request(f'gen n={n2} {how} bytes=BYTES', p2)
        ids3, out3, end3 = request(f'gen n={n2} {how} keep=0 bytes=BYTES', p2)
        in_ctx = int(end1['prompt_tokens']) + len(ids1) - (end1['stop'] == 'length')  # a reply cut by n: its last token was not run
        check(end1['kept'] == '0' and end3['kept'] == '0' and int(end2['kept']) > 0 and ids2 == ids3 and len(ids2) > 0,
              f"{label}: {end2['prompt_tokens']} prompt tokens, kept {end2['kept']} (the context held {in_ctx}), {float(end2['prompt_s']):.2f} s; "
              f"keep=0: {float(end3['prompt_s']):.2f} s; the two replies ({len(ids2)} tokens, {fnv1a64_ids(ids2)}) {'are equal' if ids2 == ids3 else 'DIFFER'}")
        check(int(end2['kept']) == in_ctx, f'{label}: all of the context was kept ({end2["kept"]} of {in_ctx})')
        s2, s3 = seen[-2], seen[-1]  # step 164: the same in the engine's words
        check(s2['how'] == 'continue' and int(s2['ctx']) == int(s2['common']) == in_ctx and s3['how'] == 'empty' and s3['kept'] == '0'
              and int(s3['common']) == int(s3['tokens']) < int(s3['ctx']),
              f"{label}: the prompt line says how={s2['how']} ctx={s2['ctx']} common={s2['common']}; with keep=0 how={s3['how']} ctx={s3['ctx']} common={s3['common']}")
    # the same prompt again: nothing to go on from (the context holds it and its reply)
    ids4, out4, end4 = request(f'gen n={n2} {how} bytes=BYTES', p2)
    check(end4['kept'] == '0' and ids4 == ids2, f'the same request again: kept {end4["kept"]}, the same reply')
    # a prompt cancelled between its chunks, then sent again: it goes on from the chunks that ran
    body = render('chat', 'Summarize: ' + FOX * 300)
    ref = request('gen n=30 keep=0 bytes=BYTES', body)
    request('gen n=5 bytes=BYTES', render('chat', 'Reply with the single word: yes'))
    t = threading.Timer(1.0, lambda: eng.send_signal(signal.SIGUSR1))
    t.start()
    ids, out, end = request('gen n=30 bytes=BYTES', body)
    t.join()
    ids5, out5, end5 = request('gen n=30 bytes=BYTES', body)
    check(end['stop'] == 'cancel' and int(end5['kept']) > 0 and int(end5['kept']) % 512 == 0 and ids5 == ref[0],
          f'a prompt cancelled after 1 s and sent again: kept {end5["kept"]} of {end5["prompt_tokens"]} tokens, the reply of the empty context')
    # README step 160: conversations that share their start (a system prompt) but do not go on from
    # one another. The second one puts a saved point at the end of what they share (saved=N); the
    # later ones return to it (kept=N). Every reply is the same request's on an empty context.
    # (the users' lines start with different words: the point lands at the end of what two prompts share, and
    # "Name a ..." twice would put it two tokens into the user's turn, for the next prompt to end it)
    SYS = BOS + '<｜System｜>You answer in one short line. Background, not to be repeated: ' + FOX * 150

    def conv(user, system=SYS):
        return system + '<｜User｜>' + user + '<｜Assistant｜></think>'

    def ask(user, how='', keep='', system=SYS, n=20):
        ids, out, end = request(f'gen n={n} {how} {keep} bytes=BYTES', conv(user, system))
        return ids, int(end['kept']), int(end['saved']), float(end['prompt_s']), int(end['prompt_tokens'])

    request('gen n=5 bytes=BYTES', render('chat', 'Reply with the single word: yes'))
    a_ids, a_kept, a_saved, a_s, a_n = ask('Color: name one.')
    b_ids, b_kept, b_saved, b_s, b_n = ask('Fruit: name one.')
    check(a_kept == 0 and a_saved == 0 and b_kept == 0 and b_saved > 1000,
          f'two conversations with one system prompt: the first runs whole ({a_n} tokens, {a_s:.1f} s), the second too and saves a point at {b_saved} ({b_s:.1f} s)')
    d = b_saved
    sa, sb = seen[-2], seen[-1]
    check(sa['how'] == 'empty' and int(sa['common']) < 64 and sb['how'] == 'empty' and int(sb['common']) == d and int(sb['ctx']) > d,
          f"the prompt lines: how={sa['how']} common={sa['common']} (ctx {sa['ctx']}), then how={sb['how']} common={sb['common']} (ctx {sb['ctx']})")
    LONG = 'Ignore the one-line rule: write a haiku about rain, then explain it in three sentences.'
    for label, user, how, n_gen in [('a third conversation', 'City: name one.', '', 20), ('a fourth, a long reply', LONG, '', 120), ('a fifth, the same sampled', LONG, S1, 120),
                                    ('the second one again', 'Fruit: name one.', '', 20)]:
        ids, kept, sv, t, n = ask(user, how, n=n_gen)
        s1 = seen[-1]
        ids0, kept0, sv0, t0, _ = ask(user, how, 'keep=0', n=n_gen)
        check(s1['how'] == 'saved' and int(s1['common']) >= d and seen[-1]['how'] == 'empty',
              f"{label}: the prompt line says how={s1['how']} common={s1['common']} ctx={s1['ctx']}; with keep=0 how={seen[-1]['how']}")
        check(kept == d and sv == d and kept0 == 0 and sv0 == d and ids == ids0 and len(ids) > 0 and (user != 'Fruit: name one.' or ids == b_ids),
              f'{label}: {n} prompt tokens, kept {kept}, {t:.2f} s; keep=0: kept {kept0}, {t0:.2f} s; the replies ({len(ids)} tokens) {"are equal" if ids == ids0 else "DIFFER"}; the point stays at {sv0}')
    # a reply's next turn goes on from the context, not from the point
    ids1, out1, end1 = request('gen n=20 bytes=BYTES', conv('Tree: name one.'))
    p2 = turn2(conv('Tree: name one.') + out1.decode(), 'And another?')
    ids2, out2, end2 = request('gen n=20 bytes=BYTES', p2)
    ids3, out3, end3 = request('gen n=20 keep=0 bytes=BYTES', p2)
    check(int(end1['kept']) == d and int(end2['kept']) > d and ids2 == ids3, f'the next turn of a conversation: kept {end2["kept"]} (the point is at {d}), the reply of the empty context')
    # a prompt that does not start with the point's tokens ends the point; the next two conversations set it again
    mode, n, text = prompts['tcp-udp']
    r = request(f'gen n={n} bytes=BYTES', render(mode, text))
    check(fnv1a64_ids(r[0]) == greedy['tcp-udp'] and r[2]['kept'] == '0' and r[2]['saved'] == '0', f'another prompt: the single run\'s reply, kept {r[2]["kept"]}, the point is gone (saved {r[2]["saved"]})')
    e = ask('Bird: name one.')
    f = ask('Fish: name one.')
    g = ask('City: name one.')
    g0 = ask('City: name one.', keep='keep=0')
    check(e[1:3] == (0, 0) and f[1:3] == (0, d) and g[1:3] == (d, d) and g[0] == g0[0],
          f'then three conversations: kept {e[1]}, {f[1]}, {g[1]}; the point at {e[2]}, {f[2]}, {g[2]}; the third\'s reply is the empty context\'s')
    # a system prompt that shares only the first part with the point's: the point ends, and the next conversation sets one at the shared part
    SYS2 = SYS[:len(SYS) // 2] + ' Now answer in French.'
    h = ask('Color: name one.', system=SYS2)
    i_ = ask('Fruit: name one.', system=SYS2)
    j = ask('City: name one.', system=SYS2)
    j0 = ask('City: name one.', keep='keep=0', system=SYS2)
    check(h[1] == 0 and 64 <= h[2] < d and i_[1] == h[2] and j[1] == h[2] and j[0] == j0[0] and len(j[0]) > 0,
          f'a system prompt that shares half: kept {h[1]} and a point at {h[2]} (what it shares with the last context); the next two keep {i_[1]}, {j[1]} of {j[4]}; the reply is the empty context\'s')
    r = request('generate n=10', '')
    check(r[0] == 'error' and 'closing' in r[1], f'an unknown request line: {r[0]} {r[1]}')
    eng.stdin.close()
    rc = eng.wait()
    check(rc == 0, f'the engine exits with code {rc} when its input ends')
    # README step 164: the engine's stderr has one "# context:" line per accepted request, and a second one
    # with the place wherever a prompt and the context part before either ends
    lines = [l for l in open(err.name, encoding='utf-8', errors='replace').read().split('\n') if l.startswith('# context: ')]
    first = [l for l in lines if l.startswith('# context: held ') or l.startswith('# context: empty')]
    parts = [int(m.group(1)) for l in lines for m in [re.match(r'# context: they part at token (\d+): ', l)] if m]
    want = [int(f['common']) for f in seen if int(f['common']) < int(f['ctx']) and int(f['common']) < int(f['tokens'])]
    hows = {h: sum(f['how'] == h for f in seen) for h in ('continue', 'saved', 'empty')}
    check(len(first) == len(seen), f'stderr: {len(first)} "# context:" lines for {len(seen)} accepted requests ({hows})')
    check(parts == want and len(want) > 0, f'stderr: {len(parts)} "they part at token N" lines, the N of each is its request\'s common ({len(want)} such requests)')
    words = {'continue': 'it goes on from the context', 'saved': 'it returns to the saved point', 'empty': 'it starts on the empty context'}
    check(all(words[f['how']] in l for f, l in zip(seen, first)), 'stderr: every line says what its prompt line says')
    print('# the engine\'s lines for the second and third conversations with one system prompt (the saved point set, then used):')
    k = next(i for i, f in enumerate(seen) if f['how'] == 'saved')
    pos = [i for i, l in enumerate(lines) if not l.startswith('# context: they part') and not l.startswith('# context: the prompt ends')]
    for l in lines[pos[k - 1]:pos[k + 1]]:
        print('#   ' + l)
    print('FAILED' if failed else 'all checks passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
