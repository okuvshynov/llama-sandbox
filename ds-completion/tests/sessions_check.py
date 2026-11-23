#!/usr/bin/env python3
"""Interactive sessions against their stored replies, and single runs on the whole conversation
against the sessions (README steps 108-109; gathered for `scripts/check.py complete` in step 190).

  python3 tests/sessions_check.py        exit code 1 if a check fails

Six sessions of three turns through scripts/chat.sh (the fastest settings; --cache-wait, so the
cache loads during them): a chat at -n 200 and -n 12, and with thinking at -n 600, 20 and 45 and
with --effort max -n 2500 (replies of over a thousand tokens: rewinds longer than the window).
Each session's stdout must be the stored replies (results/chat/*.out, results/chat/think/*.out).
Then ten single runs, `ds-completion -f` on the conversation before a reply as one raw prompt
(results/chat/whole*.txt, results/chat/think/<tag>-ref<k>.txt): each must print that reply.
--think-show all: the stored files hold the reasoning. About 45 minutes: 16 starts.
"""
import os
import subprocess
import sys
import time

HOME = os.path.expanduser('~')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = HOME + '/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf'
F8 = HOME + '/llms/ds-1017-fp8/dense-fp8.gguf'
D = HOME + '/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf'
T = 'results/chat/think'
CHAT_Q = 'Name three prime numbers above 100.\nNow multiply the first two of them.\nWrite a C function that checks whether a number is prime.\n'
THINK = ['--think', '--think-show', 'all']
# (tag, the stored replies, the questions (a file, or None: CHAT_Q), chat.sh's arguments)
SESSIONS = [
    ('chat200', 'results/chat/session.out', None, ['-n', '200']),
    ('chat12', 'results/chat/session-n12.out', None, ['-n', '12']),
    ('th', f'{T}/th.out', f'{T}/th.q', THINK + ['--effort', '75', '-n', '600']),
    ('th20', f'{T}/th20.out', f'{T}/th20.q', THINK + ['--effort', '75', '-n', '20']),
    ('th45', f'{T}/th45.out', f'{T}/th45.q', THINK + ['--effort', '75', '-n', '45']),
    ('long', f'{T}/long.out', f'{T}/long.q', THINK + ['--effort', 'max', '-n', '2500']),
]
# (tag, the conversation before the reply, -n, the stored replies, which reply)
FRESH = [('chat200', 'results/chat/whole.txt', 200, 'results/chat/session.out', 3), ('chat12', 'results/chat/whole-n12.txt', 12, 'results/chat/session-n12.out', 3)]
for k in (2, 3):
    FRESH += [('th', f'{T}/th-ref{k}.txt', 600, f'{T}/th.out', k), ('th20', f'{T}/th20-ref{k}.txt', 20, f'{T}/th20.out', k),
              ('th45', f'{T}/th45-ref{k}.txt', 45, f'{T}/th45.out', k), ('long', f'{T}/long-ref{k}.txt', 2500, f'{T}/long.out', k)]


def is_session(new, stored):
    """a session's stdout: its replies, each ended by a newline; the stored file joins them by \\x1e or by a newline"""
    return new.rstrip(b'\n') == stored.replace(b'\x1e', b'\n').rstrip(b'\n')


def is_reply(new, stored, k):
    x = new.rstrip(b'\n')
    if b'\x1e' in stored:
        return x == stored.split(b'\x1e')[k - 1].rstrip(b'\n')
    t = stored.rstrip(b'\n')  # joined by newlines, and a reply may hold newlines: reply 3 is the file's end
    lines = t.split(b'\n')
    return len(x) > 0 and (x == lines[k - 1] or (t.endswith(b'\n' + x) if k == 3 else b'\n' + x + b'\n' in t))


def main():
    os.chdir(ROOT)
    os.makedirs('data/check', exist_ok=True)
    failed = 0

    def check(ok, what, new):
        nonlocal failed
        failed += not ok
        print(('ok    ' if ok else 'FAIL  ') + what, flush=True)
        if not ok:
            f = 'data/check/sessions-' + what.split(':')[0].replace(' ', '-') + '.out'
            open(f, 'wb').write(new)
            print(f'      what it printed: {f}')

    for tag, stored, q, args in SESSIONS:
        t0 = time.time()
        r = subprocess.run(['scripts/chat.sh', '--cache-wait'] + args, input=open(q, 'rb').read() if q else CHAT_Q.encode(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        check(r.returncode == 0 and is_session(r.stdout, open(stored, 'rb').read()), f'session {tag}: chat.sh {" ".join(args)}: the bytes of {stored} ({time.time() - t0:.0f} s)', r.stdout)
    base = ['bin-hip/ds-completion', '-m', M, '--rt', F8, '--placement', 'placements/study30.txt', '--dspark', D, '--spec', '--canon',
            '--cache-loads', '800', '--cache-period', '8', '--cache-wait']
    for tag, f, n, stored, k in FRESH:
        t0 = time.time()
        r = subprocess.run(base + ['-f', f, '-n', str(n)], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        check(r.returncode == 0 and is_reply(r.stdout, open(stored, 'rb').read(), k), f'single {tag} {k}: -f {f} -n {n}: reply {k} of {stored} ({time.time() - t0:.0f} s)', r.stdout)
    if failed:
        print('# next: cmp the printed file with the stored one for the turn; then tests/outputs_check.py (the same loop without -i),\n'
              '#   and the session again without --dspark --spec (scripts/chat.sh passes its arguments on)')
    print('FAILED' if failed else 'all checks passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
