#!/usr/bin/env python3
"""Interactive sessions against their stored replies (README steps 108-109; gathered for
`scripts/check.py complete` in step 190).

  python3 tests/sessions_check.py        exit code 1 if a check fails

Six sessions of three turns through scripts/chat.sh (the fastest settings; --cache-wait, so the
cache loads during them): a chat at -n 200 and -n 12, and with thinking at -n 600, 20 and 45 and
with --effort max -n 2500 (replies of over a thousand tokens: rewinds longer than the window).
Each session's stdout must be the stored replies (results/chat/*.out, results/chat/think/*.out).
--think-show all: the stored files hold the reasoning. Six starts.

Until step 208 ten single runs followed (`ds-completion -f` on the conversation before a reply:
each must print that reply). They are cases of tests/cases_check.py now (full/single/...), which
uses is_reply() below.
"""
import os
import subprocess
import sys
import time

HOME = os.path.expanduser('~')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    if failed:
        print('# next: cmp the printed file with the stored one for the turn; then tests/cases_check.py complete --only full/single (the same conversations as single prompts),\n'
              '#   and the session again without --dspark --spec (scripts/chat.sh passes its arguments on)')
    print('FAILED' if failed else 'all checks passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
