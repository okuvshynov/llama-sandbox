#!/usr/bin/env python3
"""One interactive session in a fresh process, against its stored bytes: the part of
`scripts/check.py default` that is not a request to the long engine (README steps 188, 206).

  python3 tests/outputs_check.py        exit code 1 if the check fails

Until step 206 this file also started the engine with --requests and sent it the held-out
prompts; those requests are now cases of tests/cases_check.py (its `default` run), on the engine
that check starts anyway. What stays here is what a request cannot be: the command line and the
-i loop. One process, a session (-i) with thinking and three questions, with the placement and
speculation but without the expert cache: the session's bytes against results/chat/think/th20.out.
Without the cache because its start is what takes long: the model loads in ~13 s, and registering
the 289 GB of expert tensors for the cache's loads takes 40 s and more, which --cache-wait waits
for at the start and any run with --cache-loads at its exit.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import D, F8, HOME, M, ROOT  # noqa: E402

SESSION_Q, SESSION_OUT = 'results/chat/think/th20.q', 'results/chat/think/th20.out'


def main():
    os.chdir(ROOT)
    t1 = time.time()
    sess = ['bin-hip/ds-completion', '-m', M, '--rt', F8, '--placement', 'placements/study30.txt', '--canon', '--dspark', D, '--spec',
            '--chat', '-i', '--think', '--think-show', 'all', '--effort', '75', '-n', '20']
    s = subprocess.run(sess, stdin=open(SESSION_Q), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    ok = s.returncode == 0 and s.stdout == open(SESSION_OUT, 'rb').read()
    print(('ok    ' if ok else 'FAIL  ') + f'a session of three turns with thinking (-i, a fresh process; {SESSION_Q}): the bytes of {SESSION_OUT}; '
          f'{time.time() - t1:.0f} s with its load', flush=True)
    if not ok:
        os.makedirs('data', exist_ok=True)
        open('data/outputs_check-session.out', 'wb').write(s.stdout)
        print(f'      exit code {s.returncode}; what it printed: data/outputs_check-session.out (cmp with {SESSION_OUT}); the command:')
        print('      ' + ' '.join(c.replace(HOME, '~') for c in sess) + f' < {SESSION_Q}')
        print('# next: tests/cases_check.py default (the same loop without -i); if that passes, the difference is in the session\'s loop')
    print('FAILED' if not ok else 'all checks passed')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
