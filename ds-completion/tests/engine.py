"""One engine with --requests for the checks: started once per configuration, asked many times
(README step 205; docs/requests.md is the protocol).

A check that needs a reply does not start a process of its own: it names a configuration (what
cannot vary between requests: the placement, the cache, the draft and speculation, --spec-n) and
sends a request (what can: n, temp, top_p, seed, keep). tests/cases_check.py runs a table of such
cases; this file is what it and other checks share: the files' paths, the configurations, the
engine as an object, the held-out prompts and their rendering, the hash of a reply's ids.
"""
import collections
import os
import subprocess
import time

HOME = os.path.expanduser('~')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = HOME + '/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf'
F8 = HOME + '/llms/ds-1017-fp8/dense-fp8.gguf'
D = HOME + '/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf'
BOS, EOS = '<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>'

BASE = ['bin-hip/ds-completion', '-m', M, '--rt', F8, '--canon']
CACHE = ['--cache-loads', '800', '--cache-period', '8', '--cache-wait']
# A configuration: the engine's arguments that a request cannot change. --canon in all of them:
# only with it a reply does not depend on where an expert ran, so stored ids can be expected.
CONFIGS = {
    # the fastest settings (QUICKSTART): the placement, the adaptive cache, the draft and speculation
    'full': BASE + ['--placement', 'placements/study30.txt'] + CACHE + ['--dspark', D, '--spec'],
    # the same, every step verifying all five drafts: another sequence of steps, the same replies (step 151)
    'spec-n5': BASE + ['--placement', 'placements/study30.txt'] + CACHE + ['--dspark', D, '--spec', '--spec-n', '5'],
    # the plain loop, one token per step, on a fixed placement (no registration: it starts in ~13 s)
    'plain': BASE + ['--placement', 'placements/study30.txt'],
    # the plain loop on another fixed placement, and with the cache, which moves experts during a reply
    'plain-oracle': BASE + ['--placement', 'placements/oracle-concurrency.txt'],
    'plain-cache': BASE + ['--placement', 'placements/study30.txt'] + CACHE,
}

Reply = collections.namedtuple('Reply', 'ids text prompt end')  # the ids, their bytes, the fields of the "prompt" and "end" lines


def render(mode, user):
    """a held-out prompt as the engine's --chat / --think render it"""
    if mode == 'think':
        return (BOS + '<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n'
                + '<｜User｜>' + user + '<｜Assistant｜><think>')
    return BOS + '<｜User｜>' + user + '<｜Assistant｜></think>'


def fnv1a64_ids(ids):
    """the hash --record writes as output.hash"""
    h = 0xcbf29ce484222325
    for i in ids:
        for b in range(4):
            h = ((h ^ ((i >> (8 * b)) & 0xff)) * 0x100000001b3) & 0xffffffffffffffff
    return 'fnv1a64:%016x' % h


def heldout():
    """{name: (mode, n, text)}: results/spec-heldout/prompts.txt"""
    prompts = {}
    for l in open(os.path.join(ROOT, 'results/spec-heldout/prompts.txt'), encoding='utf-8'):
        name, mode, n, text = l.rstrip('\n').split('|', 3)
        prompts[name] = (mode, int(n), text)
    return prompts


class Engine:
    """ds-completion --requests as a child: Engine(config, err_path), then request() any number of times, then close()"""

    def __init__(self, config, err=None):
        self.cmd = CONFIGS[config] + ['--requests']
        self.err = open(err, 'wb') if err else subprocess.DEVNULL  # the engine's own lines ("# ...")
        t0 = time.time()
        self.p = subprocess.Popen(self.cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err)
        self.ready = self.p.stdout.readline().decode().strip() == 'ready'
        self.load_s = time.time() - t0
        self.said = ''  # what the engine answered when a request got no reply

    def command(self):
        return ' '.join(c.replace(HOME, '~') for c in self.cmd)

    def request(self, head, text):
        """head: the header without bytes= ("gen n=300 temp=1 seed=1"). -> Reply, or None: an error line, or the engine is gone (see .said, .alive())"""
        body = text.encode('utf-8')
        try:
            self.p.stdin.write(f'{head} bytes={len(body)}\n'.encode() + body)
            self.p.stdin.flush()
        except (BrokenPipeError, ValueError):
            self.said = '(closed)'
            return None
        ids, out, pr = [], b'', {}
        while True:
            w = self.p.stdout.readline().decode().split()
            if not w or w[0] == 'error':
                self.said = ' '.join(w) or '(closed)'
                return None
            if w[0] == 'prompt':
                pr = dict(f.split('=') for f in w[1:])
            elif w[0] == 'tok':
                ids.append(int(w[1]))
                out += bytes.fromhex(w[2]) if len(w) > 2 else b''
            elif w[0] == 'end':
                return Reply(ids, out, pr, dict(f.split('=') for f in w[1:]))

    def alive(self):
        return self.p.poll() is None

    def close(self):
        """ends the input; -> the engine's exit code"""
        try:
            self.p.stdin.close()
        except BrokenPipeError:
            pass
        rc = self.p.wait()
        if self.err is not subprocess.DEVNULL:
            self.err.close()
        return rc
