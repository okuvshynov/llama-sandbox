#!/usr/bin/env python3
"""The model's files in the page cache, in a state in which the engine starts fast (README step 204).

  scripts/model_cache.py [--waits N] [--status] [-m <first shard>]

A start of the engine with --cache-wait waits for the registration of the 289 GB of expert
tensors with the GPUs, and how long that takes is a property of the page cache's pages: 75 s
when the files came in by one sequential read into free memory, 285 s when page faults brought
them in (README steps 198-199). So, before runs that start the engine many times:

  1. what is not in the page cache is read, one file after another, sequentially;
  2. bin-hip/reg-probe registers a sample of the expert tensors (4 GiB, a few seconds) and gives
     a rate per shard;
  3. a shard that registers below 2 GB/s is dropped from the page cache (posix_fadvise, no root)
     and read again, if that pays: when N starts would wait longer for it than the reading takes
     (--waits N, default 16: what `scripts/check.py speed` starts; check.py passes its run's number). Then the probe again.
     Only below 2 GB/s, because reading again gives 3 to 4 GB/s and no more: the 6 GB/s of a
     read right after a boot do not come back, and a shard at that rate would lose them.

--status: steps 1 and 3 left out: what is cached and, if all of it is, the probe's rates.
scripts/check.py calls prepare() before the first check that needs the model.
Nothing is dropped while an engine or a test program runs (it would read the files again
through page faults, which is the slow state).
"""
import glob
import os
import re
import subprocess
import sys
import time

HOME = os.path.expanduser('~')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = HOME + '/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf'
PROBE = 'bin-hip/reg-probe'
SLOW = 2.0      # GB/s (the probe's): below it a shard is read again. Page faults over hours gave 1.0 (step 199)
AGAIN = 3.0     # GB/s: what a shard read again registers at, at least (3.1 to 4.2 measured, steps 199 and 204). A read
                # right after a boot gives 6 and more, and a shard at that rate read again loses it (step 204): hence SLOW
DISK = 2.5      # GB/s: a sequential read from storage (474 GiB in 188 s, step 199)
USERS = ('ds-completion', 'ds-test', 'reg-probe')


def shards(first):
    m = re.match(r'(.*-)\d{5}(-of-\d{5}\.gguf)$', first)
    return sorted(glob.glob(m.group(1) + '?????' + m.group(2))) if m else [first]


def cached(files):
    """{file: (bytes in the page cache, size)}"""
    out = subprocess.run(['fincore', '--bytes', '--noheadings', '-o', 'RES,SIZE,FILE'] + files, capture_output=True, text=True).stdout
    r = {}
    for l in out.splitlines():
        res, size, f = l.split(None, 2)
        r[f] = (int(res), int(size))
    return r


def read(f, say):
    """one sequential pass over a file: into the page cache"""
    t0, n, buf = time.time(), 0, bytearray(16 << 20)
    fd = os.open(f, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
    say(f'reading {os.path.basename(f)} ({os.fstat(fd).st_size / 1e9:.1f} GB)')  # (before the work: a line per file as it starts)
    while True:
        k = os.readv(fd, [buf])
        if k <= 0:
            break
        n += k
    os.close(fd)
    return n, time.time() - t0


def drop(f):
    fd = os.open(f, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def probe(first):
    """-> ({shard file: (GB/s, GB of expert tensors)}, the seconds the whole registration would take); None if the probe fails"""
    r = subprocess.run([PROBE, first], capture_output=True, text=True, cwd=ROOT)
    per, seconds = {}, None
    for l in r.stdout.splitlines():
        w = l.split(None, 3)
        if w and w[0] == 'shard':
            per[w[3]] = (float(w[1]), float(w[2]))
        elif w and w[0] == 'rate':
            seconds = float(l.split()[5])
    if r.returncode != 0 or seconds is None:
        return None
    return per, seconds


def running():
    out = subprocess.run(['pgrep', '-l', '-f', '|'.join('bin(-hip)?/' + u for u in USERS)], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if str(os.getpid()) != l.split()[0] and 'pgrep' not in l]


def prepare(waits=16, first=MODEL, say=print, status=False):
    """-> the seconds one registration is expected to take now; None if that could not be measured"""
    files = shards(first)
    c = cached(files)
    missing = [f for f in files if f not in c or c[f][0] < c[f][1] - (1 << 20)]
    have, size = sum(v[0] for v in c.values()), sum(v[1] for v in c.values())
    say(f'{have / 2**30:.0f} of {size / 2**30:.0f} GiB of the model in the page cache' + (f'; not whole: {len(missing)} of {len(files)} files' if missing else ''))
    if missing and not status:
        for f in missing:
            n, s = read(f, say)
            say(f'  {n / 1e9:.1f} GB in {s:.0f} s = {n / 1e9 / max(s, 1e-9):.2f} GB/s')
    if missing and status:
        say('the registration\'s rate is not measured: the probe would read the missing parts through page faults')
        return None
    if not os.path.exists(os.path.join(ROOT, PROBE)):
        say(f'no {PROBE} (make HIP=1): the registration\'s rate is not measured')
        return None
    p = probe(first)
    if p is None:
        say(f'{PROBE} failed: the registration\'s rate is not measured')
        return None
    per, seconds = p
    say(f'registering the expert tensors would take {seconds:.0f} s (' + ', '.join(f'{v[0]:.1f}' for v in per.values()) + ' GB/s by shard)')
    if status:
        return seconds
    # a slow shard again, if `waits` starts lose more on it than the reading takes
    slow = [f for f, (rate, gb) in per.items() if rate < SLOW and waits * gb * (1 / rate - 1 / AGAIN) > os.path.getsize(f) / 1e9 / DISK]
    if not slow:
        return seconds
    busy = running()
    if busy:
        say(f'{len(slow)} shards register slowly, and are left as they are: a program that maps the model runs ({busy[0]})')
        return seconds
    say(f'{len(slow)} shards register slowly (their pages came by page faults, README step 199): dropped from the page cache and read again')
    for f in slow:
        drop(f)
    for f in slow:
        n, s = read(f, say)
        say(f'  {n / 1e9:.1f} GB in {s:.0f} s = {n / 1e9 / max(s, 1e-9):.2f} GB/s')
    p = probe(first)
    if p is None:
        return None
    per, seconds = p
    say(f'now registering would take {seconds:.0f} s (' + ', '.join(f'{v[0]:.1f}' for v in per.values()) + ' GB/s by shard)')
    return seconds


def main():
    a = sys.argv[1:]
    if [x for i, x in enumerate(a) if x not in ('--waits', '--status', '-m') and (i == 0 or a[i - 1] not in ('--waits', '-m'))]:
        print(__doc__.split('\n\n')[1], file=sys.stderr)
        return 2
    waits = int(a[a.index('--waits') + 1]) if '--waits' in a else 16
    first = a[a.index('-m') + 1] if '-m' in a else MODEL
    prepare(waits, first, lambda s: print(s, flush=True), status='--status' in a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
