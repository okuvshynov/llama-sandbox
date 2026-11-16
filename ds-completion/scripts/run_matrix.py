#!/usr/bin/env python3
"""Run ds-completion over prompts x configurations x repeats, one --record file per run
(docs/record.md). It starts runs and nothing else: scripts/records.py reads the records.

  scripts/run_matrix.py MATRIX [--prompts a,b] [--reps N] [--out DIR] [--keep-out] [--dry-run] [--resume]

MATRIX is a text file, one statement per line, '#' comments; words are split as a shell does,
and ~, $NAME and {prompt} (the prompt's name) are replaced in every word:

  set NAME VALUE       a default for $NAME, used if the environment does not have it
  bin PATH             the program (default bin-hip/ds-completion)
  prompts FILE         lines "name|mode|n|text" (mode: raw, chat or think; n: tokens to generate)
  out DIR              where the records go (records of a one-off: under data/, which git ignores)
  reps N               repeats of every run (default 1)
  base ARG...          arguments of every run
  config NAME [reps=N] [ENV=VALUE...] [ARG...]
                       a configuration: its name, then environment variables for its runs (words
                       NAME=VALUE before the first argument), then its own arguments

Order: repeat by repeat, within it prompt by prompt, within it the configurations in turn, so
that a slow drift of the machine falls on every configuration alike. Each run gets
  -m $M -p TEXT --MODE -n N  BASE...  CONFIG...
  --record DIR/CONFIG-PROMPT-REP.json --record-label config=CONFIG,prompt=PROMPT,rep=REP
Its stdout is dropped (the record has the output's hash) unless --keep-out, which writes it to
DIR/CONFIG-PROMPT-REP.out (to compare with a stored output file); its stderr is kept as
DIR/CONFIG-PROMPT-REP.err only if the run failed or left no record. DIR/matrix.txt gets the
'#' header (this command, commit, GPU clock, date) and a copy of MATRIX.
A DIR that already has records is refused, or with --resume only the missing runs are made.
Exit code 1 if a run failed.
"""
import argparse
import datetime
import glob
import json
import os
import shlex
import subprocess
import sys

say = lambda *a: print(*a, flush=True)


def sclk():
    """the cards' clock setting, as the record writes it: a level where it is fixed, else the mode"""
    out = []
    for dev in sorted(glob.glob('/sys/class/drm/card[0-9]*/device')):
        try:
            if not os.path.exists(dev + '/vendor') or open(dev + '/vendor').read().strip() != '0x1002':
                continue  # (a connector such as card0-DP-1, or another vendor's card)
            mode = open(dev + '/power_dpm_force_performance_level').read().strip()
            level = [l.split(':')[0] for l in open(dev + '/pp_dpm_sclk') if '*' in l]
            out.append(level[0] if mode == 'manual' and level else mode)
        except OSError:
            out.append('unknown')
    return ' '.join(out) or 'no AMD card'


def git(*args):
    return subprocess.run(['git'] + list(args), capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('matrix')
    ap.add_argument('--prompts', help='only these prompt names, comma-separated')
    ap.add_argument('--reps', type=int)
    ap.add_argument('--out')
    ap.add_argument('--keep-out', action='store_true', help="keep each run's stdout as CONFIG-PROMPT-REP.out")
    ap.add_argument('--dry-run', action='store_true', help='print the commands, run nothing')
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()

    sets = {}
    m = {'bin': 'bin-hip/ds-completion', 'prompts': None, 'out': None, 'reps': 1, 'base': [], 'configs': []}

    def expand(word, prompt=None):
        for k, v in sets.items():
            os.environ.setdefault(k, v)
        w = os.path.expanduser(os.path.expandvars(word))
        if '$' in w:
            sys.exit(f'{a.matrix}: {word}: a variable is not set (set NAME VALUE, or the environment)')
        return w.replace('{prompt}', prompt) if prompt is not None else w

    for n, line in enumerate(open(a.matrix), 1):
        words = shlex.split(line, comments=True)
        if not words:
            continue
        kind, rest = words[0], words[1:]
        where = f'{a.matrix}:{n}'
        if kind == 'set' and len(rest) == 2:
            sets[rest[0]] = os.path.expanduser(os.path.expandvars(rest[1]))
        elif kind in ('bin', 'prompts', 'out') and len(rest) == 1:
            m[kind] = rest[0]
        elif kind == 'reps' and len(rest) == 1:
            m['reps'] = int(rest[0])
        elif kind == 'base':
            m['base'] += rest
        elif kind == 'config' and rest:
            c = {'name': rest[0], 'reps': None, 'env': {}, 'args': []}
            if any(ch in c['name'] for ch in ',=/ '):
                sys.exit(f'{where}: a configuration\'s name cannot hold , = / or a space')
            for w in rest[1:]:
                if not c['args'] and not w.startswith('-') and '=' in w:
                    k, v = w.split('=', 1)
                    if k == 'reps':
                        c['reps'] = int(v)
                    else:
                        c['env'][k] = v
                else:
                    c['args'].append(w)
            if c['name'] in [x['name'] for x in m['configs']]:
                sys.exit(f'{where}: configuration {c["name"]} twice')
            m['configs'].append(c)
        else:
            sys.exit(f'{where}: not understood: {line.strip()}')
    if a.reps:
        m['reps'] = a.reps
    if a.out:
        m['out'] = a.out
    if not m['prompts'] or not m['out'] or not m['configs']:
        sys.exit(f'{a.matrix}: needs prompts, out (or --out) and at least one config')

    prompts = []
    for line in open(expand(m['prompts'])):
        if line.strip() and not line.startswith('#'):
            name, mode, n, text = line.rstrip('\n').split('|', 3)
            prompts.append((name, mode, n, text))
    if a.prompts:
        want = a.prompts.split(',')
        unknown = [w for w in want if w not in [p[0] for p in prompts]]
        if unknown:
            sys.exit(f'--prompts: not in {m["prompts"]}: {", ".join(unknown)}')
        prompts = [p for p in prompts if p[0] in want]

    out = expand(m['out'])
    runs = []
    for rep in range(1, m['reps'] + 1):
        for name, mode, n, text in prompts:
            for c in m['configs']:
                if rep > (c['reps'] or m['reps']):
                    continue
                stem = f'{out}/{c["name"]}-{name}-{rep}'
                cmd = [expand(m['bin']), '-m', expand('$M'), '-p', text] + ([] if mode == 'raw' else ['--' + mode]) + ['-n', n]
                cmd += [expand(w, name) for w in m['base'] + c['args']]
                cmd += ['--record', stem + '.json', '--record-label', f'config={c["name"]},prompt={name},rep={rep}']
                runs.append((stem, cmd, {k: expand(v, name) for k, v in c['env'].items()}, f'{c["name"]} {name} {rep}'))

    if a.dry_run:
        for stem, cmd, env, what in runs:
            say(' '.join([f'{k}={shlex.quote(v)}' for k, v in env.items()] + [shlex.quote(w) for w in cmd]))
        say(f'# {len(runs)} runs')
        return

    done = lambda stem: os.path.exists(stem + '.json') and os.path.getsize(stem + '.json') > 0
    os.makedirs(out, exist_ok=True)
    if glob.glob(out + '/*.json') and not a.resume:
        sys.exit(f'{out} has records already: another --out, or --resume to make only the missing runs')
    with open(out + '/matrix.txt', 'a' if a.resume else 'w') as f:
        changed = '+changes' if git('status', '--porcelain', '--untracked-files=no', '.') else ''
        f.write(f'# scripts/run_matrix.py {" ".join(shlex.quote(w) for w in sys.argv[1:])}; commit {git("rev-parse", "--short", "HEAD")}{changed}, '
                f'sclk {sclk()}, {datetime.datetime.now().astimezone().isoformat(timespec="seconds")}\n')
        f.write(f'# {len(runs)} runs: {len(prompts)} prompts x {len(m["configs"])} configurations x up to {m["reps"]} repeats; '
                f'records: CONFIG-PROMPT-REP.json; read with scripts/records.py\n')
        f.write(open(a.matrix).read())

    failed = 0
    for i, (stem, cmd, env, what) in enumerate(runs, 1):
        if a.resume and done(stem):
            say(f'[{i}/{len(runs)}] {what}: has a record, skipped')
            continue
        say(f'[{i}/{len(runs)}] running {what} ...')
        with open(stem + '.err', 'w') as err, open(stem + '.out' if a.keep_out else os.devnull, 'wb') as out_f:
            rc = subprocess.run(cmd, env={**os.environ, **env}, stdin=subprocess.DEVNULL, stdout=out_f, stderr=err).returncode
        if rc == 0 and done(stem):
            os.remove(stem + '.err')
            r = json.load(open(stem + '.json'))
            c = r['counters']
            read = (r.get('read_bytes') or {}).get('gen')
            say(f'    {c["gen_tokens"]} tokens, {c["gen_tps"]:.2f} t/s' + (f', read {read} bytes from storage in generation' if read else ''))
        else:
            failed += 1
            say(f'    FAILED: exit code {rc}, {"a" if done(stem) else "no"} record; stderr kept: {stem}.err')
    say(f'# {len(runs)} runs, {failed} failed; records in {out}/')
    say(f'#   scripts/records.py {out}/*.json --by prompt,config --show counters.gen_tps --same output.hash --within prompt')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
