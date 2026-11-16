#!/usr/bin/env python3
"""Read ds-completion --record files (docs/record.md): a table of chosen fields per group of
runs, with the spread of the repeats; a comparison against one configuration; a condition.

  scripts/records.py FILE.json... --by config,prompt --show counters.gen_tps,spec.steps
                     [--vs config=all] [--same output.hash --within prompt]
                     [--name-label REGEX]

--by     the label keys that make a group (a record's "label", from --record-label). The runs of
         a group are repeats of each other. Default: every label key the records have, but "rep".
--show   fields, comma-separated: a path into the record ("counters.gen_tps", "experts.placed.0"
         for a list's element), or a ratio of two ("spec.accepted/spec.steps"; the record holds
         totals, not ratios). Per group: the mean and, with repeats, the spread as
         (max - min) / mean. A field that is not a number: its value if the group agrees.
--vs     KEY=VALUE, a key of --by: every group against the group that has VALUE for KEY and the
         same other keys. A difference is printed only if it is larger than the spread of both
         groups' repeats; else "~" (inside the spread), or "?" when a side has one run and so
         no spread. Two repeats give a low estimate of the spread: a difference just outside it
         is not yet a result.
--same   a field that must have one value among all the runs that agree on the --within keys
         (default: the --by keys; '': all the runs), e.g. the output's hash per prompt whatever
         the configuration.
         Exit code 1 if it does not.
--name-label  for records written before --record-label (README steps 124-125): a regular
         expression with named groups, searched in the file's path; the groups become the label.

An empty file is a run that did not finish (the record is written at the end): reported, exit 1.
"""
import argparse
import collections
import json
import re
import sys


def field(rec, path):
    """the value at a dotted path (list elements by index), or None"""
    v = rec
    for part in path.split('.'):
        if isinstance(v, dict) and part in v:
            v = v[part]
        elif isinstance(v, list) and part.isdigit() and int(part) < len(v):
            v = v[int(part)]
        else:
            return None
    return v


def value(rec, spec):
    """a field, or the ratio of two fields"""
    if '/' in spec:
        a, b = (field(rec, p) for p in spec.split('/', 1))
        ok = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b))
        return a / b if ok and b else None
    return field(rec, spec)


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def stats(vals):
    """(mean, spread) of numbers: spread = (max - min) / |mean|, None with one run"""
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, None
    return m, ((max(vals) - min(vals)) / abs(m) if m else 0.0)


def fmt(x):
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f'{x:.4g}' if abs(x) < 100 else f'{x:.1f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+')
    ap.add_argument('--by')
    ap.add_argument('--show', default='counters.gen_tps')
    ap.add_argument('--vs')
    ap.add_argument('--same')
    ap.add_argument('--within')
    ap.add_argument('--name-label')
    a = ap.parse_args()

    bad = 0
    recs = []
    for path in a.files:
        text = open(path).read()
        if not text.strip():
            print(f'did not finish (empty record): {path}')
            bad = 1
            continue
        r = json.loads(text)
        label = dict(r.get('label') or {})
        if a.name_label:
            m = re.search(a.name_label, path)
            if not m:
                sys.exit(f'--name-label does not match {path}')
            label.update(m.groupdict())
        recs.append((label, r, path))
    if not recs:
        sys.exit('no records')

    keys = a.by.split(',') if a.by else [k for k in dict.fromkeys(k for l, _, _ in recs for k in l) if k != 'rep']
    for label, _, path in recs:
        missing = [k for k in keys if k not in label]
        if missing:
            sys.exit(f'{path}: no label {", ".join(missing)} (its label: {label}); --by or --name-label')
    shows = a.show.split(',')
    groups = collections.OrderedDict()
    for label, r, path in recs:
        groups.setdefault(tuple(label[k] for k in keys), []).append(r)

    # per group and field: ('num', mean, spread) or ('text', what to print)
    cells = {}
    for g, rs in groups.items():
        for s in shows:
            vals = [value(r, s) for r in rs]
            if all(is_num(v) for v in vals):
                cells[g, s] = ('num',) + stats(vals)
            else:
                texts = list(dict.fromkeys(json.dumps(v) if not isinstance(v, str) else v for v in vals))
                cells[g, s] = ('text', texts[0] if len(texts) == 1 else f'{len(texts)} values')

    base_key = base_val = None
    if a.vs:
        base_key, base_val = a.vs.split('=', 1)
        if base_key not in keys:
            sys.exit(f'--vs: {base_key} is not one of the group keys ({", ".join(keys)})')
        ki = keys.index(base_key)
        if not any(g[ki] == base_val for g in groups):
            sys.exit(f'--vs: no group has {base_key}={base_val}')

    def versus(g, s):
        if g[ki] == base_val:
            return ''
        b = g[:ki] + (base_val,) + g[ki + 1:]
        if b not in groups or cells[g, s][0] != 'num' or cells[b, s][0] != 'num':
            return ''
        (_, m, sp), (_, bm, bsp) = cells[g, s], cells[b, s]
        if not bm:
            return ''
        d = m / bm - 1
        if d == 0:
            return '='
        if sp is None or bsp is None:
            return f'{100 * d:+.1f}% ?'
        noise = max(sp, bsp)
        return f'{100 * d:+.1f}%' if abs(d) > noise else f'~ (spread {100 * noise:.1f}%)'

    header = keys + ['runs']
    for s in shows:
        header += [s, 'spread'] + ([f'vs {base_val}'] if a.vs else [])
    rows = []
    for g, rs in groups.items():
        row = list(g) + [str(len(rs))]
        for s in shows:
            c = cells[g, s]
            if c[0] == 'num':
                row += [fmt(c[1]), '' if c[2] is None else f'{100 * c[2]:.1f}%']
            else:
                row += [c[1], '']
            if a.vs:
                row.append(versus(g, s))
        rows.append(row)
    width = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    for r in [header] + rows:
        print('  '.join(c.ljust(w) if i < len(keys) else c.rjust(w) for i, (c, w) in enumerate(zip(r, width))).rstrip())
    if a.vs:
        print(f'# vs {base_val}: a difference of the means, printed if larger than the spread of both groups\' repeats; '
              '~: inside it; ?: a side has one run, no spread; =: equal')

    if a.same:
        within = keys if a.within is None else [k for k in a.within.split(',') if k]  # '': all the runs together
        seen = collections.OrderedDict()
        for label, r, path in recs:
            v = value(r, a.same)
            seen.setdefault(tuple(label.get(k, '') for k in within), collections.OrderedDict()).setdefault(json.dumps(v), []).append(path)
        fails = 0
        for g, vals in seen.items():
            if len(vals) > 1 or 'null' in vals:
                fails += 1
                print(f'--same {a.same}: {" ".join(g) or "all the runs"}: ' + ('; '.join(f'{v} in {len(p)} runs (e.g. {p[0]})' for v, p in vals.items())))
        n = sum(len(p) for vals in seen.values() for p in vals.values())
        print(f'# --same {a.same} within {",".join(within) or "all the runs"}: ' + (f'FAILED for {fails} of {len(seen)}' if fails else f'holds for all {len(seen)} ({n} runs)'))
        bad = bad or (1 if fails else 0)
    sys.exit(bad)


if __name__ == '__main__':
    main()
