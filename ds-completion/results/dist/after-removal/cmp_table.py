import sys
def rows(p):
    out = []
    for l in open(p):
        f = l.split()
        if len(f) > 3 and f[0].isdigit() and f[1].isdigit():
            out.append(f[:-1])
    return out
a, b = rows(sys.argv[1]), rows(sys.argv[2])
d = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
print(f"{len(a)} rows new, {len(b)} stored, {d} differ", flush=True)
