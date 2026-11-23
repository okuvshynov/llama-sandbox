# session stdout against the stored replies (joined by \x1e or by newline); a fresh run's stdout against reply k
import sys
mode, new, stored = sys.argv[1:4]
n, s = open(new, 'rb').read(), open(stored, 'rb').read()
if mode == 'session':
    ok = n.rstrip(b'\n') == s.replace(b'\x1e', b'\n').rstrip(b'\n')
else:
    k = int(sys.argv[4])
    r = s.split(b'\x1e') if b'\x1e' in s else s.rstrip(b'\n').split(b'\n')
    ok = n.rstrip(b'\n') == r[k - 1].rstrip(b'\n')
    if not ok and b'\x1e' not in s:  # replies with newlines in a file joined by newlines: reply 3 is the file's end
        t, x = s.rstrip(b'\n'), n.rstrip(b'\n')
        ok = len(x) > 0 and (t.endswith(b'\n' + x) if k == 3 else (b'\n' + x + b'\n') in t)
print('same' if ok else 'DIFFERS', flush=True)
