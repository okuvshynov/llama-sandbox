#!/usr/bin/env python3
"""Drafting ahead, estimated from the draft's token logs (README step 106; not built).

While a verify step runs, the next draft could start on another die if its anchor is guessed:
every verified draft accepted, and the main model's own next token equal to the draft's next
guess (drafted token n + 1 when n < 5 were verified). Per held-out prompt: the steps where that
holds (A), and an upper bound that also counts every fully accepted 5-draft step, as if a sixth
draft token existed and were always right (B). The gain: a hit takes min(draft, overlap) ms off
the next step; overlap 5.5 ms (the draft reads the main model's hidden states at layers 37-39 of
40, so it can start only when the verify's last layer does) and 8 ms (the whole draft: a bound).

Input: the DS_DRAFT_TOKEN_LOG files and .err files of README step 92's held-out run.

    python3 scripts/draft_ahead_sim.py [results/decode-next/draft-vocab/]
"""
import re
import sys

D = sys.argv[1] if len(sys.argv) > 1 else 'results/decode-next/draft-vocab/'
names=['tcp-udp','py-lru','cpp-matrix','sql-report','geometry','zh-travel','ja-email','summary-json']
def steps(n):
    out=[]
    for l in open(D+n+'.log'):
        if l.startswith('#') or not l.strip(): continue
        h,dr,em=l.split('|'); nv,k,v,d=h.split()
        out.append((int(nv),int(k),float(v),float(d),[int(x) for x in dr.split()],[int(x) for x in em.split()]))
    return out
def gen(n):
    m=re.search(r'generation: (\d+) tokens in ([0-9.]+) s',open(D+f'vocab-{n}.err').read()); return int(m[1]),float(m[2])
print(f"{'prompt':13s} steps  drafted  n=5   all-acc  hitA   hitB(ub)  by n: hit/steps (n=0..5)                       gain%: 5.5ms A / B, 8ms A / B")
tot=[]
for n in names:
    st=steps(n); ntok,secs=gen(n)
    S=len(st); drafted=sum(1 for s in st if s[4]); allacc=sum(1 for s in st if s[4] and s[1]==s[0])
    hitA=hitB=0; byn=[[0,0] for _ in range(6)]
    for nv,k,v,d,dr,em in st:
        if not dr: continue
        byn[nv][1]+=1
        if k==nv:
            if nv<len(dr):
                if em[nv]==dr[nv]: hitA+=1; hitB+=1; byn[nv][0]+=1
            else: hitB+=1  # n = block: a 6th draft would be needed; upper bound: always right
    n5=byn[5][1]
    # a hit saves the NEXT step's draft time on the critical path (min(draft, overlap)); only if the next step drafts (it does unless backoff: ignore)
    g=lambda h,o: h*min(o,8.0)/ (secs*1000)*100
    row=f"{n:13s} {S:5d} {drafted:7d} {n5:5d} {allacc:8d} {hitA:5d} {hitB:8d}   "+' '.join(f"{a}/{b}" for a,b in byn).ljust(46)+f" {g(hitA,5.5):5.2f} / {g(hitB,5.5):5.2f}   {g(hitA,8):5.2f} / {g(hitB,8):5.2f}"
    print(row); tot.append((g(hitA,5.5),g(hitB,5.5),g(hitA,8),g(hitB,8),hitA/S,hitB/S))
m=[sum(t[i] for t in tot)/len(tot) for i in range(6)]
print(f"mean gain%: 5.5 ms overlap A {m[0]:.2f} / B {m[1]:.2f};  8 ms A {m[2]:.2f} / B {m[3]:.2f};  hit share of steps A {m[4]:.1%} B {m[5]:.1%}")
