import os,sys,time,subprocess
def snap():
    d={}
    for p in os.listdir('/proc'):
        if not p.isdigit(): continue
        try:
            for t in os.listdir(f'/proc/{p}/task'):
                s=open(f'/proc/{p}/task/{t}/stat').read()
                name=s[s.index('(')+1:s.rindex(')')]; f=s[s.rindex(')')+2:].split()
                d[t]=(int(f[11]),int(f[12]),name, int(f[36]))
        except Exception: pass
    return d
a=snap(); t0=time.time()
subprocess.run(sys.argv[1:],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
dt=time.time()-t0; b=snap()
rows=[]
for k,v in b.items():
    o=a.get(k,(0,0,'',0)); new=k not in a; u=(v[0]-o[0])/100; s=(v[1]-o[1])/100
    if u+s>=0.1: rows.append((u+s,u,s,k,v[3]))
print('wall %.1f s'%dt)
for r in sorted(rows,reverse=True)[:15]: print('%6.2f s (user %.2f sys %.2f)  %s tid %s last cpu %d'%(r[0],r[1],r[2],b[r[3]][2],r[3],r[4]))
