import sys,time
t0=time.time()
for l in sys.stdin.buffer:
    sys.stdout.write('%7.2f  %s'%(time.time()-t0,l.decode('utf8','replace')[:150].rstrip('\n')+'\n')); sys.stdout.flush()
print('%7.2f  (end of output)'%(time.time()-t0))
