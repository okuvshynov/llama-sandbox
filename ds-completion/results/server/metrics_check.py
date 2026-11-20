#!/usr/bin/env python3
"""README step 165: GET /metrics and /health of server/ds_server.py with the model.

  results/server/metrics_check.py SERVER_LOG > results/server/metrics.txt

Starts the server (the fastest single-machine settings, a free port), sends four requests with
one long system prompt (two conversations, which set a saved point; a third, which returns to
it; the third's next turn, which goes on from the context) and one that is refused, then prints
the usage of each and the metrics page. The server's and the engine's lines go to SERVER_LOG.
"""
import json, subprocess, sys, time, urllib.request, os
H=os.path.expanduser('~')
M=H+'/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf'
cmd=[sys.executable,'server/ds_server.py','--port','0','--','bin-hip/ds-completion','-m',M,'--rt',H+'/llms/ds-1017-fp8/dense-fp8.gguf','--placement','placements/study30.txt',
     '--dspark',H+'/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf','--spec','--canon','--cache-loads','800','--cache-period','8','--cache-wait','--requests']
srv=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=open(sys.argv[1],'w'))
port=int(srv.stdout.readline().split()[1]); base=f'http://127.0.0.1:{port}'
def get(p):
    return urllib.request.urlopen(base+p,timeout=10).read().decode()
def post(body):
    r=urllib.request.Request(base+'/v1/messages',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(r,timeout=300).read())
SYS='You answer in one short line. Background, not to be repeated: '+'The quick brown fox jumps over the lazy dog. '*150
def ask(msgs): 
    m=post({'max_tokens':60,'system':SYS,'messages':msgs}); print('usage',m['usage'],m['stop_reason'],flush=True); return m
print('health',get('/health'))
ask([{'role':'user','content':'Color: name one.'}])
ask([{'role':'user','content':'Fruit: name one.'}])
m=ask([{'role':'user','content':'City: name one.'}])
ask([{'role':'user','content':'City: name one.'},{'role':'assistant','content':m['content']},{'role':'user','content':'And another?'}])
try:
    post({'max_tokens':5,'messages':[]})
except Exception as e: print('refused',e)
print('---metrics'); print(get('/metrics'))
srv.terminate(); srv.wait()
