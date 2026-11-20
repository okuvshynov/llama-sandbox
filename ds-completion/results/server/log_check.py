#!/usr/bin/env python3
"""README step 166: the --log of server/ds_server.py with the model.

  python3 results/server/log_check.py SERVER_LOG > results/server/log.txt

Starts the server with --log (the fastest single-machine settings, a free port) and sends:
four requests with one long system prompt (two conversations, which set a saved point; a
third, which returns to it; the third's next turn, which goes on from the context), one that
is refused, a stream whose client leaves after 3 s, and one more. Prints the request log's
lines. The server's and the engine's own lines go to SERVER_LOG.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

H = os.path.expanduser('~')
M = H + '/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf'
req_log = tempfile.NamedTemporaryFile(prefix='log_check-', suffix='.jsonl')
cmd = [sys.executable, 'server/ds_server.py', '--port', '0', '--log', req_log.name, '--', 'bin-hip/ds-completion', '-m', M,
       '--rt', H + '/llms/ds-1017-fp8/dense-fp8.gguf', '--placement', 'placements/study30.txt',
       '--dspark', H + '/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf', '--spec', '--canon',
       '--cache-loads', '800', '--cache-period', '8', '--cache-wait', '--requests']
srv = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=open(sys.argv[1], 'w'))
port = int(srv.stdout.readline().split()[1])
base = f'http://127.0.0.1:{port}'
SYS = 'You answer in one short line. Background, not to be repeated: ' + 'The quick brown fox jumps over the lazy dog. ' * 150


def post(body):
    r = urllib.request.Request(base + '/v1/messages', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=300).read())


def ask(msgs):
    return post({'max_tokens': 60, 'system': SYS, 'messages': msgs})


ask([{'role': 'user', 'content': 'Color: name one.'}])
ask([{'role': 'user', 'content': 'Fruit: name one.'}])
m = ask([{'role': 'user', 'content': 'City: name one.'}])
ask([{'role': 'user', 'content': 'City: name one.'}, {'role': 'assistant', 'content': m['content']}, {'role': 'user', 'content': 'And another?'}])
try:
    post({'max_tokens': 5, 'messages': []})
except Exception as e:
    print('# the refused one:', e)
body = json.dumps({'max_tokens': 4000, 'stream': True, 'system': SYS, 'thinking': {'type': 'adaptive'},
                   'messages': [{'role': 'user', 'content': 'Ignore the one-line rule: write a long essay about rain.'}]}).encode()
c = socket.create_connection(('127.0.0.1', port))
c.sendall(b'POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n' % len(body) + body)
time.sleep(3)
c.close()
ask([{'role': 'user', 'content': 'Tree: name one.'}])
print(open(req_log.name, encoding='utf-8').read(), end='')
srv.terminate()
srv.wait()
