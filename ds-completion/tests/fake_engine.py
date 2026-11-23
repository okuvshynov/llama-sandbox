#!/usr/bin/env python3
"""A stand-in for `ds-completion --requests` (docs/requests.md) for tests/test_server.py:
no model. It answers the i-th request with the i-th entry of a JSON file, a list of
{"text": ..., "stop": "eos"|"length", "delay": seconds, "kept": tokens, "saved": tokens} (SIGUSR1 during the
delay cancels the reply, as the engine does; "how" in the prompt line is continue with kept tokens, else
empty), and appends every request it gets (header and prompt) to a log file as one JSON line.
An entry {"error": message} is answered with the engine's "error <message>" line; an entry
{"exit": code} ends the process in place of a reply: an engine that is gone.

  fake_engine.py REPLIES.json REQUESTS.log --requests
"""
import json
import signal
import sys
import time

cancel = []
signal.signal(signal.SIGUSR1, lambda *a: cancel.append(1))

replies = json.load(open(sys.argv[1], encoding='utf-8'))
log = open(sys.argv[2], 'a', encoding='utf-8')
out, inp = sys.stdout.buffer, sys.stdin.buffer
out.write(b'ready\n')
out.flush()
i = 0
while True:
    head = inp.readline().decode().strip()
    if not head:
        break
    f = dict(w.split('=') for w in head.split()[1:])
    prompt = inp.read(int(f['bytes'])).decode('utf-8')
    log.write(json.dumps({'head': head, 'prompt': prompt}, ensure_ascii=False) + '\n')
    log.flush()
    r = replies[i % len(replies)]
    i += 1
    if 'exit' in r:
        sys.exit(r['exit'])
    if 'error' in r:
        out.write(f'error {r["error"]}\n'.encode())
        out.flush()
        continue
    kept = r.get('kept', 0)
    out.write(f'prompt tokens={len(prompt) // 4} n={f["n"]} kept={kept} ctx={kept} common={kept} how={"continue" if kept else "empty"}\n'.encode())
    out.flush()
    del cancel[:]
    t_end = time.time() + r.get('delay', 0)
    while time.time() < t_end and not cancel:
        time.sleep(0.05)
    if cancel:  # SIGUSR1 during the "generation": as the engine, the reply ends there
        out.write(f'end stop=cancel prompt_tokens={len(prompt) // 4} gen_tokens=0 prompt_s=0.1000 gen_s=0.5000\n'.encode())
        out.flush()
        continue
    data = r['text'].encode('utf-8')
    n = 0
    for k in range(0, len(data), 3):  # 3 bytes a "token": cuts UTF-8 characters, as real tokens can
        out.write(b'tok %d %s\n' % (k, data[k:k + 3].hex().encode()))
        n += 1
    out.write(f'end stop={r.get("stop", "eos")} prompt_tokens={len(prompt) // 4} kept={kept} saved={r.get("saved", 0)} gen_tokens={n} prompt_s=0.1000 gen_s=0.5000\n'.encode())
    out.flush()
