#!/usr/bin/env python3
"""README step 161: a stream from a running server (the model), against the same request without stream.

  results/server/stream_check.py [http://127.0.0.1:8080] > results/server/stream.txt

Three requests (reasoning + text; a tool call; no reasoning), each sent twice: as a stream,
timing its events, and plain. One seed gives one reply, so the message rebuilt from the
stream's events must be the plain message. Exit code 1 if not.
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8080'
TOOL = {'name': 'read', 'description': 'Read a file.', 'input_schema': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}
CASES = [
    ('reasoning and text', {'max_tokens': 2000, 'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'high'},
                            'messages': [{'role': 'user', 'content': 'A triangle has sides 13, 14 and 15. Find its area and its inradius. Explain briefly.'}]}),
    ('a tool call', {'max_tokens': 2000, 'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'high'}, 'tools': [TOOL],
                     'messages': [{'role': 'user', 'content': 'Say in one sentence what you are about to do, then read the file notes.txt.'}]}),
    ('no reasoning', {'max_tokens': 300, 'messages': [{'role': 'user', 'content': 'Write four lines about rain.'}]}),
]


def post(body):
    r = urllib.request.Request(BASE + '/v1/messages', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(r, timeout=3600)


failed = 0
for label, body in CASES:
    t0 = time.time()
    at, n, blocks, stop, usage, name = {}, {}, {}, None, {}, None
    with post(dict(body, stream=True)) as f:
        for line in f:
            line = line.decode('utf-8').rstrip('\n')
            if line.startswith('event: '):
                name = line[7:]
            elif line.startswith('data: '):
                d, t = json.loads(line[6:]), time.time() - t0
                key = name if name != 'content_block_delta' else d['delta']['type']
                at.setdefault(key, t)
                at['last ' + key] = t
                n[key] = n.get(key, 0) + 1
                if name == 'message_start':
                    usage = dict(d['message']['usage'])
                elif name == 'content_block_start':
                    blocks[d['index']] = dict(d['content_block'], _json='')
                elif name == 'content_block_delta':
                    b, dl = blocks[d['index']], d['delta']
                    if dl['type'] == 'input_json_delta':
                        b['_json'] += dl['partial_json']
                    else:
                        b[b['type']] += dl[b['type']]
                elif name == 'message_delta':
                    stop = d['delta']['stop_reason']
                    usage.update(d['usage'])
    content = []
    for k in sorted(blocks):
        b = blocks[k]
        if b['type'] == 'tool_use':
            b['input'] = json.loads(b.pop('_json'))
            b['id'] = 'toolu'  # (an id names its request: the two differ)
        b.pop('_json', None)
        content.append(b)
    t1 = time.time()
    with post(body) as f:
        m = json.loads(f.read())
    for b in m['content']:
        if b['type'] == 'tool_use':
            b['id'] = 'toolu'
    same = content == m['content'] and stop == m['stop_reason'] and usage['output_tokens'] == m['usage']['output_tokens']
    failed += not same
    print(f'{label}: {usage["output_tokens"]} tokens, {stop}, blocks {[b["type"] for b in content]}')
    print(f'  stream: message_start at {at["message_start"]:.2f} s'
          + ''.join(f'; {k}: {n[k]} from {at[k]:.2f} to {at["last " + k]:.2f} s' for k in ('thinking_delta', 'text_delta', 'input_json_delta') if k in n)
          + f'; message_stop at {at["message_stop"]:.2f} s; pings: {n.get("ping", 0)}')
    print(f'  plain: the whole message after {time.time() - t1:.2f} s ({m["usage"]["cache_read_input_tokens"]} of its '
          f'{m["usage"]["cache_read_input_tokens"] + m["usage"]["input_tokens"]} prompt tokens kept; the stream: {usage["cache_read_input_tokens"]})')
    print(f'  the message rebuilt from the stream {"is" if same else "IS NOT"} the plain message')
print('FAILED' if failed else 'all checks passed')
sys.exit(1 if failed else 0)
