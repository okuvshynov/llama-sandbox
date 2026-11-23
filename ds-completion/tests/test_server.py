#!/usr/bin/env python3
"""server/ds_server.py against a fake engine (tests/fake_engine.py): no model, a few seconds.

  python3 tests/test_server.py            exit code 1 if a check fails

What is checked: the prompt the engine receives for a request (the model's encoding.py
renders it: system, tools, effort, thinking), the message built from a canned reply (thinking,
text, a tool call), the same as a stream of events, replies that are cut or malformed, a tool
loop's second request (the reply sent back with a tool result), requests that are refused, and
GET /metrics and /health (step 165): the page while a request runs and another waits, its
totals against this test's own count of what it sent, and both with the engine gone; --log
and --dump (step 166): a JSON line and a dump file for every request, however it ended.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(os.path.dirname(HERE), 'server')
sys.path.insert(0, SERVER)
import ds_server  # noqa: E402

ENC = os.path.expanduser('~/llms/ds-flash/encoding/encoding.py')
enc = ds_server.load_encoding(ENC)

TOOL = {'name': 'make_guesses', 'description': 'Submit guesses.', 'strict': True,
        'input_schema': {'type': 'object', 'properties': {'guesses': {'type': 'array', 'items': {
            'type': 'object', 'properties': {'codemaker': {'type': 'string'}, 'guess': {'type': 'string'}}}}}, 'required': ['guesses']}}
ARGS = {'guesses': [{'codemaker': 'Red', 'guess': '1234'}, {'codemaker': 'Grün', 'guess': '5678'}]}
# a tool call as the model writes it: the reference's own rendering of an assistant message
CALL = enc.render_message(0, [{'role': 'assistant', 'content': '', 'tool_calls': [
    {'type': 'function', 'function': {'name': 'make_guesses', 'arguments': json.dumps(ARGS, ensure_ascii=False)}}]}], thinking_mode='chat')
assert CALL.startswith('\n\n<' + enc.dsml_token) and CALL.endswith(enc.eos_token), CALL
CALL = CALL[:-len(enc.eos_token)]  # two line ends and the tool-calls block: what follows the reasoning or the text

REPLIES = [
    {'text': 'Red is 1234 because… 思考' + enc.thinking_end_token + CALL},                    # 0 thinking + tool call
    {'text': 'Hello there.'},                                                                # 1 chat, text
    {'text': 'reasoning' + enc.thinking_end_token + 'An answer.' + CALL, 'delay': 2.5},       # 2 stream: thinking + text + tool call
    {'text': 'reasoning that never en', 'stop': 'length'},                                   # 3 cut inside the reasoning
    {'text': 'r' + enc.thinking_end_token + 'half an ans', 'stop': 'length'},                # 4 cut inside the answer
    {'text': 'r' + enc.thinking_end_token + CALL.replace('</', '<')},                        # 5 a broken tool call
    {'error': 'the prompt is 70000 tokens; --max-ctx is 65536'},                             # 6 the engine refuses
    {'text': 'after the error'},                                                             # 7
    {'text': 'never sent', 'delay': 60},                                                     # 8 the client leaves: cancelled
    {'text': 'never sent', 'delay': 60},                                                     # 9 a stream, the same
    {'text': 'after the cancels'},                                                           # 10
    {'text': 'r2' + enc.thinking_end_token + 'Done.', 'kept': 300, 'saved': 280},           # 12 after a tool result; 300 tokens were in the context
    {'text': 'ok'},                                                                          # 13 the same without thinking
    {'exit': 3},                                                                             # 14 the engine is gone
]
REPLIES.insert(11, REPLIES[0])                                                               # 11 a tool call again

failed = 0


def check(ok, what):
    global failed
    failed += not ok
    print(('ok    ' if ok else 'FAIL  ') + what, flush=True)


def main():
    tmp = tempfile.mkdtemp(prefix='ds-server-test-')
    replies, reqlog = os.path.join(tmp, 'replies.json'), os.path.join(tmp, 'requests.log')
    json.dump(REPLIES, open(replies, 'w', encoding='utf-8'), ensure_ascii=False)
    req_log, dump_dir = os.path.join(tmp, 'requests.jsonl'), os.path.join(tmp, 'dump')
    srv = subprocess.Popen([sys.executable, os.path.join(SERVER, 'ds_server.py'), '--port', '0', '--encoding', ENC, '--seed', '7', '--ping', '1', '--progress', '1',
                            '--log', req_log, '--dump', dump_dir, '--',
                            sys.executable, os.path.join(HERE, 'fake_engine.py'), replies, reqlog, '--requests'],
                           stdout=subprocess.PIPE, stderr=open(os.path.join(tmp, 'server.err'), 'w'))
    port = int(srv.stdout.readline().split()[1])
    base = f'http://127.0.0.1:{port}'

    codes = []  # of every POST /v1/messages sent through post(): what /metrics must add up to

    def post(body, path='/v1/messages?beta=true', raw=False):
        r = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(r, timeout=30) as f:
                data = f.read().decode('utf-8')
                codes.append(200)
                return 200, data if raw else json.loads(data)
        except urllib.error.HTTPError as e:
            codes.append(e.code)
            return e.code, json.loads(e.read().decode('utf-8'))

    def metrics():
        """GET /metrics -> ({'name' or 'name{label="value"}': number}, the content type, the text)"""
        with urllib.request.urlopen(base + '/metrics', timeout=5) as f:
            text, kind = f.read().decode('utf-8'), f.headers['Content-Type']
        return {l.rsplit(' ', 1)[0]: float(l.rsplit(' ', 1)[1]) for l in text.split('\n') if l and not l.startswith('#')}, kind, text

    def health():
        try:
            with urllib.request.urlopen(base + '/health', timeout=5) as f:
                return 200, json.loads(f.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    # (step 165) before any request: the page is there, every series at 0, the engine up
    m0, kind, text = metrics()
    names = [l.split()[2] for l in text.split('\n') if l.startswith('# TYPE')]
    check(kind.startswith('text/plain; version=0.0.4') and len(names) == 17 and text.count('# HELP') == 17 and all(n.startswith('ds_') for n in names)
          and all(l.split()[3] in ('counter', 'gauge') for l in text.split('\n') if l.startswith('# TYPE')),
          f'GET /metrics: the Prometheus text format, {len(names)} metrics with HELP and TYPE (counters and gauges), {len(m0)} series')
    check(all(v == 0 for k, v in m0.items() if k.endswith('_total') or '_total{' in k) and m0['ds_engine_up'] == 1 and m0['ds_engine_busy'] == 0
          and m0['ds_requests_waiting'] == 0 and m0['ds_start_time_seconds'] > 1.7e9 and len([k for k in m0 if k.startswith('ds_requests_total{')]) == 4,
          'before any request: every counter is there at 0, ds_engine_up 1, busy 0, waiting 0')
    check(health() == (200, {'status': 'ok'}), f'GET /health with the engine up: {health()}')

    def seen():
        return [json.loads(l) for l in open(reqlog, encoding='utf-8')]

    # 0: the benchmark's request, without stream
    bench = {'model': 'x', 'max_tokens': 65536, 'messages': [{'role': 'user', 'content': 'Play the game.'}], 'tools': [TOOL],
             'tool_choice': {'type': 'auto', 'disable_parallel_tool_use': True}, 'thinking': {'type': 'adaptive'},
             'output_config': {'effort': 'medium'}, 'fallbacks': 'default'}
    code, m = post(bench)
    p = seen()[-1]
    want = enc.encode_messages([{'role': 'system', 'content': '', 'tools': [{'type': 'function', 'function': {
        'name': TOOL['name'], 'parameters': TOOL['input_schema'], 'description': TOOL['description']}}]},
        {'role': 'user', 'content': 'Play the game.'}], thinking_mode='thinking', reasoning_effort=62)
    check(p['prompt'] == want and want.startswith(enc.bos_token) and want.endswith('<｜Assistant｜><think>') and 'Reasoning Effort: 62 ' in want
          and '"make_guesses"' in want, 'the prompt: encoding.py\'s rendering of system + tools + user, effort medium = 62, ends with <think>')
    check(p['head'] == f'gen n=65536 temp=1.0 top_p=0.95 seed=7 bytes={len(want.encode())}', f'the engine\'s header: {p["head"]}')
    kinds = [b['type'] for b in m.get('content', [])]
    check(code == 200 and kinds == ['thinking', 'tool_use'] and m['stop_reason'] == 'tool_use' and m['content'][1]['input'] == ARGS
          and m['content'][1]['name'] == 'make_guesses' and m['content'][0]['thinking'] == 'Red is 1234 because… 思考',
          f'thinking + a tool call: blocks {kinds}, stop_reason {m.get("stop_reason")}, the input as sent (non-ASCII included)')
    check(m['type'] == 'message' and m['role'] == 'assistant' and m['usage']['input_tokens'] == len(want) // 4 and m['usage']['output_tokens'] > 0
          and m['model'] == 'deepseek-v4.1-flash-1017', f'the message\'s fields: usage {m["usage"]}')

    # 1: no thinking, a system prompt, earlier turns, the sampler from the request
    code, m = post({'max_tokens': 100, 'system': [{'type': 'text', 'text': 'Be brief.'}], 'temperature': 0, 'seed': 3,
                    'messages': [{'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Hello.'}]},
                                 {'role': 'user', 'content': 'Again'}]})
    p = seen()[-1]
    want = enc.encode_messages([{'role': 'system', 'content': 'Be brief.'}, {'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello.'},
                                {'role': 'user', 'content': 'Again'}], thinking_mode='chat')
    check(p['prompt'] == want and want.endswith('<｜Assistant｜></think>') and p['head'].startswith('gen n=100 temp=0.0 top_p=1.0 seed=3 '),
          f'no thinking, a system prompt, three turns: the prompt is encoding.py\'s; header {p["head"]}')
    check(code == 200 and m['content'] == [{'type': 'text', 'text': 'Hello there.'}] and m['stop_reason'] == 'end_turn', f'a text reply: stop_reason {m.get("stop_reason")}')

    # 2: the stream (step 161: as the reply is generated). The fake engine takes the request, waits 2.5 s,
    # then writes the reply 3 bytes a token.
    def stream_of(body):
        """-> (the events, the message rebuilt from them: content blocks, stop_reason, usage)"""
        code, raw = post(dict(body, stream=True), raw=True)
        events = []
        for part in raw.strip().split('\n\n'):
            lines = part.split('\n')
            events.append((lines[0][len('event: '):], json.loads(lines[1][len('data: '):])))
        blocks, stop, usage = {}, None, {}
        for n, d in events:
            if n == 'message_start':
                usage = dict(d['message']['usage'])
            elif n == 'content_block_start':
                blocks[d['index']] = dict(d['content_block'], _json='')
            elif n == 'content_block_delta':
                b, dl = blocks[d['index']], d['delta']
                if dl['type'] == 'input_json_delta':
                    b['_json'] += dl['partial_json']
                else:
                    b[b['type']] += dl[b['type']]
            elif n == 'message_delta':
                stop = d['delta']['stop_reason']
                usage.update(d['usage'])
        content = []
        for k in sorted(blocks):
            b = blocks[k]
            if b['type'] == 'tool_use':
                b['input'] = json.loads(b['_json'])
            del b['_json']
            content.append(b)
        return events, content, stop, usage

    events, content, stop, usage = stream_of(bench)
    names = [n for n, _ in events]
    rest = [n for n in names if n != 'ping']
    check(names[0] == 'message_start' and names[1] == 'ping' and names.count('ping') >= 2 and names.index('content_block_start') > names.index('ping'),
          f'a stream starts with message_start when the engine takes the request, then pings while it works: {names.count("ping")} pings')
    shape = [n for k, n in enumerate(rest) if n != 'content_block_delta' or rest[k - 1] != 'content_block_delta']
    check(shape == ['message_start'] + ['content_block_start', 'content_block_delta', 'content_block_stop'] * 3 + ['message_delta', 'message_stop'],
          f'then three blocks, each started, its deltas, stopped: {" ".join(shape[1:4])} x3, message_delta, message_stop')
    n_delta = [sum(1 for n, d in events if n == 'content_block_delta' and d['index'] == k) for k in range(3)]
    check(n_delta[0] >= 2 and n_delta[1] >= 2 and n_delta[2] == 1, f'the thinking and the text come in pieces, the tool call whole: {n_delta} deltas')
    check(all(n == d['type'] for n, d in events) and events[0][1]['message']['content'] == [] and events[0][1]['message']['usage']['output_tokens'] == 0
          and [b['type'] for b in content] == ['thinking', 'text', 'tool_use'] and content[0]['thinking'] == 'reasoning' and content[0]['signature'] == ds_server.SIGNATURE
          and content[1]['text'] == 'An answer.' and content[2]['input'] == ARGS and content[2]['name'] == 'make_guesses' and stop == 'tool_use' and usage['output_tokens'] > 0,
          'the events carry the thinking, the text, the tool\'s input as JSON, stop_reason tool_use')

    # 3-5: cut and malformed replies
    code, m = post(bench)
    check(code == 200 and m['stop_reason'] == 'max_tokens' and [b['type'] for b in m['content']] == ['thinking'], f'cut inside the reasoning: stop_reason {m.get("stop_reason")}, blocks {[b["type"] for b in m.get("content", [])]}')
    code, m = post(bench)
    check(code == 200 and m['stop_reason'] == 'max_tokens' and [b['type'] for b in m['content']] == ['thinking', 'text'] and m['content'][1]['text'] == 'half an ans',
          f'cut inside the answer: stop_reason {m.get("stop_reason")}, blocks {[b["type"] for b in m.get("content", [])]}')
    code, m = post(bench)
    check(code == 200 and m['stop_reason'] == 'end_turn' and [b['type'] for b in m['content']] == ['thinking', 'text'] and 'DSML' in m['content'][1]['text'],
          f'a broken tool call: no tool_use block, the raw text returned: stop_reason {m.get("stop_reason")}')

    # (step 161) the same three as streams: the canned replies come round again below; here, without a server,
    # every canned reply through Live in pieces of 1, 2, 3, 7 bytes and whole: the stream's thinking and text
    # are the message's, the tool calls come after them
    n_live = 0
    tricky = [{'text': t} for t in ('text that ends with line ends\n\n', 'a\n\n<b> is not a tool call\n\n<｜DSML and more', 'r</th' + enc.thinking_end_token + '\n\n', '',
                                    'r' + enc.thinking_end_token + 'x\n\n<' + enc.dsml_token + ' calls> cut here')]
    tricky[-1]['stop'] = 'length'
    for r in REPLIES + tricky:
        if 'text' not in r:
            continue
        for mode in ('thinking', 'chat'):
            data = r['text'].encode('utf-8')
            blocks, stop_reason, note = ds_server.parse_reply(enc, r['text'], mode, r.get('stop', 'eos') == 'eos', lambda p: p + '_x')
            for size in (1, 2, 3, 7, max(1, len(data))):
                live = ds_server.Live(enc, mode)
                ev = []
                for k in range(0, len(data), size):
                    ev += live.feed(data[k:k + size])
                ev += live.finish(blocks)
                got = {}
                for name, d in ev:
                    if name == 'content_block_start':
                        got[d['index']] = dict(d['content_block'])
                    elif name == 'content_block_delta' and d['delta']['type'] != 'input_json_delta':
                        got[d['index']][got[d['index']]['type']] += d['delta'][got[d['index']]['type']]
                    elif name == 'content_block_delta':
                        got[d['index']]['input'] = json.loads(d['delta']['partial_json'])
                n_live += 1
                if [got[k] for k in sorted(got)] != blocks or live.differs:
                    check(False, f'Live, {mode}, pieces of {size}: {[got[k] for k in sorted(got)]} is not {blocks}')
    check(n_live >= 150, f'Live: {n_live} cases (every canned reply and five odd ones, both modes, pieces of 1, 2, 3, 7 bytes and whole): the stream\'s blocks are the message\'s')

    # 6-7: an engine error, and the request after it
    code, m = post(bench)
    check(code == 400 and m['error']['type'] == 'invalid_request_error' and '70000 tokens' in m['error']['message'], f'the engine\'s error line: {code} {m["error"]["message"]}')
    code, m = post({'max_tokens': 5, 'messages': [{'role': 'user', 'content': 'x'}]})
    check(code == 200 and m['content'][0]['text'] == 'after the error', 'the next request is answered')

    # 8-10: a client that leaves while the engine works (60 s of it): the request is cancelled, the next one answered
    import socket
    import time
    # (step 165) both are sent before the first leaves: /metrics answers meanwhile, with one request running and one waiting
    conns = []
    for stream in (False, True):
        body = json.dumps(dict(bench, stream=stream)).encode()
        c = socket.create_connection(('127.0.0.1', port))
        c.sendall(b'POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n' % len(body) + body)
        conns.append(c)
        time.sleep(0.75)
    t = time.time()
    mm = metrics()[0]
    check(mm['ds_engine_busy'] == 1 and mm['ds_requests_waiting'] == 1 and time.time() - t < 1,
          f'GET /metrics while one request runs and one waits: busy {mm["ds_engine_busy"]:.0f}, waiting {mm["ds_requests_waiting"]:.0f}, answered in {time.time() - t:.2f} s')
    conns[0].close()
    time.sleep(3)  # (the second one has the engine by now: it is cancelled there, as before, not in the queue)
    conns[1].close()
    t = time.time()
    code, m = post({'max_tokens': 5, 'messages': [{'role': 'user', 'content': 'x'}]})
    dt = time.time() - t
    check(code == 200 and m['content'][0]['text'] == 'after the cancels' and dt < 10,
          f'two clients left 60 s replies (one plain, one a stream): the next request is answered {dt:.1f} s later')

    # 11-13 (step 157): the next request of a tool loop, as pi sends it: the reply of request 0 back as
    # thinking + tool_use blocks, the tool's result in a user message
    code, m0 = post(bench)
    p0 = seen()[-1]['prompt']
    check(m0['content'][0]['signature'] == ds_server.SIGNATURE != '', f'a thinking block has a signature that is not empty: {m0["content"][0]["signature"]!r}')
    call = m0['content'][1]
    result = [{'type': 'tool_result', 'tool_use_id': call['id'], 'content': [{'type': 'text', 'text': 'Red: 2 right.'}, {'type': 'text', 'text': ' Grün: 0 right.'}],
               'is_error': False}, {'type': 'text', 'text': 'Go on.', 'cache_control': {'type': 'ephemeral'}}]
    loop = dict(bench, messages=bench['messages'] + [{'role': 'assistant', 'content': m0['content']}, {'role': 'user', 'content': result}])
    code, m = post(loop)
    p = seen()[-1]['prompt']
    tail = p[len(p0) + len(REPLIES[0]['text']):]
    check(code == 200 and p.startswith(p0 + REPLIES[0]['text']),
          f'a tool loop\'s second prompt starts with the first prompt and the reply as the model wrote it (reasoning, tool call): {code}')
    check(tail == enc.eos_token + '<｜User｜><tool_result>Red: 2 right. Grün: 0 right.</tool_result>\n\nGo on.<｜Assistant｜><think>', f'and goes on with the result and the text: {tail!r}')
    check([b['type'] for b in m.get('content', [])] == ['thinking', 'text'] and m['stop_reason'] == 'end_turn', 'it is answered')
    mm = metrics()[0]
    check(mm['ds_saved_point_tokens'] == 280 and mm['ds_prompt_tokens_total{kind="kept"}'] == 300 and mm['ds_context_total{how="continue"}'] == 1,
          '/metrics after a request that kept 300 tokens and left a saved point at 280: kept %.0f, the point %.0f, how=continue %.0f'
          % (mm['ds_prompt_tokens_total{kind="kept"}'], mm['ds_saved_point_tokens'], mm['ds_context_total{how="continue"}']))
    check(m['usage']['cache_read_input_tokens'] == 300 and m['usage']['input_tokens'] == len(p) // 4 - 300 and m0['usage']['cache_read_input_tokens'] == 0,
          f'the tokens the engine kept (step 158) are the usage\'s cache_read_input_tokens, the others its input_tokens: {m["usage"]}')
    code, m = post(dict(loop, thinking={'type': 'disabled'}, messages=loop['messages'][:2] + [{'role': 'user', 'content': [result[0]]}]))
    p = seen()[-1]['prompt']
    check(code == 200 and p.endswith('<｜User｜><tool_result>Red: 2 right. Grün: 0 right.</tool_result><｜Assistant｜></think>') and 'Red is 1234 because' not in p
          and CALL in p, 'without thinking, a result alone: the call is in the prompt, the reasoning is not')

    # refused before the engine sees them
    n_seen = len(seen())
    for what, body in [
            ('tool_choice any', dict(bench, tool_choice={'type': 'any'})),
            ('no max_tokens', {k: v for k, v in bench.items() if k != 'max_tokens'}),
            ('an image block', dict(bench, messages=[{'role': 'user', 'content': [{'type': 'image', 'source': {}}]}])),
            ('an image in a tool result', dict(bench, messages=[{'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'a', 'content': [{'type': 'image', 'source': {}}]}]}])),
            ('a tool_use block from the user', dict(bench, messages=[{'role': 'user', 'content': [{'type': 'tool_use', 'id': 'a', 'name': 'b', 'input': {}}]}])),
            ('redacted thinking', dict(bench, messages=bench['messages'] + [{'role': 'assistant', 'content': [{'type': 'redacted_thinking', 'data': 'x'}]}, {'role': 'user', 'content': 'x'}])),
            ('the last message from the assistant', dict(bench, messages=bench['messages'] + [{'role': 'assistant', 'content': 'I'}])),
            ('effort "extreme"', dict(bench, output_config={'effort': 'extreme'})),
            ('no messages', dict(bench, messages=[]))]:
        code, m = post(body)
        check(code == 400 and m.get('type') == 'error', f'refused, {what}: {code} {m.get("error", {}).get("message")}')
    check(len(seen()) == n_seen, 'none of the refused requests reached the engine')
    r = urllib.request.Request(base + '/v1/messages', data=b'{not json', headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(r, timeout=30)
        code = 200
    except urllib.error.HTTPError as e:
        code = e.code
    check(code == 400, f'a body that is not JSON: {code}')
    codes.append(400)
    with urllib.request.urlopen(base + '/v1/models', timeout=30) as f:
        models = json.loads(f.read())
    check(models['data'][0]['id'] == 'deepseek-v4.1-flash-1017', 'GET /v1/models names the model')
    try:
        urllib.request.urlopen(base + '/v1/other', timeout=30)
        code = 200
    except urllib.error.HTTPError as e:
        code = e.code
    check(code == 404, f'another path: {code}')

    # (step 165) the page's totals against this test's own count: every POST through post() and the two clients that left
    mm = metrics()[0]
    n_ok, n_bad = codes.count(200), codes.count(400)
    outcome = {k: mm['ds_requests_total{outcome="%s"}' % k] for k in ('ok', 'cancelled', 'bad_request', 'engine_error')}
    check(outcome == {'ok': n_ok, 'cancelled': 2, 'bad_request': n_bad, 'engine_error': 0} and len(codes) == n_ok + n_bad,
          f'/metrics, requests by outcome: {({k: int(v) for k, v in outcome.items()})} (this test sent {n_ok} answered, {n_bad} refused, 2 left)')
    stops = {k: mm['ds_stop_reason_total{reason="%s"}' % k] for k in ('end_turn', 'tool_use', 'max_tokens')}
    check(sum(stops.values()) == n_ok and stops['max_tokens'] == 2 and stops['tool_use'] == 3 and mm['ds_reply_parse_failures_total'] == 1 and mm['ds_stream_mismatch_total'] == 0,
          f'stop reasons {({k: int(v) for k, v in stops.items()})} add up to the answered ones; 1 reply did not parse, no stream differed')
    how = {k: mm['ds_context_total{how="%s"}' % k] for k in ('continue', 'saved', 'empty')}
    check(how == {'continue': 1, 'saved': 0, 'empty': n_ok + 2 - 1},
          f'the engine\'s choice of context {({k: int(v) for k, v in how.items()})}: every request it took, the two cancelled ones too')
    check(abs(mm['ds_generation_seconds_total'] - 0.5 * n_ok) < 1e-6 and abs(mm['ds_prompt_seconds_total'] - 0.1 * n_ok) < 1e-6 and mm['ds_generated_tokens_total'] > 100
          and mm['ds_prompt_tokens_total{kind="run"}'] > 1000 and mm['ds_first_token_seconds_total'] > 2.5 and mm['ds_queue_wait_seconds_total'] >= 0
          and mm['ds_saved_point_tokens'] == 0,
          f'seconds: generation {mm["ds_generation_seconds_total"]:.1f} and prompt {mm["ds_prompt_seconds_total"]:.1f} (the fake engine says 0.5 and 0.1 a request), '
          f'to the first token {mm["ds_first_token_seconds_total"]:.1f} (one reply took 2.5 s to start); tokens: {mm["ds_generated_tokens_total"]:.0f} generated, '
          + '%.0f of prompts run; the saved point is gone (0)' % mm['ds_prompt_tokens_total{kind="run"}'])
    # the engine goes away (the fake one exits at this request): a 500, then /health and /metrics say so
    code, m = post({'max_tokens': 5, 'messages': [{'role': 'user', 'content': 'x'}]})
    time.sleep(0.3)
    mm = metrics()[0]
    hc, hb = health()
    check(code == 500 and m['error']['type'] == 'api_error' and mm['ds_engine_up'] == 0 and mm['ds_requests_total{outcome="engine_error"}'] == 1
          and mm['ds_engine_busy'] == 0 and mm['ds_requests_waiting'] == 0 and hc == 503 and hb.get('exit_code') == 3,
          f'the engine gone: the request is a {code}, /metrics has ds_engine_up 0 and one engine_error, /health is {hc} {hb}')

    srv.terminate()
    srv.wait()
    # (step 166) --log: one JSON line per request to /v1/messages, whichever way it ended; --dump: a file for each
    recs = [json.loads(l) for l in open(req_log, encoding='utf-8')]
    by = {o: [r for r in recs if r['outcome'] == o] for o in ('ok', 'cancelled', 'bad_request', 'engine_error')}
    check(len(recs) == len(codes) + 2 and [r['request'] for r in recs] == sorted(r['request'] for r in recs) and {r['request'] for r in recs} == set(range(1, len(recs) + 1))
          and all(tuple(r) == ds_server.RECORD for r in recs),
          f'--log: {len(recs)} lines for {len(codes)} requests sent and 2 clients that left, numbered 1..{len(recs)}, every line with the same {len(ds_server.RECORD)} keys')
    check({o: len(v) for o, v in by.items()} == {'ok': n_ok, 'cancelled': 2, 'bad_request': n_bad, 'engine_error': 1},
          f'--log, by outcome: {({o: len(v) for o, v in by.items()})}: what /metrics counted, and the request the engine died on')
    check(sum(r['generated_tokens'] for r in by['ok']) == mm['ds_generated_tokens_total'] and sum(r['prompt_tokens'] - r['kept'] for r in by['ok']) == mm['ds_prompt_tokens_total{kind="run"}']
          and all(r['stop_reason'] and r['blocks'] is not None and r['prompt_s'] == 0.1 and r['generation_s'] == 0.5 and r['total_s'] >= r['first_token_s'] >= r['wait_s'] >= 0
                  and r['how'] in ('continue', 'empty') and r['mode'] in ('chat', 'thinking') and r['error'] is None for r in by['ok']),
          'the lines of the answered requests: tokens add up to /metrics\' totals; each has its stop reason, blocks, seconds, mode and how')
    k300 = [r for r in recs if r['kept'] == 300]
    slow = max(by['ok'], key=lambda r: r['first_token_s'])
    check(len(k300) == 1 and k300[0]['how'] == 'continue' and k300[0]['ctx'] == 300 and k300[0]['saved'] == 280 and slow['stream'] is True and slow['first_token_s'] >= 2.5
          and slow['stream_differs'] is False and slow['blocks'] == ['thinking', 'text', 'tool_use'] and slow['tools'] == 1 and slow['max_tokens'] == 65536,
          f'the request that kept 300 tokens: how={k300[0]["how"]} ctx={k300[0]["ctx"]} saved={k300[0]["saved"]}; the slow stream: first token after {slow["first_token_s"]} s, blocks {slow["blocks"]}')
    notes = [r['note'] for r in by['ok'] if r['note']]
    check(len(notes) == 1 and 'did not parse' in notes[0], f'one line has a note: {notes[0] if notes else None!r:.60}')
    check(all(r['client_left'] and r['generated_tokens'] == 0 and r['stop_reason'] is None and r['prompt_tokens'] > 0 for r in by['cancelled'])
          and [r['stream'] for r in by['cancelled']] == [False, True],
          f'the two cancelled ones: client_left, no stop reason, the prompt\'s tokens ({by["cancelled"][0]["prompt_tokens"]}); one plain, one a stream')
    refused = [r for r in by['bad_request'] if r['mode'] is None]  # by the server, before the engine saw them
    check(len(refused) == n_bad - 1 and all(r['error'] and r['how'] is None and r['generated_tokens'] is None for r in by['bad_request'])
          and sum('70000 tokens' in r['error'] for r in by['bad_request']) == 1 and sum('not JSON' in r['error'] for r in by['bad_request']) == 1
          and 'gone' in by['engine_error'][0]['error'],
          f'the refused ones have the error\'s text ({len(refused)} by the server, 1 by the engine); the last line: {by["engine_error"][0]["error"]!r}')
    check(all(re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}[+-]\d\d:\d\d', r['time']) for r in recs) and [r['time'] for r in recs] == sorted(r['time'] for r in recs),
          f'every line has its arrival\'s time with the zone, in order: {recs[0]["time"]}')
    dumps = {f[:-5]: json.load(open(os.path.join(dump_dir, f), encoding='utf-8')) for f in os.listdir(dump_dir)}
    d_can, d_bad, d_ok = dumps[by['cancelled'][0]['id']], dumps[refused[0]['id']], dumps[k300[0]['id']]
    not_json = dumps[[r for r in recs if r['error'] and 'not JSON' in r['error']][0]['id']]
    check(set(dumps) == {r['id'] for r in recs} and all(d['outcome'] == r['outcome'] for r in recs for d in [dumps[r['id']]])
          and d_can['prompt'] and d_can['message'] is None and d_can['reply'] == '' and d_bad['request'] and d_bad['prompt'] is None and d_bad['error']
          and d_ok['message']['usage']['cache_read_input_tokens'] == 300 and d_ok['reply'] and not_json['request'] is None and not_json['body'] == '{not json',
          f'--dump: {len(dumps)} files, one per line, each with its outcome: a cancelled one has its prompt and no message, a refused one its request and error, '
          'the body that is not JSON as text')
    stamped = [l for l in open(os.path.join(tmp, 'server.err'), encoding='utf-8') if l.startswith('# server')]
    check(len(stamped) > 20 and all(re.match(r'# server \d{4}-\d\d-\d\d \d\d:\d\d:\d\d: ', l) for l in stamped),
          f'the server\'s {len(stamped)} log lines start with the date and time: {stamped[-1].strip()[:70]!r}')
    cancelled = [l for l in open(os.path.join(tmp, 'server.err'), encoding='utf-8') if ': cancelled after' in l]
    check(len(cancelled) == 2, f'the log has the two cancelled requests: {[l.strip() for l in cancelled]}')
    said = [l for l in open(os.path.join(tmp, 'server.err'), encoding='utf-8') if 'tokens so far' in l]
    check(len(said) >= 1, f'the log has progress lines for the slow request: {len(said)}, the first: {said[0].strip() if said else None}')
    print('FAILED' if failed else 'all checks passed')
    if failed:
        print(f'(the server\'s log: {tmp}/server.err)')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
