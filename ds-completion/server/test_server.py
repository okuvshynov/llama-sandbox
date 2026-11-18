#!/usr/bin/env python3
"""server/ds_server.py against a fake engine (server/fake_engine.py): no model, a few seconds.

  server/test_server.py            exit code 1 if a check fails

What is checked: the prompt the engine receives for a request (the model's encoding.py
renders it: system, tools, effort, thinking), the message built from a canned reply (thinking,
text, a tool call), the same as a stream of events, replies that are cut or malformed, a tool
loop's second request (the reply sent back with a tool result), and requests that are refused.
"""
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
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
    {'text': 'r2' + enc.thinking_end_token + 'Done.', 'kept': 300},                         # 12 after a tool result; 300 tokens were in the context
    {'text': 'ok'},                                                                          # 13 the same without thinking
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
    srv = subprocess.Popen([sys.executable, os.path.join(HERE, 'ds_server.py'), '--port', '0', '--encoding', ENC, '--seed', '7', '--ping', '1', '--progress', '1', '--',
                            sys.executable, os.path.join(HERE, 'fake_engine.py'), replies, reqlog, '--requests'],
                           stdout=subprocess.PIPE, stderr=open(os.path.join(tmp, 'server.err'), 'w'))
    port = int(srv.stdout.readline().split()[1])
    base = f'http://127.0.0.1:{port}'

    def post(body, path='/v1/messages?beta=true', raw=False):
        r = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(r, timeout=30) as f:
                data = f.read().decode('utf-8')
                return 200, data if raw else json.loads(data)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode('utf-8'))

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
    for stream in (False, True):
        body = json.dumps(dict(bench, stream=stream)).encode()
        c = socket.create_connection(('127.0.0.1', port))
        c.sendall(b'POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n' % len(body) + body)
        time.sleep(1.5)
        c.close()
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
    with urllib.request.urlopen(base + '/v1/models', timeout=30) as f:
        models = json.loads(f.read())
    check(models['data'][0]['id'] == 'deepseek-v4.1-flash-1017', 'GET /v1/models names the model')
    try:
        urllib.request.urlopen(base + '/v1/other', timeout=30)
        code = 200
    except urllib.error.HTTPError as e:
        code = e.code
    check(code == 404, f'another path: {code}')

    srv.terminate()
    srv.wait()
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
