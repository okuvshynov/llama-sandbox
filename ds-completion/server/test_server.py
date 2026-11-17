#!/usr/bin/env python3
"""server/ds_server.py against a fake engine (server/fake_engine.py): no model, a few seconds.

  server/test_server.py            exit code 1 if a check fails

What is checked: the prompt the engine receives for a request (the model's encoding.py
renders it: system, tools, effort, thinking), the message built from a canned reply (thinking,
text, a tool call), the same as a stream of events, replies that are cut or malformed, and
requests that are refused.
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
]

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

    # 2: the stream (the reply takes 2.5 s: pings first)
    code, raw = post(dict(bench, stream=True), raw=True)
    events = []
    for part in raw.strip().split('\n\n'):
        lines = part.split('\n')
        events.append((lines[0][len('event: '):], json.loads(lines[1][len('data: '):])))
    names = [n for n, _ in events]
    rest = [n for n in names if n != 'ping']
    check(names[0] == 'ping' and names.count('ping') >= 2, f'a stream starts with pings while the engine works: {names.count("ping")} pings')
    check(rest == ['message_start'] + ['content_block_start', 'content_block_delta', 'content_block_stop'] * 3 + ['message_delta', 'message_stop'],
          f'then the events of three blocks: {" ".join(rest)}')
    by = {i: [d for n, d in events if d.get('index') == i] for i in range(3)}
    tool_json = ''.join(d['delta']['partial_json'] for d in by[2] if d['type'] == 'content_block_delta')
    check(all(n == d['type'] for n, d in events) and events[names.index('message_start')][1]['message']['content'] == []
          and by[0][0]['content_block']['type'] == 'thinking' and by[0][1]['delta'] == {'type': 'thinking_delta', 'thinking': 'reasoning'}
          and by[1][1]['delta'] == {'type': 'text_delta', 'text': 'An answer.'}
          and by[2][0]['content_block']['type'] == 'tool_use' and by[2][0]['content_block']['input'] == {} and json.loads(tool_json) == ARGS
          and events[-2][1]['delta']['stop_reason'] == 'tool_use' and events[-2][1]['usage']['output_tokens'] > 0,
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

    # refused before the engine sees them
    n_seen = len(seen())
    for what, body in [
            ('tool_choice any', dict(bench, tool_choice={'type': 'any'})),
            ('no max_tokens', {k: v for k, v in bench.items() if k != 'max_tokens'}),
            ('an image block', dict(bench, messages=[{'role': 'user', 'content': [{'type': 'image', 'source': {}}]}])),
            ('a tool_result block', dict(bench, messages=[{'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'a', 'content': 'b'}]}])),
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
