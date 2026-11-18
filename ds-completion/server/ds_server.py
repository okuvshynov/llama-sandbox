#!/usr/bin/env python3
"""An HTTP front for ds-completion: the Anthropic Messages API, as far as a single-turn
benchmark with one tool needs it (README step 153).

  server/ds_server.py [--port 8080] [--host 127.0.0.1] [--encoding ~/llms/ds-flash/encoding/encoding.py]
                      [--temp 1 --top-p 0.95 --seed 0] [--model-name NAME] [--dump DIR] [--ping 10] -- ENGINE COMMAND ...

  server/ds_server.py -- bin-hip/ds-completion -m $M --rt $F8 --placement placements/study30.txt \\
      --dspark $D --spec --canon --cache-loads 800 --cache-period 8 --cache-wait --requests

The engine is started as a child in its --requests mode (docs/requests.md) and answers one
request at a time; requests that arrive meanwhile wait. This program renders a request into
the model's prompt and parses the reply with the model's own encoding.py (system prompt,
tools, reasoning effort, the tool-call tags), so nothing of the format is written twice.
Only the standard library is used.

POST /v1/messages   (docs/server.md says what is read of a request, and what is refused)
GET  /v1/models
GET  /health
"""
import argparse
import codecs
import importlib.util
import itertools
import json
import os
import queue
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# output_config.effort -> the model's reasoning effort, 1-100. low / high / max are encoding.py's
# own (REASONING_EFFORT_MAPPINGS); medium and xhigh are placed between them.
EFFORT = {'low': 50, 'medium': 62, 'high': 75, 'xhigh': 88, 'max': 100}

# A thinking block's signature. Nothing is signed; it is not empty because a client may send a
# block without a signature back as text (pi does), and the reasoning must come back as reasoning.
SIGNATURE = 'ds-completion'


class BadRequest(Exception):
    pass


def log(*a):
    print('# server:', *a, file=sys.stderr, flush=True)


def load_encoding(path):
    spec = importlib.util.spec_from_file_location('ds_encoding', os.path.expanduser(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Engine:
    """The child process and its protocol (docs/requests.md). One request at a time."""

    def __init__(self, cmd):
        self.lock = threading.Lock()
        self.state = threading.Lock()  # guards `active`
        self.active = None             # the token of the request the engine is working on
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        t0 = time.time()
        line = self.proc.stdout.readline().decode().strip()
        if line != 'ready':
            raise RuntimeError(f'the engine did not say "ready" (got {line!r}, exit code {self.proc.poll()})')
        log(f'the engine is ready after {time.time() - t0:.0f} s')

    def cancel(self, token):
        """Stop the request `token` if the engine is working on it (SIGUSR1: docs/requests.md);
        if it is still waiting for the engine, it is cancelled when its turn comes."""
        with self.state:
            token['cancel'] = True
            if self.active is token:
                self.proc.send_signal(signal.SIGUSR1)

    def generate(self, prompt, n, temp, top_p, seed, on_start=None, on_token=None, token=None):
        """-> (the reply's text as bytes, the end line's fields). Raises BadRequest on an
        engine "error" line, RuntimeError if the engine is gone. on_start(prompt_tokens, kept) is
        called when the engine has accepted the request, on_token(its bytes) after every token.
        token: a dict naming this request for cancel(); a cancelled reply ends with stop=cancel."""
        body = prompt.encode('utf-8')
        head = f'gen n={n} temp={temp!r} top_p={top_p!r} seed={seed} bytes={len(body)}\n'
        with self.lock:
            with self.state:
                if token is not None and token.get('cancel'):  # cancelled while it waited
                    return b'', {'stop': 'cancel', 'prompt_tokens': '0', 'gen_tokens': '0', 'prompt_s': '0', 'gen_s': '0'}
                try:
                    self.proc.stdin.write(head.encode() + body)
                    self.proc.stdin.flush()
                except OSError as e:
                    raise RuntimeError(f'the engine is gone ({e})')
                self.active = token
            try:
                return self.read_reply(on_start, on_token)
            finally:
                with self.state:  # (after the reply's end line: a signal can no longer be for this request)
                    self.active = None

    def read_reply(self, on_start, on_token):
        out = b''
        while True:
            w = self.proc.stdout.readline().decode().split()
            if not w:
                raise RuntimeError(f'the engine is gone (exit code {self.proc.poll()})')
            if w[0] == 'error':
                raise BadRequest(' '.join(w[1:]))
            if w[0] == 'prompt':
                if on_start:
                    f = dict(f.split('=') for f in w[1:])
                    on_start(int(f['tokens']), int(f.get('kept', 0)))
            elif w[0] == 'tok':
                piece = bytes.fromhex(w[2]) if len(w) > 2 else b''
                out += piece
                if on_token:
                    on_token(piece)
            elif w[0] == 'end':
                return out, dict(f.split('=') for f in w[1:])

    def close(self):
        try:
            with self.state:
                if self.active is not None:
                    self.proc.send_signal(signal.SIGUSR1)
            self.proc.stdin.close()
            self.proc.wait(timeout=120)
        except Exception:
            self.proc.kill()


def text_of(content, where):
    """A message's content as text: a string, or a list of text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise BadRequest(f'{where}: content must be a string or a list of blocks')
    parts = []
    for b in content:
        if not isinstance(b, dict) or b.get('type') != 'text' or not isinstance(b.get('text'), str):
            kind = b.get('type') if isinstance(b, dict) else type(b).__name__
            raise BadRequest(f'{where}: only text blocks are supported (got {kind!r})')
        parts.append(b['text'])
    return ''.join(parts)


def blocks_of(content, where):
    """A message's content as a list of blocks (a string is one text block)."""
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}]
    if not isinstance(content, list) or not all(isinstance(b, dict) for b in content):
        raise BadRequest(f'{where}: content must be a string or a list of blocks')
    return content


def user_message(content, where):
    """A user message for encoding.py: text, and the results of tool calls (step 157), which the
    model reads inside a user message, in the order given."""
    blocks = blocks_of(content, where)
    if all(b.get('type') == 'text' for b in blocks):
        return {'role': 'user', 'content': text_of(blocks, where)}
    out = []
    for b in blocks:
        if b.get('type') == 'text' and isinstance(b.get('text'), str):
            if out and out[-1]['type'] == 'text':
                out[-1]['text'] += b['text']
            else:
                out.append({'type': 'text', 'text': b['text']})
        elif b.get('type') == 'tool_result':
            # is_error is not passed on: the model's format has no place for it, the text says it
            out.append({'type': 'tool_result', 'tool_use_id': str(b.get('tool_use_id', '')),
                        'content': text_of(b.get('content', ''), f'{where}: a tool_result')})
        else:
            raise BadRequest(f'{where}: only text and tool_result blocks are supported (got {b.get("type")!r})')
    return {'role': 'user', 'content_blocks': out}


def assistant_message(content, where):
    """An earlier reply for encoding.py: its thinking, text and tool_use blocks (step 157).
    A thinking block's signature is not read."""
    msg = {'role': 'assistant', 'content': ''}
    for b in blocks_of(content, where):
        kind = b.get('type')
        if kind == 'text' and isinstance(b.get('text'), str):
            msg['content'] += b['text']
        elif kind == 'thinking' and isinstance(b.get('thinking'), str):
            msg['reasoning_content'] = msg.get('reasoning_content', '') + b['thinking']
        elif kind == 'tool_use' and isinstance(b.get('name'), str) and isinstance(b.get('input'), dict):
            msg.setdefault('tool_calls', []).append({'id': str(b.get('id', '')), 'type': 'function',
                                                     'function': {'name': b['name'], 'arguments': b['input']}})
        else:
            raise BadRequest(f'{where}: only text, thinking and tool_use blocks are supported (got {kind!r})')
    return msg


def build_prompt(enc, req):
    """An Anthropic Messages request -> (prompt text, thinking_mode). Raises BadRequest."""
    msgs = req.get('messages')
    if not isinstance(msgs, list) or not msgs:
        raise BadRequest('messages: a non-empty list is required')
    out = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get('role') not in ('user', 'assistant'):
            raise BadRequest(f'messages[{i}]: role must be user or assistant')
        out.append((user_message if m['role'] == 'user' else assistant_message)(m.get('content'), f'messages[{i}]'))
    if out[-1]['role'] != 'user':
        raise BadRequest('the last message must be from the user (no prefill of the reply)')
    tools = req.get('tools') or []
    choice = (req.get('tool_choice') or {'type': 'auto'}).get('type')
    if choice == 'none':
        tools = []
    elif choice != 'auto':
        raise BadRequest(f'tool_choice {choice!r} is not supported: only auto and none')
    fns = []
    for i, t in enumerate(tools):
        if not isinstance(t, dict) or not isinstance(t.get('name'), str) or 'input_schema' not in t:
            raise BadRequest(f'tools[{i}]: a name and an input_schema are required (server tools are not supported)')
        fn = {'name': t['name'], 'parameters': t['input_schema']}
        if t.get('description'):
            fn['description'] = t['description']
        fns.append({'type': 'function', 'function': fn})
    system = req.get('system')
    system = text_of(system, 'system') if system is not None else ''
    if system or fns:
        head = {'role': 'system', 'content': system}
        if fns:
            head['tools'] = fns
        out.insert(0, head)
    thinking = (req.get('thinking') or {}).get('type', 'disabled')
    if thinking not in ('adaptive', 'enabled', 'disabled'):
        raise BadRequest(f'thinking.type {thinking!r}: adaptive, enabled or disabled')
    mode = 'chat' if thinking == 'disabled' else 'thinking'
    effort = (req.get('output_config') or {}).get('effort', 'high')
    if effort not in EFFORT:
        raise BadRequest(f'output_config.effort {effort!r}: one of {", ".join(EFFORT)}')
    try:
        return enc.encode_messages(out, thinking_mode=mode, reasoning_effort=EFFORT[effort]), mode
    except (AssertionError, NotImplementedError, ValueError, KeyError, TypeError) as e:
        raise BadRequest(f'the request could not be rendered: {e}')


def parse_reply(enc, text, mode, ended, new_id):
    """The engine's text -> (content blocks, stop_reason, a note for the log or '').
    ended: the model ended the reply itself (else it ran into max_tokens)."""
    blocks, note = [], ''
    msg = None
    if ended:
        try:
            msg = enc.parse_message_from_completion_text(text + enc.eos_token, mode)
            calls = [(c['function']['name'], json.loads(c['function']['arguments'])) for c in msg['tool_calls']]
            if any(not isinstance(a, dict) for _, a in calls):
                raise ValueError('tool arguments are not an object')
        except (AssertionError, ValueError, KeyError, TypeError) as e:
            msg, note = None, f'the reply did not parse ({e}); returned as text'
    if msg is None:  # cut by max_tokens, or not in the expected format: the reasoning, then all the rest as text
        reasoning, rest = '', text
        if mode == 'thinking':
            reasoning, sep, rest = text.partition(enc.thinking_end_token)
            if not sep:
                rest = ''
        msg, calls = {'reasoning_content': reasoning, 'content': rest}, []
    if msg.get('reasoning_content'):
        blocks.append({'type': 'thinking', 'thinking': msg['reasoning_content'], 'signature': SIGNATURE})
    if msg.get('content'):
        blocks.append({'type': 'text', 'text': msg['content']})
    for name, args in calls:
        blocks.append({'type': 'tool_use', 'id': new_id('toolu'), 'name': name, 'input': args})
    return blocks, ('max_tokens' if not ended else 'tool_use' if calls else 'end_turn'), note


class Live:
    """A reply as the Messages API's stream of events while it is generated (step 161).

    The model writes: reasoning </think> text, then perhaps a block of tool calls. feed(bytes)
    returns the events that the bytes so far allow: thinking and text as deltas, token by token.
    A tool call can only be parsed when the reply is complete, so from the tool block's first
    characters on nothing is sent until finish(), which gets the blocks that parse_reply() made
    of the whole text: it sends what was held back, then the tool_use blocks. What a stream
    carried as thinking and as text is exactly the message's thinking and text blocks; `differs`
    says if it ever is not.
    """

    def __init__(self, enc, mode):
        self.dec = codecs.getincrementaldecoder('utf-8')(errors='replace')  # a token can end inside a character
        self.state = 'thinking' if mode == 'thinking' else 'text'
        self.stops = {'thinking': enc.thinking_end_token, 'text': '\n\n<' + enc.dsml_token + enc.tool_calls_block_name}
        self.buf = ''
        self.index = -1
        self.open = None
        self.sent = {'thinking': '', 'text': ''}
        self.differs = False

    def _close(self, out):
        if self.open:
            out.append(('content_block_stop', {'type': 'content_block_stop', 'index': self.index}))
            self.open = None

    def _emit(self, out, kind, text):
        if not text:
            return
        if self.open != kind:
            self._close(out)
            self.index += 1
            self.open = kind
            first = {'type': 'thinking', 'thinking': '', 'signature': SIGNATURE} if kind == 'thinking' else {'type': 'text', 'text': ''}
            out.append(('content_block_start', {'type': 'content_block_start', 'index': self.index, 'content_block': first}))
        out.append(('content_block_delta', {'type': 'content_block_delta', 'index': self.index, 'delta': {'type': kind + '_delta', kind: text}}))
        self.sent[kind] += text

    def feed(self, data, final=False):
        out = []
        self.buf += self.dec.decode(data, final)
        while self.state != 'tools':
            stop = self.stops[self.state]
            i = self.buf.find(stop)
            if i >= 0:
                self._emit(out, self.state, self.buf[:i])
                if self.state == 'thinking':
                    self.state, self.buf = 'text', self.buf[i + len(stop):]
                    continue
                self.state, self.buf = 'tools', self.buf[i:]  # (the text block stays open: finish() may add to it)
                break
            # what could still become the stop string is held back: its longest start that ends the buffer
            hold = next((k for k in range(min(len(stop) - 1, len(self.buf)), 0, -1) if self.buf.endswith(stop[:k])), 0)
            self._emit(out, self.state, self.buf[:len(self.buf) - hold])
            self.buf = self.buf[len(self.buf) - hold:]
            break
        return out

    def finish(self, blocks):
        """The reply ended: `blocks` are the message's. -> the events that are left."""
        out = self.feed(b'', final=True)
        tools = [b for b in blocks if b['type'] == 'tool_use']
        if self.state == 'tools' and tools:
            self.buf = ''
        self._emit(out, 'text' if self.state != 'thinking' else 'thinking', self.buf)  # held back and never a stop string; a tool block that did not parse
        self._close(out)
        for b in tools:
            self.index += 1
            out.append(('content_block_start', {'type': 'content_block_start', 'index': self.index, 'content_block': dict(b, input={})}))
            out.append(('content_block_delta', {'type': 'content_block_delta', 'index': self.index,
                                                'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(b['input'], ensure_ascii=False)}}))
            out.append(('content_block_stop', {'type': 'content_block_stop', 'index': self.index}))
        want = {k: ''.join(b[k] for b in blocks if b['type'] == k) for k in ('thinking', 'text')}
        self.differs = want != self.sent
        return out


class Handler(BaseHTTPRequestHandler):
    server_version = 'ds-server'
    # (HTTP/1.0: the connection closes after every response, which also ends a stream)

    def log_message(self, fmt, *a):
        pass

    def send_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, code, kind, message):
        log(f'{code} {kind}: {message}')
        self.send_json(code, {'type': 'error', 'error': {'type': kind, 'message': message}})

    def do_GET(self):
        s = self.server
        path = self.path.split('?')[0]
        if path == '/health':
            self.send_json(200, {'status': 'ok'})
        elif path == '/v1/models':
            self.send_json(200, {'object': 'list', 'data': [{'id': s.model_name, 'object': 'model', 'type': 'model', 'display_name': s.model_name}],
                                 'has_more': False, 'first_id': s.model_name, 'last_id': s.model_name})
        else:
            self.send_error_json(404, 'not_found_error', f'no such path: {path}')

    def do_POST(self):
        s = self.server
        path = self.path.split('?')[0]
        if path != '/v1/messages':
            return self.send_error_json(404, 'not_found_error', f'no such path: {path}')
        try:
            req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            if not isinstance(req, dict):
                raise ValueError('the body is not a JSON object')
        except ValueError as e:
            return self.send_error_json(400, 'invalid_request_error', f'the body is not JSON: {e}')
        try:
            prompt, mode = build_prompt(s.enc, req)
            n = req.get('max_tokens')
            if type(n) is not int or n < 1:
                raise BadRequest('max_tokens: a positive integer is required')
            temp = req.get('temperature', s.temp)
            top_p = req.get('top_p', s.top_p)
            seed = req.get('seed', s.seed)  # (not in the Messages API: a client can send it as an extra field)
            if not isinstance(temp, (int, float)) or not isinstance(top_p, (int, float)) or type(seed) is not int or seed < 0:
                raise BadRequest('temperature, top_p: numbers; seed: a non-negative integer')
            if temp == 0:
                top_p = 1.0
        except BadRequest as e:
            return self.send_error_json(400, 'invalid_request_error', str(e))

        stream = bool(req.get('stream'))
        number = next(s.counter)
        mid = f'msg_{s.started:x}{number:06d}'
        ids = itertools.count(1)
        result = {'tokens': 0}
        token = {}
        t0 = time.time()

        events = queue.Queue()  # a stream: what the engine's thread saw, for this thread to send

        def on_start(tokens, kept):
            result.update(t_start=time.time(), prompt_tokens=tokens)
            events.put(('start', tokens, kept))

        def on_token(piece):
            result['tokens'] += 1
            events.put(('tok', piece))

        def work():
            try:
                result['out'], result['end'] = s.engine.generate(prompt, n, float(temp), float(top_p), seed,
                                                                 on_start=on_start, on_token=on_token, token=token)
            except BadRequest as e:
                result['bad'] = str(e)
            except Exception as e:
                result['failed'] = str(e)
            events.put(('end',))  # (wakes this request's thread at once)

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        live = Live(s.enc, mode)

        def send_live(wait):
            """A stream: sends what the engine made since the last call (waits up to `wait` s for
            the first of it). -> False if the client is gone."""
            try:
                item = events.get(timeout=wait)
            except queue.Empty:
                return True
            while True:
                if item[0] == 'start':  # the engine took the request: the message starts, its content empty
                    out = [('message_start', {'type': 'message_start', 'message': {
                        'id': mid, 'type': 'message', 'role': 'assistant', 'model': s.model_name, 'content': [], 'stop_reason': None, 'stop_sequence': None,
                        'usage': {'input_tokens': item[1] - item[2], 'output_tokens': 0, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': item[2]}}})]
                elif item[0] == 'tok':
                    out = live.feed(item[1])
                else:
                    return True
                for name, data in out:
                    if not self.try_sse(name, data):
                        return False
                try:
                    item = events.get_nowait()
                except queue.Empty:
                    return True

        if stream:  # the reply's events as it is generated (step 161); pings while there is nothing to send
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
        gone = False
        t_said = t_ping = time.time()
        while worker.is_alive():
            if stream and not gone:
                n_sent = result['tokens']
                if not send_live(min(1.0, s.ping)):
                    gone = True
                    log(f'request {number}: the client left; cancelling')
                    s.engine.cancel(token)
                if result['tokens'] != n_sent:
                    t_ping = time.time()
            else:
                worker.join(min(1.0, s.ping))
            if worker.is_alive() and not gone and self.client_left():
                gone = True
                log(f'request {number}: the client left; cancelling')
                s.engine.cancel(token)
            if worker.is_alive() and time.time() - t_said >= s.progress:  # a long reply is otherwise silent until its end
                t_said = time.time()
                if 't_start' in result:
                    log(f'request {number}: {result["prompt_tokens"]} prompt tokens, {result["tokens"]} tokens so far, {t_said - result["t_start"]:.0f} s since it started')
                else:
                    log(f'request {number}: waiting for the engine, {t_said - t0:.0f} s')
            if stream and worker.is_alive() and not gone and time.time() - t_ping >= s.ping:
                t_ping = time.time()
                if not self.try_sse('ping', {'type': 'ping'}):
                    gone = True
                    log(f'request {number}: the client left; cancelling')
                    s.engine.cancel(token)
        if 'failed' in result:
            log(f'request {number}: {result["failed"]}')
            if stream:
                return self.try_sse('error', {'type': 'error', 'error': {'type': 'api_error', 'message': result['failed']}})
            return self.send_error_json(500, 'api_error', result['failed'])
        if 'bad' in result:
            if stream:
                log(f'request {number}: {result["bad"]}')
                return self.try_sse('error', {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': result['bad']}})
            return self.send_error_json(400, 'invalid_request_error', result['bad'])

        end = result['end']
        if end['stop'] == 'cancel':
            log(f'request {number}: cancelled after {end["gen_tokens"]} tokens, {time.time() - t0:.1f} s after it arrived')
            return
        text = result['out'].decode('utf-8', errors='replace')
        blocks, stop_reason, note = parse_reply(s.enc, text, mode, end['stop'] == 'eos', lambda p: f'{p}_{s.started:x}{number:06d}{next(ids):02d}')
        kept = int(end.get('kept', 0))
        message = {'id': mid, 'type': 'message', 'role': 'assistant', 'model': s.model_name, 'content': blocks,
                   'stop_reason': stop_reason, 'stop_sequence': None,
                   # kept: the prompt's first tokens that the engine's context already held (step 158);
                   # as the API counts, input_tokens are the others
                   'usage': {'input_tokens': int(end['prompt_tokens']) - kept, 'output_tokens': int(end['gen_tokens']),
                             'cache_creation_input_tokens': 0, 'cache_read_input_tokens': kept}}
        gen_s = float(end['gen_s'])
        log(f'request {number}: {mode}, {end["prompt_tokens"]} prompt tokens ({kept} kept) in {float(end["prompt_s"]):.1f} s, '
            f'{end["gen_tokens"]} tokens in {gen_s:.1f} s ({int(end["gen_tokens"]) / gen_s if gen_s > 0 else 0:.1f} t/s), '
            f'waited {result.get("t_start", t0) - t0:.1f} s, {stop_reason}: {", ".join(b["type"] for b in blocks) or "nothing"}'
            + (f'; {note}' if note else '') + ('; the client had left' if gone else ''))
        if s.dump_dir:
            with open(os.path.join(s.dump_dir, f'{mid}.json'), 'w', encoding='utf-8') as f:
                json.dump({'request': req, 'prompt': prompt, 'reply': text, 'message': message, 'note': note}, f, ensure_ascii=False, indent=1)
        if not stream:
            return self.send_json(200, message)
        if not gone and send_live(0):  # (what the loop above did not get to), then the reply's end
            for name, data in live.finish(blocks) + [
                    ('message_delta', {'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': message['usage']['output_tokens']}}),
                    ('message_stop', {'type': 'message_stop'})]:
                if not self.try_sse(name, data):
                    break
            if live.differs:
                log(f'request {number}: the stream\'s thinking or text is not the message\'s (a bug: please keep the dump)')

    def client_left(self):
        """The client closed its side: the socket is readable and has no bytes (a client that
        waits for its reply sends nothing more)."""
        try:
            if not select.select([self.connection], [], [], 0)[0]:
                return False
            return self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b''
        except (BlockingIOError, InterruptedError):
            return False
        except OSError:
            return True

    def sse(self, name, data):
        self.wfile.write(f'event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'.encode('utf-8'))
        self.wfile.flush()

    def try_sse(self, name, data):
        try:
            self.sse(name, data)
            return True
        except OSError:
            return False


def main():
    argv = sys.argv[1:]
    if '--' not in argv:
        print(__doc__, file=sys.stderr)
        return 2
    cut = argv.index('--')
    ap = argparse.ArgumentParser(usage='ds_server.py [options] -- ENGINE COMMAND ... --requests')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--encoding', default='~/llms/ds-flash/encoding/encoding.py', help="the model's encoding.py")
    ap.add_argument('--temp', type=float, default=1.0, help='for requests without a temperature (the model card: 1.0)')
    ap.add_argument('--top-p', type=float, default=0.95, help='for requests without a top_p (the model card: 0.95 or 1.0)')
    ap.add_argument('--seed', type=int, default=0, help='for requests without a seed: every request uses it, so a repeated request gives the same reply')
    ap.add_argument('--model-name', default='deepseek-v4.1-flash-1017', help='the model name in replies')
    ap.add_argument('--ping', type=float, default=10, help='seconds between ping events of a stream while the engine works')
    ap.add_argument('--progress', type=float, default=30, help='seconds between progress lines of a running request in the log')
    ap.add_argument('--dump', metavar='DIR', help='write every request, its prompt, the raw reply and the message to DIR/<message id>.json')
    a = ap.parse_args(argv[:cut])
    cmd = argv[cut + 1:]
    if not cmd or '--requests' not in cmd:
        print('the engine command after -- must have --requests', file=sys.stderr)
        return 2
    enc = load_encoding(a.encoding)
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)
    engine = Engine(cmd)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    srv.enc, srv.engine, srv.temp, srv.top_p, srv.seed = enc, engine, a.temp, a.top_p, a.seed
    srv.model_name, srv.dump_dir, srv.ping = a.model_name, a.dump, a.ping
    srv.progress = a.progress
    srv.counter, srv.started = itertools.count(1), int(time.time())
    log(f'listening on http://{a.host}:{srv.server_address[1]} (POST /v1/messages); defaults: temperature {a.temp}, top-p {a.top_p}, seed {a.seed}')
    print(f'listening {srv.server_address[1]}', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
