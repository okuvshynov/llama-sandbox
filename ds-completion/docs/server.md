# `server/ds_server.py`: an HTTP front for the engine

README step 153. A small program (Python, standard library only) that speaks a part of the
Anthropic Messages API and runs the engine as a child process in its `--requests` mode
(`docs/requests.md`). It was made for a benchmark that sends single-turn requests with one tool
and then for the pi coding harness ([pi.md](pi.md)); it does what those need and refuses the
rest.

    python3 server/ds_server.py [--port 8080] [--host 127.0.0.1] [--temp 1 --top-p 0.95 --seed 0] \\
        [--dump DIR] -- bin-hip/ds-completion -m $M --rt $F8 --placement placements/study30.txt \\
        --dspark $D --spec --canon --cache-loads 800 --cache-period 8 --cache-wait --requests

Everything after `--` is the engine's command; it must have `--requests`. The server prints
`listening <port>` on stdout when the engine is loaded (about two minutes); its log and the
engine's own lines go to stderr. `--port 0` takes a free port. Ctrl-C stops both.

A client: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` (or the SDK's `base_url`), any API key.

## The format is the model's own

The prompt is rendered and the reply is parsed by the model's `encoding.py`
(`--encoding`, default `~/llms/ds-flash/encoding/encoding.py`): the system block, the tools'
schemas, the reasoning effort, the tool-call tags. The server only converts between the API's
JSON and that module's messages.

## `POST /v1/messages`

Read of a request:

| field | |
|---|---|
| `messages` | `user` and `assistant` turns, the last one from the user. A user turn: a string, or `text` and `tool_result` blocks (a result's content: a string or `text` blocks; `is_error` is not passed on, the model's format has no place for it). An assistant turn: a string, or `text`, `thinking` and `tool_use` blocks, as a reply of this server (a `signature` is not read) |
| `system` | a string or a list of `text` blocks |
| `tools` | `name`, `description`, `input_schema` (rendered into the system block); `strict` is ignored: nothing constrains the model's output to the schema |
| `tool_choice` | `auto` (the default) or `none`; `disable_parallel_tool_use` is ignored |
| `thinking` | `adaptive` or `enabled`: the model reasons first; absent or `disabled`: it does not. `budget_tokens` is ignored |
| `output_config.effort` | the model's reasoning effort: `low` 50, `medium` 62, `high` 75 (the default), `xhigh` 88, `max` 100. `low` / `high` / `max` are the model's own names; the two others are placed between them |
| `max_tokens` | required; capped by the room left in the context (`--max-ctx`) |
| `temperature`, `top_p` | default `--temp` 1 and `--top-p` 0.95 (the model card's); temperature 0 is greedy |
| `seed` | not in the API (a client can send it as an extra field); default `--seed`. One seed, one request: the same reply every time |
| `stream` | `true`: events, below |

Refused with 400 `invalid_request_error`: other block types (images, `redacted_thinking`),
`tool_choice` `any` / `tool` (forcing a tool would need constrained
output), a last message from the assistant, an unknown effort. Other fields (`model`, `betas`,
`metadata`, ...) are ignored.

The reply is a message with, in this order, a `thinking` block (the reasoning; `signature` is
the fixed word `ds-completion`: nothing is signed, but a client may send a block without a
signature back as text, and pi does), a `text` block, `tool_use` blocks (`input` as an object). `stop_reason`: `tool_use`,
`end_turn`, or `max_tokens`. `usage` has the prompt's and the reply's tokens (reasoning
included). `model` is `--model-name`.

- A reply cut by `max_tokens` is returned as far as it got: reasoning, and text if the
  reasoning had ended. A tool call cut in the middle is returned as text.
- A reply that does not parse (a broken tool call, arguments that are not JSON) is returned
  as text with `end_turn`, and the log says why.

## Streaming

With `stream: true` the reply comes as it is generated (step 161):

- `message_start` when the engine has taken the request, before the prompt runs; its usage has
  the prompt's tokens (`input_tokens`, and `cache_read_input_tokens` for those kept).
- `ping` every `--ping` seconds (default 10) while nothing else is sent: the prompt running,
  a request waiting for the engine, a tool call being written.
- The reasoning as `thinking_delta` events and the text as `text_delta` events, as the engine
  makes the tokens (a verify step's tokens come together, 38-60 tokens a second).
- **Tool calls at the end, whole.** A call can only be parsed when the reply is complete, so
  from the first characters of the tool block on nothing is sent until the reply ends; then
  each call as one `tool_use` block with its input in one `input_json_delta`. A long call (a
  file being written) is a silence with pings.
- `message_delta` (the stop reason, `output_tokens`) and `message_stop`.

The message a stream carries is the message the same request gets without `stream`: the
thinking and text sent in pieces are those blocks' contents exactly, also for a reply that is
cut or does not parse. (The last characters before a possible tool block, the two line ends,
are held back until the next token shows what they are.) An engine error after
`message_start` comes as an `error` event.

## One request at a time

Requests that arrive while the engine is busy wait for it (a stream gets pings meanwhile).

If a client closes its connection before its reply (a timeout, Ctrl-C), the request is
cancelled: the server checks every waiting connection once a second and sends the engine its
cancel signal (`docs/requests.md`); the engine stops within a step, or within a prompt chunk.
A request that was still waiting for the engine is dropped when its turn comes. The log has a
line for both. Stopping the server cancels a running request the same way.

## Other paths

`GET /v1/models` (one model).

`GET /health`: 200 `{"status": "ok"}` while the engine's process lives; 503 with the engine's
exit code when it is gone (every request is then a 500: restart the server).

## Metrics

`GET /metrics` (step 165): counters and gauges in the Prometheus text format, for a collector
to read every few seconds, or `curl -s http://127.0.0.1:8080/metrics`. It does not wait for
the engine, so it answers while a request runs. Every series is there from the server's start,
at 0. No histograms.

| metric | kind | what |
|---|---|---|
| `ds_requests_total{outcome}` | counter | requests to `POST /v1/messages`: `ok` (a message was made), `cancelled` (the client left), `bad_request` (refused by the server or the engine, a 400), `engine_error` (a 500) |
| `ds_stop_reason_total{reason}` | counter | completed requests by `end_turn`, `tool_use`, `max_tokens` |
| `ds_prompt_tokens_total{kind}` | counter | prompt tokens of completed requests: `run` through the model, or `kept` (already in the context) |
| `ds_generated_tokens_total` | counter | tokens of completed replies |
| `ds_prompt_seconds_total`, `ds_generation_seconds_total` | counter | the engine's time on prompts and on replies, completed requests |
| `ds_queue_wait_seconds_total` | counter | from a request's arrival to the engine taking it |
| `ds_first_token_seconds_total` | counter | from a request's arrival to its reply's first token (the wait and the prompt are in it) |
| `ds_context_total{how}` | counter | requests the engine took, by where it started: `continue`, `saved`, `empty` (`docs/requests.md`) |
| `ds_reply_parse_failures_total` | counter | replies returned as text because they did not parse |
| `ds_stream_mismatch_total` | counter | streams that differed from their message (a bug; should stay 0) |
| `ds_engine_up` | gauge | 1 while the engine's process lives |
| `ds_engine_busy`, `ds_requests_waiting` | gauge | a request is running; requests waiting behind it |
| `ds_saved_point_tokens` | gauge | where the saved point is (0: none) |
| `ds_engine_load_seconds`, `ds_start_time_seconds` | gauge | the load's time; when the server started |

Tokens and seconds are those of completed requests only: a cancelled or failed request adds to
`ds_requests_total` and, if the engine took it, to `ds_context_total`. So the page's ratios are
of the same requests:

- generation speed: `ds_generated_tokens_total / ds_generation_seconds_total`;
- prompt speed: `ds_prompt_tokens_total{kind="run"} / ds_prompt_seconds_total`;
- the share of prompt tokens that did not have to run: `kept / (kept + run)`;
- mean time to the first token: `ds_first_token_seconds_total / ds_requests_total{outcome="ok"}`.

With a collector these are rates over a window (`rate(a[5m]) / rate(b[5m])`). The counters
start at 0 with every start of the server.

Not there yet: the tokens in the engine's context, the speculative loop, the routed experts,
bytes read from storage, the GPU clock (the engine does not report them per request; plan
item 33, step 4).

## The log

One line per finished request: the mode, prompt tokens and seconds, reply tokens, seconds and
t/s, how long it waited for the engine, the stop reason and the blocks. While a request runs,
a line every `--progress` seconds (default 30) with its tokens so far. `--dump DIR` writes
each request, its prompt, the raw reply and the message to `DIR/<message id>.json`.

The engine's own lines are in the same stream. Per request, before the prompt runs, a line
`# context: ...` says where it starts (on from the context, from the saved point, or empty)
and, when the prompt and the context part, a second one says at which token and shows the
text on both sides (step 164, `docs/requests.md` "What the engine says about its choice").

## A conversation with tool calls

A client that runs tools (an agent's loop) sends the reply back as it got it, `thinking` and
`tool_use` blocks included, then a user turn with the `tool_result` blocks. `encoding.py` writes
a result into the user turn as `<tool_result>...</tool_result>`, and, when the request has tools,
keeps the reasoning of every earlier reply. So the next prompt is the last prompt, the reply as
the model wrote it, and the new turn: the test checks this to the character for a reply with
reasoning and a tool call. The engine goes on from its context for such a request and runs only
the new tokens (step 158, `docs/requests.md` "A kept context"); the message's usage has them as
`input_tokens` and the kept ones as `cache_read_input_tokens`. A new conversation with the
same system prompt and tools as the last ones returns to a point saved at their end (step 160);
the first two conversations after the server's start run whole.

Without tools in the request, `encoding.py` drops the reasoning of replies before the last user
turn that carries text, as the model's chat format says.

## pi

The pi coding harness (`pi.dev`) works with the server (step 157). The whole setup, step by
step, for a machine that has none of it: [pi.md](pi.md). In short: its side is
`server/pi-models.json`: copy it to pi's agent directory as `models.json` (`~/.pi/agent`, or
`PI_CODING_AGENT_DIR`) and set `baseUrl` to the server's address, without `/v1`. Then
`~/projects/pi/pi-test.sh --model ds/deepseek-v4.1-flash-1017` (pi from its sources). What the file says: the Anthropic Messages API, any
key, text only, a 65,536-token context (`--max-ctx`), pi's thinking levels as efforts
(`forceAdaptiveThinking`; `--thinking off` is the model's chat mode). `results/server/pi.txt`:
two tasks in pi's print mode, 47 and 71 s; `results/server/pi-kept.txt`: 33 and 30 s with the
context kept between a task's requests (step 158); `results/server/pi-saved.txt`: a third task
in 7.6 s, its system prompt and tools (~1,500 tokens, 14 s for the first two tasks) taken
from the saved point (step 160).

## Tests

`server/test_server.py`: the server against `server/fake_engine.py` (no model, seconds).
With the model: `results/server/endgame.txt`, `results/server/pi.txt`, and
`results/server/stream_check.py` (a stream against the same request without it).
