# `server/ds_server.py`: an HTTP front for the engine

README step 153. A small program (Python, standard library only) that speaks a part of the
Anthropic Messages API and runs the engine as a child process in its `--requests` mode
(`docs/requests.md`). It exists for a benchmark that sends single-turn requests with one tool;
it does what that needs and refuses the rest.

    server/ds_server.py [--port 8080] [--host 127.0.0.1] [--temp 1 --top-p 0.95 --seed 0] \\
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

A tool call can only be parsed when the reply is complete, so a stream does not carry the
reply as it is generated: the server sends `ping` events (every `--ping` seconds, default 10)
while the engine works, and then all the message's events at once (`message_start`, per block
`content_block_start` / `content_block_delta` / `content_block_stop`, `message_delta`,
`message_stop`). This keeps a client's read timeout from firing during long reasoning.

## One request at a time

Requests that arrive while the engine is busy wait for it (a stream gets pings meanwhile).

If a client closes its connection before its reply (a timeout, Ctrl-C), the request is
cancelled: the server checks every waiting connection once a second and sends the engine its
cancel signal (`docs/requests.md`); the engine stops within a step, or within a prompt chunk.
A request that was still waiting for the engine is dropped when its turn comes. The log has a
line for both. Stopping the server cancels a running request the same way.

## Other paths

`GET /v1/models` (one model), `GET /health`.

## The log

One line per finished request: the mode, prompt tokens and seconds, reply tokens, seconds and
t/s, how long it waited for the engine, the stop reason and the blocks. While a request runs,
a line every `--progress` seconds (default 30) with its tokens so far. `--dump DIR` writes
each request, its prompt, the raw reply and the message to `DIR/<message id>.json`.

## A conversation with tool calls

A client that runs tools (an agent's loop) sends the reply back as it got it, `thinking` and
`tool_use` blocks included, then a user turn with the `tool_result` blocks. `encoding.py` writes
a result into the user turn as `<tool_result>...</tool_result>`, and, when the request has tools,
keeps the reasoning of every earlier reply. So the next prompt is the last prompt, the reply as
the model wrote it, and the new turn: the test checks this to the character for a reply with
reasoning and a tool call. Each request is still processed from its first token (a prefix kept
between requests: not yet).

Without tools in the request, `encoding.py` drops the reasoning of replies before the last user
turn that carries text, as the model's chat format says.

## pi

The pi coding harness (`pi.dev`) works with the server (step 157). Its side is
`server/pi-models.json`: copy it to pi's agent directory as `models.json` (`~/.pi/agent`, or
`PI_CODING_AGENT_DIR`) and set `baseUrl` to the server's address, without `/v1`. Then
`pi --model ds/deepseek-v4.1-flash-1017`. What the file says: the Anthropic Messages API, any
key, text only, a 65,536-token context (`--max-ctx`), pi's thinking levels as efforts
(`forceAdaptiveThinking`; `--thinking off` is the model's chat mode). `results/server/pi.txt`:
two tasks in pi's print mode. Every request of a task is processed from its first token, so
most of the time is prompts.

## Tests

`server/test_server.py`: the server against `server/fake_engine.py` (no model, seconds).
With the model: `results/server/endgame.txt`, `results/server/pi.txt`.
