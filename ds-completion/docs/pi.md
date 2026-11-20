# The pi coding harness on this engine: setup

README steps 157-158. [pi](https://github.com/earendil-works/pi) is a coding agent for the
terminal (tools: `read`, `bash`, `edit`, `write`). It talks to `server/ds_server.py` over the
Anthropic Messages API. This page is the whole setup on a machine that has none of it, in the
order to do it. What was run and measured: `results/server/pi.txt`, `results/server/pi-kept.txt`.

Three parts, which need not be on one machine:

| part | where | needs |
|---|---|---|
| the engine and the server | the Mac Pro with the model | everything in [QUICKSTART.md](../QUICKSTART.md), and the model's `encoding.py` |
| pi | any machine that reaches the server | Node 22.19 or newer |
| pi's `models.json` | pi's machine | the server's address |

## 1. The engine and the server

1. Build the engine and check that a single prompt runs, as [QUICKSTART.md](../QUICKSTART.md)
   says (`make && make HIP=1`, the model files under `~/llms`). The checkout must be at step
   158 or later: `git log --oneline -1` in `ds-completion` shows `bec889e` or a later commit
   (step 160 for the saved point, 161 for the stream, under "What to expect").
2. The server also reads the model's own chat format, which is not in the GGUF files:
   `~/llms/ds-flash/encoding/encoding.py`, from the model's Hugging Face repository
   (the directory `encoding/` of DeepSeek-V4.1-Flash). Another place: `--encoding PATH`.
3. After a reboot, read the model files once, or the first minutes are several times slower:

       cat ~/llms/UD-Q8_K_XL/*.gguf > /dev/null        # 508 GB, ~3 min

4. Start the server. It starts the engine itself; everything after `--` is the engine's
   command.

       cd ~/projects/llama-sandbox/ds-completion
       M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
       server/ds_server.py --port 8080 -- \
           bin-hip/ds-completion -m $M --rt ~/llms/ds-1017-fp8/dense-fp8.gguf \
           --placement placements/study30.txt \
           --dspark ~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf --spec \
           --canon --cache-loads 800 --cache-period 8 --cache-wait --requests

   It prints `listening 8080` after about two minutes. Its log (stderr) has one line per
   request. Ctrl-C stops the server and the engine.

   For pi on another machine add `--host 0.0.0.0` before the `--`: by default the server
   listens on 127.0.0.1 only. It has no password and no encryption, so only on a network you
   trust.

5. Check it without pi (the model-free test, then one real request):

       server/test_server.py                           # "all checks passed", a few seconds
       curl -s http://127.0.0.1:8080/v1/messages -H 'content-type: application/json' -d \
         '{"max_tokens": 50, "messages": [{"role": "user", "content": "Reply with the single word: yes"}]}'

   The reply is a JSON message whose `content` has the text `yes`.

## 2. Node and pi

pi is run from its sources here; nothing is installed system-wide.

1. Node 22.19 or newer. With nvm (as on the first machine, Node 26.13.1):

       curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.41.0/install.sh | bash
       # a new shell, then:
       nvm install 26
       node -v

   nvm puts Node on the PATH of interactive shells only. In a script or over `ssh host cmd`:
   `export PATH=$HOME/.nvm/versions/node/<version>/bin:$PATH`.

2. pi, at the commit that was tested (a later one will most likely work as well):

       git clone https://github.com/earendil-works/pi.git ~/projects/pi
       cd ~/projects/pi
       git checkout f0f9478a2
       npm install --ignore-scripts                    # ~700 MB in node_modules
       npm run build                                   # also fetches pi's model data; a few minutes

   Without `npm run build` pi stops at start with `Cannot find module
   .../providers/data/amazon-bedrock.json`: the model data is not in the repository.

3. There is no `pi` command after this: pi is started by the script in its checkout,
   `~/projects/pi/pi-test.sh`, from any directory. The commands below write it out.

       ~/projects/pi/pi-test.sh --version              # prints a version, e.g. 1.5.2

   For a `pi` command in interactive shells, add `alias pi=~/projects/pi/pi-test.sh` to
   `~/.bashrc`; scripts still need the path. (pi's own installer, `curl -fsSL
   https://pi.dev/install.sh | sh`, gives a real `pi` command of the released version; that
   was not tried here.)

## 3. pi's `models.json`

pi learns about the server from one file in its agent directory, `~/.pi/agent/models.json`
(another directory: the environment variable `PI_CODING_AGENT_DIR`).

    mkdir -p ~/.pi/agent
    cp ~/projects/llama-sandbox/ds-completion/server/pi-models.json ~/.pi/agent/models.json

If the server is on another machine or port, edit `baseUrl` in the copy: the server's address
without `/v1`, for example `http://192.168.1.20:8080`. If the directory already has a
`models.json`, add the provider `ds` to its `providers` instead of copying over it.

What the file says, and why:

| field | value | why |
|---|---|---|
| `api` | `anthropic-messages` | the API the server speaks |
| `apiKey` | any word | pi wants one; the server does not read it |
| `reasoning`, `compat.forceAdaptiveThinking` | `true` | pi then sends a reasoning effort (low ... max), which the server maps to the model's 1-100; without the second one pi sends a token budget, which the server ignores |
| `thinkingLevelMap` | pi's level -> effort | pi's `minimal` is the model's `low`; `xhigh` and `max` are passed on |
| `input` | `["text"]` | the server refuses images |
| `contextWindow` | 65536 | the engine's `--max-ctx` default; change both together |
| `maxTokens` | 16384 | the most pi asks for in one reply |
| `cost` | zeros | pi requires the fields |

Check that pi sees it:

    ~/projects/pi/pi-test.sh --offline --list-models ds

prints one line: `ds  deepseek-v4.1-flash-1017  65.5K  16.4K  yes  no`.

## 4. Running it

In the directory pi should work in (it reads and writes files there and runs shell commands
without asking):

    cd <a project>
    ~/projects/pi/pi-test.sh --offline --model ds/deepseek-v4.1-flash-1017 --thinking medium

`--offline` skips pi's own network calls at start (updates, model catalogs). `--thinking`:
`off` (the model's chat mode, no reasoning), `minimal`/`low`, `medium`, `high`, `xhigh`, `max`.
To make these the defaults, in `~/.pi/agent/settings.json`:

    { "defaultProvider": "ds", "defaultModel": "deepseek-v4.1-flash-1017", "defaultThinkingLevel": "medium" }

One task without the interface (this is what was tested):

    ~/projects/pi/pi-test.sh --offline --no-session --model ds/deepseek-v4.1-flash-1017 --thinking medium \
        -p "What does notes.txt say?" < /dev/null

**The `< /dev/null` matters:** with `-p`, pi waits for its standard input to end before it
sends anything. In a terminal that looks like a hang with no request in the server's log.

A first check, the one in `results/server/pi.txt`:

    mkdir /tmp/pi-check && cd /tmp/pi-check && echo "The answer is 42." > notes.txt
    ~/projects/pi/pi-test.sh --offline --no-session --model ds/deepseek-v4.1-flash-1017 --thinking medium \
        -p "What does notes.txt say?" < /dev/null

About half a minute, two lines in the server's log (a tool call, then the answer), and the
second one says `(1565 kept)` or a number near it.

## What to expect

- **Reasoning and text appear as they are generated** (step 161; 38-60 tokens per second).
  **A tool call appears when it is complete:** while the model writes one (a file's contents
  in a `write`, for example) nothing arrives, and pi shows its working indicator.
- **The first request of the first two conversations takes ~15 s before any reply** (~30 s
  for the very first after the server's start): pi's system prompt and tools are ~1,500
  tokens, processed at ~100 tokens per second. A longer `AGENTS.md` or more tools add to it.
  From the third conversation on they are taken from a saved point (step 160) and the reply
  starts within a second, as long as the system prompt and tools stay the same (the same
  directory, the same `AGENTS.md`).
- **Later requests of the same conversation start in about a second:** the engine goes on
  from its context (`(N kept)` in the log). It keeps one context: a second pi session, or
  anything else sent to the server in between, makes the next request run whole again.
- **One request at a time.** A second one waits for the first.
- **Esc in pi cancels a request:** pi closes the connection and the server stops the engine
  within a second (a prompt being processed: within ~10 s).
- **The context is 65,536 tokens.** pi compacts a conversation (summarizes its older part,
  with a request to the same server) when it passes `contextWindow` minus 16,384, so at about
  49,000 tokens. A longer context: the engine's `--max-ctx N` (up to 1M) and the same number in
  `models.json`.
- **Replies are sampled** (temperature 1, top-p 0.95) with one fixed seed: the same
  conversation gives the same replies. Another seed: the server's `--seed N`.

## When it does not work

| what is seen | cause |
|---|---|
| pi prints nothing, the server's log has no request | `-p` without `< /dev/null`; or `baseUrl` points elsewhere (`curl <baseUrl>/health`) |
| `No models available` | `models.json` is not in the agent directory pi uses, or is not valid JSON |
| `node: command not found` in a script | nvm's PATH (section 2) |
| an error naming a block type (`image`, `redacted_thinking`) | the server refuses it: [server.md](server.md); a conversation started with another model can carry such blocks |
| `the prompt is N tokens; --max-ctx is 65536` | the conversation outgrew the context: compact it in pi (`/compact`) or raise both limits |
| every request `(0 kept)` | something else uses the server between pi's requests; or the server is older than step 158. The engine's `# context: they part at token N: ...` line in the server's log shows where the prompt and the context part (step 164) |
| much slower than above | a cold page cache after a reboot (section 1, item 3) |
| `malloc_consolidate(): invalid chunk size` when the server stops | a known abort at the engine's exit (steps 91, 93); nothing is lost |

To see exactly what pi sent and what the model wrote, start the server with `--dump DIR`:
one JSON file per request with the request, the rendered prompt, the raw reply and the message.

## Not tested

Only pi's print mode (`-p`) was run, on two small tasks. The interactive interface uses the
same requests, but it was not driven here. Not tried: pi's extensions, MCP servers, sessions
resumed across restarts of the server, conversations long enough to be compacted, and pi on a
different machine than the server.
