# `--requests`: prompts on stdin, replies on stdout

README step 152. One engine process, loaded once, answers one request after another. A
request's reply is what a single run of that prompt prints (`results/requests/requests_test.py`
checks that): it runs on an empty context, or goes on from the last request's context where
that changes nothing but the time ("A kept context" below, step 158). It is the engine's side of a server: a front
that speaks HTTP renders the prompt, starts the engine as a child and parses the reply.

    bin-hip/ds-completion -m $M --rt $F8 [--placement .. --cache-loads .. --dspark $D --spec --canon ..] --requests

Not with `-p`, `-f`, `-i`, `--chat`, `--think`, `--record`, `--serve`, `--teacher`. Everything
on stdout is the protocol below; the engine's own lines (`# ...`) go to stderr as always.

## The engine's first line

    ready

after the load (about two minutes).

## A request

    gen n=N [temp=T] [top_p=P] [seed=S] [keep=0|1] bytes=B
    <B bytes>

- The header is one line. The `B` bytes after its line end are the whole prompt as UTF-8 text
  with the special tokens written out, the first one (`<｜begin▁of▁sentence｜>`) included: what
  the model's `encoding.py` renders. The engine adds nothing.
- `n`: the most tokens to generate (capped by the room left under `--max-ctx`).
- `temp`, `top_p`, `seed`: as `--temp`, `--top-p`, `--seed`; the defaults are 0, 1, 0 for
  every request (greedy), not the command line's values.
- `keep=0`: run on an empty context even if the prompt goes on from the last one (a control
  for tests; the reply is the same).
- Empty lines between requests are skipped. The next header starts right after the `B` bytes.
- The end of input ends the engine.

## The reply

    prompt tokens=<prompt's tokens> n=<n after the cap> kept=<tokens>
    tok <id> <the token's bytes in hex>
    ...
    end stop=eos|length prompt_tokens=.. kept=.. gen_tokens=.. prompt_s=.. gen_s=..

- `prompt` comes when the request is accepted, before the prompt runs. `kept`: how many of
  the prompt's first tokens the context already held; only the others run (`prompt_s` is
  their time).
- One `tok` line per generated token, as it is generated. The bytes are the token's text,
  special tokens included (`</think>`, the tool-call tags), in hex because one token can be a
  part of a UTF-8 character. A token without text has no third field. The end-of-sequence
  token is not sent.
- `stop=eos`: the model ended the reply; `stop=length`: `n` tokens were generated;
  `stop=cancel`: the request was cancelled (below).

## Cancelling a request

The signal `SIGUSR1` to the engine stops the request it is generating: the loop ends before
its next step, and the reply ends with `end stop=cancel` after the tokens made so far. The
context keeps what was run, as after any request.

- During the prompt it takes effect before the next chunk of 512 tokens (up to ~10 s).
- A signal with no request running does nothing: it is forgotten when the next request is
  read. So send it only while waiting for a reply's `end`, and read up to that `end` before
  sending the next request.
- (Without `--requests` the signal has its default action: it ends the process.)

## A kept context

After a request the context holds its prompt and the reply's tokens that were run through the
model: all of them but the last one of a reply cut by `n`, and never the end of sequence. If
the next prompt's tokens start with all of these and have more, only the others are run
(`kept=N`); any other prompt starts on an empty context (`kept=0`). The reply does not depend
on which happened: a position's state is the same however the tokens before it were split
into forwards (checked: the same request with `keep=0`, greedy and sampled).

This is the case of a conversation sent again with one more turn, as an agent's loop does:
the prompt is the last prompt, the last reply, and the new turn. It is all or nothing: a prompt
that shares only a part of the context (another conversation with the same system prompt, a
reply whose text tokenizes differently than the model wrote it) runs whole.

## Errors

    error <message>

in place of a reply, for a request with a mistake in a value, an empty prompt or one too long
for `--max-ctx`; the engine then reads the next request. If the header itself cannot be
followed (an unknown first word, an unknown field, no `bytes=`), the message ends with
`; closing` and the engine exits, since it cannot know where the next request starts.
An error inside the engine (for example a lost peer) ends the process as in any other mode.
