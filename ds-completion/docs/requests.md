# `--requests`: prompts on stdin, replies on stdout

README step 152. One engine process, loaded once, answers one request after another. Every
request runs on an empty context, so its reply is what a single run of that prompt prints
(`results/requests/requests_test.py` checks that). It is the engine's side of a server: a front
that speaks HTTP renders the prompt, starts the engine as a child and parses the reply.

    bin-hip/ds-completion -m $M --rt $F8 [--placement .. --cache-loads .. --dspark $D --spec --canon ..] --requests

Not with `-p`, `-f`, `-i`, `--chat`, `--think`, `--record`, `--serve`, `--teacher`. Everything
on stdout is the protocol below; the engine's own lines (`# ...`) go to stderr as always.

## The engine's first line

    ready

after the load (about two minutes).

## A request

    gen n=N [temp=T] [top_p=P] [seed=S] bytes=B
    <B bytes>

- The header is one line. The `B` bytes after its line end are the whole prompt as UTF-8 text
  with the special tokens written out, the first one (`<｜begin▁of▁sentence｜>`) included: what
  the model's `encoding.py` renders. The engine adds nothing.
- `n`: the most tokens to generate (capped by the room left under `--max-ctx`).
- `temp`, `top_p`, `seed`: as `--temp`, `--top-p`, `--seed`; the defaults are 0, 1, 0 for
  every request (greedy), not the command line's values.
- Empty lines between requests are skipped. The next header starts right after the `B` bytes.
- The end of input ends the engine.

## The reply

    prompt tokens=<prompt's tokens> n=<n after the cap>
    tok <id> <the token's bytes in hex>
    ...
    end stop=eos|length prompt_tokens=.. gen_tokens=.. prompt_s=.. gen_s=..

- `prompt` comes when the request is accepted, before the prompt runs.
- One `tok` line per generated token, as it is generated. The bytes are the token's text,
  special tokens included (`</think>`, the tool-call tags), in hex because one token can be a
  part of a UTF-8 character. A token without text has no third field. The end-of-sequence
  token is not sent.
- `stop=eos`: the model ended the reply; `stop=length`: `n` tokens were generated;
  `stop=cancel`: the request was cancelled (below).

## Cancelling a request

The signal `SIGUSR1` to the engine stops the request it is generating: the loop ends before
its next step, and the reply ends with `end stop=cancel` after the tokens made so far. The
next request starts on an empty context as always.

- During the prompt it takes effect before the next chunk of 512 tokens (up to ~10 s).
- A signal with no request running does nothing: it is forgotten when the next request is
  read. So send it only while waiting for a reply's `end`, and read up to that `end` before
  sending the next request.
- (Without `--requests` the signal has its default action: it ends the process.)

## Errors

    error <message>

in place of a reply, for a request with a mistake in a value, an empty prompt or one too long
for `--max-ctx`; the engine then reads the next request. If the header itself cannot be
followed (an unknown first word, an unknown field, no `bytes=`), the message ends with
`; closing` and the engine exits, since it cannot know where the next request starts.
An error inside the engine (for example a lost peer) ends the process as in any other mode.
