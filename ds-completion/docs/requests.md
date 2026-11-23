# `--requests`: prompts on stdin, replies on stdout

README step 152. One engine process, loaded once, answers one request after another. A
request's reply is what a single run of that prompt prints (`results/requests/requests_test.py`
checks that): it runs on an empty context, or goes on from the last request's context where
that changes nothing but the time ("A kept context" below, step 158). It is the engine's side of a server: a front
that speaks HTTP renders the prompt, starts the engine as a child and parses the reply.

    bin-hip/ds-completion -m $M --rt $F8 [--placement .. --cache-loads .. --dspark $D --spec --canon ..] --requests

Not with `-p`, `-f`, `-i`, `--chat`, `--think`, `--record`. Everything
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

    prompt tokens=<prompt's tokens> n=<n after the cap> kept=<tokens> ctx=<tokens> common=<tokens> how=continue|saved|empty
    tok <id> <the token's bytes in hex>
    ...
    end stop=eos|length prompt_tokens=.. kept=.. saved=.. gen_tokens=.. prompt_s=.. gen_s=..

- `prompt` comes when the request is accepted, before the prompt runs. `kept`: how many of
  the prompt's first tokens the context already held; only the others run (`prompt_s` is
  their time). `saved` (in `end`): where the saved point is after this request, in tokens; 0:
  there is none ("A kept context"). `ctx`, `common`, `how` (step 164): how the engine chose
  where to start, "What the engine says about its choice" below.
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
(`kept=N`); any other prompt returns to the saved point below, or starts on an empty context
(`kept=0`). The reply does not depend
on which happened: a position's state is the same however the tokens before it were split
into forwards (checked: the same request with `keep=0`, greedy and sampled).

This is the case of a conversation sent again with one more turn, as an agent's loop does:
the prompt is the last prompt, the last reply, and the new turn.

**A saved point** (step 160) is for prompts that share only their start with the context:
another conversation with the same system prompt and tools.

- When a prompt does not go on from the context but its first tokens, 64 or more, are the
  context's, it runs from the empty context, and the engine saves a point at the end of what
  the two share (`saved=N`). So the point is set by the second conversation, not the first.
- A later prompt that starts with the point's tokens and does not go on from the context
  returns to the point and runs the rest (`kept=N`).
- The point stays where it is, also when a prompt shares more than it. A prompt that does not
  start with all of its tokens ends it (`saved=0`): that prompt runs from the empty context
  over the positions before the point. The next pair of prompts that share a start sets a new
  one.
- The point is where two prompts part, which can be past the system prompt: two users' turns
  that start with the same words put it after those words, and the next conversation that
  starts otherwise ends it and sets one earlier. That costs one prompt run whole.
- There is one point. Two groups of conversations with two system prompts, taking turns, end
  each other's point every time.

A reply never depends on any of this.

## What the engine says about its choice

Step 164. In the `prompt` line:

- `ctx`: the tokens the context held when the request came (the last request's prompt and
  the reply's tokens that were run; 0 for the first request).
- `common`: how many of them the prompt starts with.
- `how`: `continue` (the prompt starts with all of the context and has more: `kept` = `ctx`),
  `saved` (it returned to the saved point: `kept` = the point), `empty` (it runs whole:
  `kept` = 0).

On stderr one line per accepted request says the same in words, with what becomes of the
saved point, and a second line when the prompt and the context part before either ends:

    # context: held 1527 tokens, the prompt (1526 tokens) starts with 1518 of them: it starts on the empty context; a point is saved at 1518
    # context: they part at token 1518: the context has 10232 "Color", the prompt 40 "F"; before it " jumps over the lazy dog. <｜User｜>", then the context "Color: name one.<｜Assistant｜></think>Brown", the prompt "Fruit: name one.<｜Assistant｜></think>"

- The first line: `it goes on from the context (N kept)`, `it returns to the saved point (N
  kept)` or `it starts on the empty context`, the last with its reason where there is one
  (`(keep=0)`; `(it has no token after the context's)` for a prompt that is the context
  itself), then `; a point is saved at N`, `; the saved point at N ends (the prompt starts
  with M of its tokens)`, or both (the old point ends and one is saved earlier). With an empty context: `# context: empty; the prompt has N tokens: ...`.
- The second line: the index of the first token that differs (the first token is 0), its id
  and text on each side, the text of the 8 tokens before it, and of 8 tokens from it on each
  side. Special tokens are written out, a line end as `\n`. A token can be a part of a
  character, so a text's first or last character may be broken.
- A prompt that ends inside the context (the same request sent again) has
  `# context: the prompt ends inside the context, at token N; the context goes on "..."`.

**When a conversation's next turn does not go on** (`how=empty` with `common` far below
`ctx`), the second line shows where the client's prompt and the model's own tokens part: a
reply that the client sent back in other words than the model wrote (a changed space, a tool
call's arguments written out again, the reasoning left out), or another request in between.

## Errors

    error <message>

in place of a reply, for a request with a value out of range (`n` below 1, `temp`, `top_p`), an empty prompt or one too long
for `--max-ctx`; the engine then reads the next request. If the header itself cannot be
followed (an unknown first word, a field that is unknown or not a number, no `bytes=`), the message ends with
`; closing` and the engine exits, since it cannot know where the next request starts.
An error inside the engine (for example a chunk past `--max-ctx`) ends the process as in any other mode.
