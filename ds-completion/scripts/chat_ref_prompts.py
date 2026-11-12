#!/usr/bin/env python3
"""A thinking chat session against the reference's prompts (README step 109).

From a three-turn session's questions (DIR/TAG.q, one per line) and replies (DIR/TAG.out, joined
by \x1e, or one per line), writes for turns 2 and 3 the prompt the reference's encoding.py renders
for the conversation before the reply (DIR/TAG-ref<k>.txt, no BOS: ds-completion -f adds it) and
the reply itself (DIR/TAG-reply<k>.out). A fresh `ds-completion -f DIR/TAG-ref<k>.txt` must print
TAG-reply<k>.out byte for byte.

    python3 scripts/chat_ref_prompts.py DIR/ TAG EFFORT [ENCODING_DIR]
"""
import os
import sys

sys.path.insert(0, sys.argv[4] if len(sys.argv) > 4 else os.path.expanduser('~/llms/ds-flash/encoding'))
from encoding import encode_messages

S, tag, effort = sys.argv[1], sys.argv[2], int(sys.argv[3])
Q = open(S + tag + '.q').read().split('\n')[:3]
text = open(S + tag + '.out').read()
replies = text.split('\x1e')[:3] if '\x1e' in text else text.split('\n')[:3]
msgs = []
for k, q in enumerate(Q):
    msgs.append({"role": "user", "content": q})
    if k > 0:
        open(S + f'{tag}-ref{k + 1}.txt', 'w').write(encode_messages(msgs, thinking_mode="thinking", reasoning_effort=effort, add_default_bos_token=False))
        open(S + f'{tag}-reply{k + 1}.out', 'w').write(replies[k] + '\n')
    r, sep, a = replies[k].partition('</think>')
    msgs.append({"role": "assistant", "content": a if sep else "", "reasoning_content": r})
