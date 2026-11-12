# thinking chat sessions against fresh runs on the reference's prompts (README step 109)
# commit 6788ced + step 109; sclk 6; tomb; 2026-11-11
# <tag>.q: the three questions; <tag>.out: the three replies, joined by \x1e (a reply may hold newlines: they were split by
#   running sessions of 1, 2 and 3 turns, each a prefix of the next; th20 and th45 by line)
#   scripts/chat.sh --think --effort E --cache-wait -n N < <tag>.q
#   th: -n 600, effort 75;  th20: -n 20 (every reply cut inside its reasoning);  th45: -n 45 (cut inside reasoning or answer);
#   long: -n 2500, effort max (replies of 1344, 2500 (cut, no answer) and 764 tokens: rewinds longer than the window ring)
# <tag>-ref2.txt, <tag>-ref3.txt: python3 scripts/chat_ref_prompts.py results/chat/think/ <tag> <effort> (encoding.py's rendering)
#   ds-completion -f <tag>-ref<k>.txt -n N (chat.sh's options without --chat -i) prints the session's reply k byte for byte: all 8 do.
# also: a session without --dspark --spec prints th's bytes; results/chat/session.out (step 108) and cpp-matrix, tcp-udp,
#   summary-json single-shot are unchanged
