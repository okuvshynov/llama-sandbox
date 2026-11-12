# an interactive chat session against fresh runs on the whole conversation (README step 108)
# printf '%s\n' "Name three prime numbers above 100." "Now multiply the first two of them." "Write a C function that checks whether a number is prime." | scripts/chat.sh --cache-wait -n 200 > session.out   (and -n 12: session-n12.out)
# commit e9b68b1 + step 108; sclk 6; tomb; 2026-11-11
# whole.txt, whole-n12.txt: the conversation before the third reply as one raw prompt (the session's own replies, encoding.py's chat format);
#   ds-completion -f whole.txt -n 200 (same options, no --chat -i) prints the session's third reply byte for byte; likewise the second reply
#   from the conversation before it, both at -n 200 and -n 12 (replies cut by -n), and a session without --dspark --spec prints the same bytes.
# single-shot outputs unchanged: cpp-matrix, tcp-udp, summary-json identical to results/cache-test/plain-canon-static
