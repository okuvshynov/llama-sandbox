#!/usr/bin/env bash
# step 181: the longer golden sets, sessions and sampling on the build after the removal
cd "$(dirname "$0")/../.."
O=data/step181
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
D=~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf
say() { echo "$(date +%H:%M:%S) $*"; }
FRESH="./bin-hip/ds-completion -m $M --rt $F8 --placement placements/study30.txt --dspark $D --spec --canon --cache-loads 800 --cache-period 8 --cache-wait"

say "== sessions"
sess() { # tag stored n args...
  local tag=$1 stored=$2 q=$3; shift 3
  say "session $tag: $*"
  scripts/chat.sh --cache-wait "$@" < $q > $O/s-$tag.out 2> $O/s-$tag.err; echo "  rc=$? $(python3 $O/cmp_session.py session $O/s-$tag.out $stored)"
}
fresh() { # tag file n stored k
  say "fresh $1 reply $5 (-f $2 -n $3)"
  $FRESH -f $2 -n $3 > $O/f-$1-$5.out 2> $O/f-$1-$5.err < /dev/null; echo "  rc=$? $(python3 $O/cmp_session.py fresh $O/f-$1-$5.out $4 $5)"
}
printf '%s\n' "Name three prime numbers above 100." "Now multiply the first two of them." "Write a C function that checks whether a number is prime." > $O/chat.q
sess chat200 results/chat/session.out $O/chat.q -n 200
sess chat12 results/chat/session-n12.out $O/chat.q -n 12
fresh chat200 results/chat/whole.txt 200 results/chat/session.out 3
fresh chat12 results/chat/whole-n12.txt 12 results/chat/session-n12.out 3
T=results/chat/think
sess th $T/th.out $T/th.q --think --think-show all --effort 75 -n 600
sess th20 $T/th20.out $T/th20.q --think --think-show all --effort 75 -n 20
sess th45 $T/th45.out $T/th45.q --think --think-show all --effort 75 -n 45
sess long $T/long.out $T/long.q --think --think-show all --effort max -n 2500
for k in 2 3; do
  fresh th $T/th-ref$k.txt 600 $T/th.out $k
  fresh th20 $T/th20-ref$k.txt 20 $T/th20.out $k
  fresh th45 $T/th45-ref$k.txt 45 $T/th45.out $k
  fresh long $T/long-ref$k.txt 2500 $T/long.out $k
done

say "== sampling"
sed "s#results/sampler/step151/spec#$O/spec#g" results/sampler/step151/spec.matrix > $O/spec.matrix
python3 scripts/run_matrix.py results/sampler/plain.matrix --prompts tcp-udp,cpp-matrix,geometry --out $O/plain > $O/plain.log 2>&1; say "plain rc=$?"
python3 $O/cmp_hash.py $O/plain results/sampler/plain
python3 scripts/run_matrix.py $O/spec.matrix --prompts tcp-udp,cpp-matrix,geometry --out $O/spec > $O/spec.log 2>&1; say "spec rc=$?"
python3 $O/cmp_hash.py $O/spec results/sampler/step151/spec
python3 scripts/run_matrix.py results/sampler/spec-heldout.matrix --out $O/spec-heldout > $O/spec-heldout.log 2>&1; say "spec-heldout rc=$?"
python3 $O/cmp_hash.py $O/spec-heldout results/sampler/step151/spec-heldout
bin/test-sampler > $O/test-sampler.txt 2>&1; say "test-sampler rc=$? $(tail -1 $O/test-sampler.txt)"

say "== golden sets, CPU reference path"
t() { # name stored cmd...
  local name=$1 stored=$2; shift 2
  say "$*"
  "$@" > $O/t-$name.txt 2> $O/t-$name.err; echo "  rc=$? $(python3 $O/cmp_table.py $O/t-$name.txt $stored)"
}
t hello-fp32-nq results/test-hello-fp32-nq.txt bin/ds-test -m $M -g data/golden/hello-fp32-nq --no-quant
t readme200 results/test-readme200.txt bin/ds-test -m $M -g data/golden/readme200
t readme200-fp32-nq results/test-readme200-fp32-nq.txt bin/ds-test -m $M -g data/golden/readme200-fp32-nq --no-quant
t readme1100 results/test-readme1100.txt bin/ds-test -m $M -g data/golden/readme1100
say "== golden set, 20,000 tokens, the runtime"
say "end to end"
bin-hip/ds-test -m $M -g data/golden/readme20k-s --runtime $F8 --e2e $O/readme20k-rt.f32 > $O/t-20k-e2e.txt 2> $O/t-20k-e2e.err; echo "  rc=$?"
cmp $O/readme20k-rt.f32 data/ds/readme20k-rt.f32 && echo "  logits: the stored bytes"
~/venvs/ds-ref/bin/python ref/compare_logits.py data/ref/readme20k-s.pt $O/readme20k-rt.f32 > $O/compare-20k.txt 2>&1; echo "  compare rc=$? $(tail -1 $O/compare-20k.txt)"
t 20k-rt results/rt-readme20k.txt bin-hip/ds-test -m $M -g data/golden/readme20k-s --runtime $F8 --dump-topk $O/readme20k-topk.bin
cmp $O/readme20k-topk.bin data/ds/readme20k-topk.bin && echo "  top-k dump: the stored bytes"
say "== done"
