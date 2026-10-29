# bin-hip/ds-test -m $M -g data/golden/readme20k --runtime $F8 --e2e data/ds/readme20k-rt.f32
# commit ddf9b43 + ds-test --max-ctx; sclk 2: 1204Mhz *; 2026-10-29
# runtime: 6.37 GiB of fp8 weights on GPU 0
# runtime: 1.75 GiB of small weights on GPU 0
# golden data/golden/readme20k: 9 steps; exact fake-quant on
step  0: 20000 tokens ->     28 (ref     28)  1684.14 s
step  1:   1 tokens ->  12919 (ref  12919)  0.09 s
step  2:   1 tokens ->   3234 (ref   3234)  0.09 s
step  3:   1 tokens ->  12139 (ref  12139)  0.09 s
step  4:   1 tokens ->     15 (ref     15)  0.10 s
step  5:   1 tokens ->  87606 (ref  87606)  0.10 s
step  6:   1 tokens ->     14 (ref     14)  0.11 s
step  7:   1 tokens ->  12919 (ref  12919)  0.09 s
step  8:   1 tokens ->   3234 (ref   3234)  0.09 s
