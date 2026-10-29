# ~/venvs/ds-ref/bin/python ref/compare_logits.py data/ref/readme20k.pt data/ds/readme20k-rt.f32
# logits from results/rt-readme20k-e2e.txt (commit ddf9b43 + ds-test --max-ctx); sclk 2: 1204Mhz *; 2026-10-29
step   pos     ref   other  top5         KL  max|d|/std
   0     0      28      28     5   1.13e-04       0.816
   1 20000   12919   12919     4   4.56e-02       0.321
   2 20001    3234    3234     4   2.23e-06       0.829
   3 20002   12139   12139     4   6.01e-02       0.307
   4 20003      15      15     4   2.62e-05       0.353
   5 20004   87606   87606     5   4.87e-04       1.293
   6 20005      14      14     5   9.61e-03       2.076
   7 20006   12919   12919     4   7.75e-02       0.390
   8 20007    3234    3234     4   1.14e-06       1.173
# argmax agrees 9/9, mean KL 2.15e-02, max KL 7.75e-02
