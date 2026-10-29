# ~/venvs/ds-ref/bin/python ref/check_topk_ties.py data/golden/readme20k-s data/ds/readme20k-topk.bin
# commit ddf9b43 + uncommitted gate tools; sclk 2: 1204Mhz *; 2026-10-29
# data/golden/readme20k-s: prompt 20000 tokens; the reference's picks are its own top-k on every row: yes
layer  rows  jaccard extra/row masked rank p50 rank p99 rank max  gap p50  gap p99  gap max
    2    40    0.907      24.9      0      547      738      890  1.3e-03  9.4e-03  1.3e-02
    8    40    0.936      16.9      0      536      655      972  3.0e-03  1.9e-02  3.8e-02
   14    40    0.941      15.6      0      533      672      776  4.7e-03  3.1e-02  5.6e-02
   20    40    0.923      20.4      0      540      692      996  4.5e-03  2.5e-02  4.2e-02
   24    40    0.853      40.8     92      566      946     1428  2.9e-03  1.8e-02  3.2e-02
   28    40    0.919      21.6      8      543      704      822  4.3e-03  2.2e-02  3.6e-02
   32    40    0.914      23.1     10      544      731      997  6.0e-03  3.5e-02  6.0e-02
   36    40    0.925      20.1     23      538      754      902  7.1e-03  4.8e-02  7.1e-02
