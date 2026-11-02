# the canonical fp4 expert kernel grouped by expert (README step 55): tried, reverted
# bin-hip/bench-fp4-moe -m $M (layer 20, GPU 0) with attempt.patch applied (git apply results/fp4-grouped/attempt.patch on da19a4f)
# commit da19a4f + the patch; sclk 6 (manual, all four cards); 2026-11-02
# chunks of M tokens, each choosing 1..6 entries from a pool of `share` experts; per pair = today's
# kernels (one pass per (token, expert) pair), grouped = pairs sorted by expert on the device, each
# weight block loaded once and applied to the group's tokens 4 at a time. Rows bit-identical in
# every case and every variant.
#
## variant 1: grouped, weights decoded per token (kCanonR 2)
M 2, experts from   6:  8 pairs,  6 distinct: per pair  309.2 us, grouped  277.2 us (-10%); rows same
M 2, experts from  12:  5 pairs,  9 distinct: per pair  217.6 us, grouped  232.7 us (+7%); rows same
M 2, experts from 384:  7 pairs, 12 distinct: per pair  281.9 us, grouped  344.6 us (+22%); rows same
M 4, experts from   6: 11 pairs,  6 distinct: per pair  401.2 us, grouped  365.6 us (-9%); rows same
M 4, experts from  12: 20 pairs, 12 distinct: per pair  724.4 us, grouped  648.8 us (-10%); rows same
M 4, experts from 384: 15 pairs, 23 distinct: per pair  554.6 us, grouped  649.3 us (+17%); rows same
M 6, experts from   6: 17 pairs,  6 distinct: per pair  623.6 us, grouped  530.4 us (-15%); rows same
M 6, experts from  12: 22 pairs, 11 distinct: per pair  796.6 us, grouped  657.7 us (-17%); rows same
M 6, experts from 384: 14 pairs, 36 distinct: per pair  519.1 us, grouped  640.4 us (+23%); rows same
M 8, experts from   6: 30 pairs,  6 distinct: per pair 1070.7 us, grouped  849.3 us (-21%); rows same
M 8, experts from  12: 34 pairs, 12 distinct: per pair 1209.1 us, grouped  996.5 us (-18%); rows same
M 8, experts from 384: 23 pairs, 44 distinct: per pair  827.4 us, grouped 1020.5 us (+23%); rows same
## variant 2: rows per 16-lane group (kCanonR; shares each activation load over more rows), one token, 6 experts
kCanonR 2: 244.5 us per layer (today); 4: 280.6 us; 8: 321.9 us (bits unchanged: 0 of 5120 differ from the CPU)
## variant 3: grouped, each weight block decoded once for the group's tokens (attempt.patch as saved)
M 2, experts from   6:  8 pairs,  6 distinct: per pair  308.5 us, grouped  352.2 us (+14%); rows same
M 2, experts from 384:  7 pairs, 12 distinct: per pair  280.7 us, grouped  415.4 us (+48%); rows same
M 6, experts from   6: 17 pairs,  6 distinct: per pair  620.2 us, grouped  629.8 us (+2%); rows same
M 6, experts from 384: 14 pairs, 36 distinct: per pair  515.8 us, grouped  774.3 us (+50%); rows same
M 8, experts from   6: 30 pairs,  6 distinct: per pair 1065.1 us, grouped 1020.7 us (-4%); rows same
M 8, experts from 384: 23 pairs, 44 distinct: per pair  823.1 us, grouped 1231.9 us (+50%); rows same
