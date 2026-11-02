# GPU memory per die after each startup phase (README step 60): ledger.csv (GB per phase, GPU and --max-ctx), slots.csv
# (expert slots per GPU); from the runtime's startup ledger (hipMemGetInfo after each phase; "free" = what is left)
# bin-hip/ds-completion -m $M --rt $F8 --placement placements/study30.txt --dspark $D --spec --canon --cache-loads 800 --cache-period 8 --max-ctx {4096,16384,65536} -p Hi --chat -n 1
# commit 8afb53f + the ledger; 2026-11-02; expert slot = 18.8 MB (gate + up + down in mxfp4); totals 34.34 GB per GPU (32 GiB)
