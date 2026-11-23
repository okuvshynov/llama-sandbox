# scripts/: the tools still in use

Since README step 130 this directory holds only what a next step would start. A script written
for one step lives with that step's results (`results/<topic>/`, or `results/` beside a single
results file); docs/reproduce.md names each by its path.

| | |
|---|---|
| `check.py` | the checks as fixed runs with one verdict each: `check.py unit` (no model files, 30 s; step 186), `check.py default` (unit and the logits against stored hashes, 105 s; step 187; its outputs part, `speed` and `complete` to come) (docs/testing.md section 7) |
| `run_matrix.py` | start prompts x configurations x repeats from a matrix file, one `--record` file per run (step 129); `--keep-out` also keeps each output (step 138) |
| `records.py` | read records: groups, the spread of repeats, a comparison, a condition (step 128) |
| `chat.sh` | an interactive chat with the fastest single-machine settings |
| `serve.sh` | the HTTP server with the same settings, `--cache-wait` and a context of 262,144 tokens (`MAX_CTX`), and all its logs (the request log, dumps, `/metrics` every minute) in a new directory per start, for a client such as pi (step 167) |
| `spec-measure.sh`, `spec_analysis.py` | the held-out prompts with speculation and the cache, logged per step, and its analysis; the measurement most steps before 129 used |
| `trace_compare.py`, `trace_critpath.py` | read `--trace` files: two traces side by side, a verify step's critical path |
| `profile_verify.py` | the `--profile` output of verify steps as a per-group timeline |
| `route_stats.py` | read `--log-routing` files (also imported by the simulators under `results/`); writes placement files (`--emit-placement`) |
| `fans_full.sh`, `set_vega_freq.sh` | the Mac Pro after a boot, with sudo: the four fans at full speed; the GPU clock fixed at one level (6) on the four dies. Before any GPU program (QUICKSTART, "What it needs") |

A new measurement: a matrix file next to its results, read with `records.py`; a new script only
for what those two cannot do, and then it is filed with its results from the start.

A results file's `#` header names the command as it was at the header's commit: for a file
older than step 130 that may be `scripts/<name>` for a script that has since moved. Its place
now is in docs/reproduce.md's row, or `git log --follow -- <path>`.

The simulators that need numpy run with the reference's environment (`~/venvs/ds-ref/bin/python`).
