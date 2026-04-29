# CLAUDE.md

## Mistakes

These are the mistakes which were encountered in the past while working on this project. You MUST try hard to avoid these mistakes in the future.

- **Documentation drift during /commit**: When using `/commit`, README.md and TECHNICAL.md were not checked for staleness. Over multiple commits (file format changes, new subcommands, BF16 reference migration, prompt restructuring), the docs fell far behind — referencing old file formats (v3 trace files instead of stats files), old subcommand interfaces (`compare -a -b` instead of `compare -f`), stale prompt counts (12 instead of 19), and Q8 as reference when BF16 was adopted. Fix: during `/commit`, always check if key documentation files (README.md, TECHNICAL.md) reference concepts that were changed in the committed code. If the commit changes file formats, CLI interfaces, model configurations, or experiment methodology, update the docs in the same commit.

- **Piping long-running commands through `tail -N`**: Smoke / sweep commands like `python validation_bench_openai.py ... 2>&1 | tail -20` were used to keep the displayed output short. The downside: while the run is active you can't see streaming progress (turn-by-turn lines); and if something interesting happened in the discarded prefix, you have to rerun the whole thing to recover it. Fix: capture full output by redirecting to a file (or use `run_in_background: true` and read the output file the harness already creates), then `head`, `tail`, or `grep` against the saved file as needed. This preserves both progress visibility and the full transcript for later inspection.
