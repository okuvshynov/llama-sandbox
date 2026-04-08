# CLAUDE.md

## Mistakes

These are the mistakes which were encountered in the past while working on this project. You MUST try hard to avoid these mistakes in the future.

- **Documentation drift during /commit**: When using `/commit`, README.md and TECHNICAL.md were not checked for staleness. Over multiple commits (file format changes, new subcommands, BF16 reference migration, prompt restructuring), the docs fell far behind — referencing old file formats (v3 trace files instead of stats files), old subcommand interfaces (`compare -a -b` instead of `compare -f`), stale prompt counts (12 instead of 19), and Q8 as reference when BF16 was adopted. Fix: during `/commit`, always check if key documentation files (README.md, TECHNICAL.md) reference concepts that were changed in the committed code. If the commit changes file formats, CLI interfaces, model configurations, or experiment methodology, update the docs in the same commit.
