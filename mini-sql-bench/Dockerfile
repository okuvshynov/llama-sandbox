# Sandbox image for mini-sql-bench.
#
# Debian (not Alpine) on purpose: Alpine's busybox `timeout` leaks zombie processes
# under the mini-swe-agent `sh -c "timeout ..."` wrapper and can saturate the pid cgroup
# (see this repo's CLAUDE.md). Debian ships GNU coreutils, so we sidestep that trap.
#
# The SQLite database is baked at build time so the task is fully deterministic and the
# container needs no network (`--network none`) to run.
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends sqlite3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# mini-swe-agent's docker run_args use `--user agent`; the user must exist in the image.
RUN useradd -m -u 1000 agent

WORKDIR /workspace

COPY seed.sql /tmp/seed.sql
RUN sqlite3 /workspace/company.db < /tmp/seed.sql \
    && chown -R agent:agent /workspace

USER agent
