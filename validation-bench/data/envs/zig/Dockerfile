FROM alpine:3.21
# coreutils provides GNU `timeout`; busybox's `timeout` leaks a zombie per
# invocation under the harness's `sh -c "timeout ... ./solution"` pattern,
# which saturates --pids-limit=256 on long corpora (678 toml tests destroy
# scoring partway through a single submission).
RUN apk add --no-cache zig coreutils
ENV ZIG_GLOBAL_CACHE_DIR=/tmp/zig-cache
ENV ZIG_LOCAL_CACHE_DIR=/tmp/zig-local-cache
WORKDIR /work
