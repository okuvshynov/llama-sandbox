FROM alpine:3.21
RUN apk add --no-cache zig
ENV ZIG_GLOBAL_CACHE_DIR=/tmp/zig-cache
ENV ZIG_LOCAL_CACHE_DIR=/tmp/zig-local-cache
WORKDIR /work
