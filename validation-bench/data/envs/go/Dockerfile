FROM golang:1.23-alpine
# coreutils provides GNU `timeout`; busybox's `timeout` (the alpine default
# under golang:*-alpine) leaks a zombie per invocation under the harness's
# `sh -c "timeout ... ./solution"` pattern, saturating --pids-limit=256
# partway through 678-test toml / 350-test yaml corpora. Zig's Dockerfile
# applies the same fix.
RUN apk add --no-cache coreutils
ENV GOCACHE=/tmp/go-build
ENV GOMODCACHE=/tmp/go-mod
ENV GOPATH=/tmp/go
WORKDIR /work
