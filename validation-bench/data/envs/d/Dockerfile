FROM debian:bookworm-slim
# `ldc` ships only the D compiler; LDC drives the system linker through `cc`,
# so we need:
#   - gcc       for /usr/bin/cc (the link driver ldc2 invokes)
#   - libc6-dev for libm.so / libpthread.so symlinks ld looks up at link time
# (--no-install-recommends drops the recommends gcc would normally pull in,
# so libc6-dev needs to be explicit; without it ld fails "cannot find -lm").
RUN apt-get update && apt-get install -y --no-install-recommends \
    ldc gcc libc6-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /work
