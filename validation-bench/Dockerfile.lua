FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends lua5.4 && rm -rf /var/lib/apt/lists/*
WORKDIR /work
