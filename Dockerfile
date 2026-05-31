# syntax=docker/dockerfile:1.7

FROM --platform=linux/amd64 rust:1.88-bookworm AS builder

WORKDIR /app
ENV CARGO_TARGET_DIR=/cargo-target

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY resources ./resources
COPY references/references.json.gz ./references/references.json.gz

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/cargo-target \
    RUSTFLAGS="-C target-cpu=x86-64-v2" \
    cargo build --release --bin rinha --bin lb --bin build-dataset \
    && cp /cargo-target/release/rinha /app/rinha \
    && cp /cargo-target/release/lb /app/lb \
    && cp /cargo-target/release/build-dataset /app/build-dataset

RUN mkdir -p /app/data/index \
    && /app/build-dataset references/references.json.gz /app/data/index

FROM --platform=linux/amd64 debian:bookworm-slim

WORKDIR /app

COPY --from=builder /app/rinha /app/rinha
COPY --from=builder /app/lb /app/lb
COPY --from=builder /app/data/index/ivf-centroids.bin /app/data/index/ivf-centroids.bin
COPY --from=builder /app/data/index/ivf-vectors.bin /app/data/index/ivf-vectors.bin
COPY --from=builder /app/data/index/ivf-labels.bin /app/data/index/ivf-labels.bin
COPY --from=builder /app/data/index/ivf-offsets.bin /app/data/index/ivf-offsets.bin
COPY resources /app/resources

RUN mkdir -p /sockets

ENV PORT=9999
ENV NORMALIZATION_PATH=/app/resources/normalization.json
ENV MCC_RISK_PATH=/app/resources/mcc_risk.json
ENV DATASET_DIR=/app/data/index

EXPOSE 9999
