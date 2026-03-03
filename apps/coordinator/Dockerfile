FROM rust:1.85-bookworm AS builder

WORKDIR /workspace
ENV CARGO_TARGET_DIR=/workspace/target

COPY apps/coordinator/Cargo.toml apps/coordinator/Cargo.toml
COPY apps/coordinator/src apps/coordinator/src

RUN cargo build --manifest-path apps/coordinator/Cargo.toml --release

FROM debian:bookworm-slim

RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /workspace/target/release/coordinator /usr/local/bin/coordinator

USER appuser
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/coordinator"]
