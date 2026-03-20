FROM rust:1.89-bookworm AS builder

WORKDIR /workspace
ENV CARGO_TARGET_DIR=/workspace/target

COPY Cargo.lock apps/coordinator/Cargo.lock
COPY apps/coordinator/Cargo.toml apps/coordinator/Cargo.toml
COPY apps/coordinator/src apps/coordinator/src

RUN cargo build --locked --manifest-path apps/coordinator/Cargo.toml --release

FROM debian:bookworm-slim

RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /workspace/target/release/coordinator /usr/local/bin/coordinator

RUN mkdir -p /app/config && chown -R appuser:appuser /app /opt/lucid

USER appuser
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/coordinator"]
