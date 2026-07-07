default:
    @just --list

check:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings
    if command -v cargo-nextest >/dev/null 2>&1; then cargo nextest run --workspace; else cargo test --workspace; fi

test:
    if command -v cargo-nextest >/dev/null 2>&1; then cargo nextest run --workspace; else cargo test --workspace; fi

fmt:
    cargo fmt --all
