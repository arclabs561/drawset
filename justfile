default:
    @just --list

check:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
    just test

test:
    if command -v cargo-nextest >/dev/null 2>&1; then cargo nextest run --workspace && cargo test --workspace --doc; else cargo test --workspace; fi

fmt:
    cargo fmt --all
