# Contributing to drawset

Drawset provides sampling and subset-selection primitives.

## Before you start

For non-trivial work (new APIs, features, large refactors), open an issue first to align on scope. Drive-by bug fixes and doc patches don't need an issue.

## Setup

- Use stable Rust for contributor checks. The library supports Rust `1.75`;
  tests and benchmarks may require a newer toolchain for development dependencies.
- Optional: `cargo-nextest` for faster test runs (`cargo install cargo-nextest`).

```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Style

- Direct, lowercase prose in commits. No marketing words ("powerful", "robust", "elegant"). No em-dashes in prose.
- Commit messages: `drawset: short lowercase description`. One commit per logical change.
- `cargo fmt` and `cargo clippy --all-targets --all-features -- -D warnings` must pass before `git add`.

## Testing

- `cargo test --workspace` includes library tests, integration tests, and doctests.
- `just check` runs formatting, Clippy, and tests, using nextest when installed.
  It also checks rustdoc links with all features enabled. Doctests run with
  either test runner.
- Test names should describe the property under test, not the function under test.

## Pull requests

- Keep PRs scoped to one concern.
- Show before/after for behavior changes.
- Link the related issue.
- CI must be green before requesting review.

## License

Dual-licensed under MIT or Apache-2.0 at your option. By contributing you agree your contributions are licensed under both.
