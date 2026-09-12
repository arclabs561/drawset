# Changelog

## Unreleased

### Fixed

- Removed a discarded selection pass from `kernel_thin`.
- Corrected kernel selector complexity and removed unsupported convergence claims.
- Corrected weighted sampling and Gumbel-Softmax example explanations.

### Changed

- Reorganized the README around sampling tasks and input requirements.
- Reduced the streaming example to a single pass; uniformity tests remain in the test suite.

## [0.1.1] - 2026-07-07

### Changed

- Re-exported quasi-Monte Carlo sequences from `lowdisc`.

## [0.1.0] - 2026-07-07

### Changed

- Renamed the crate from `kuji` to `drawset`.
- Narrowed the public scope to sampling and subset-selection primitives.

### Removed

- Removed `tconorm`, `tnorm`, and related fuzzy-logic aggregation exports.

## kuji [0.1.10] - 2026-06-10

### Changed

- Documented O(1/k) convergence for `kernel_thin` and `kernel_herd`; cross-referenced rkhs for point-level MMD in thinning docs.
