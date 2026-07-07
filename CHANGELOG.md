# Changelog

## [0.1.0] - 2026-07-07

### Changed

- Renamed the crate from `kuji` to `drawset`.
- Narrowed the public scope to sampling and subset-selection primitives.

### Removed

- Removed `tconorm`, `tnorm`, and related fuzzy-logic aggregation exports.

## [0.1.10] - 2026-06-10

### Changed

- Documented O(1/k) convergence for `kernel_thin` and `kernel_herd`; cross-referenced rkhs for point-level MMD in thinning docs.
