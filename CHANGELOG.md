# Changelog

## [Unreleased]

## [0.1.2] - 2026-09-12

### Fixed

- Removed a discarded selection pass from `kernel_thin`.
- Corrected kernel selector complexity and removed unsupported convergence claims.
- Corrected weighted sampling and Gumbel-Softmax example explanations.
- Preserved weighted reservoir priority ordering for subnormal positive weights.
- Removed arbitrary uniform-draw tail cutoffs and stabilized Algorithm L's skip logarithm.
- Stabilized Gumbel sampling and relaxations for extreme finite inputs.
- Validated kernel matrix shapes and finite entries before selection.

### Changed

- Reorganized the README around sampling tasks and input requirements.
- Reduced the streaming example to a single pass; uniformity tests remain in the test suite.
- Documented duplicate neighbor values, seeded call behavior, and weighted `seen()` counting.
- Numerical fixes can change exact seeded samples; public signatures are unchanged.
- Made doctests part of local checks and required successful CI before publication.
- Retired the fixed benchmark plot and its generator; runnable benchmarks remain.
- Included contributor documentation, the examples index, and changelog in the package.

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
