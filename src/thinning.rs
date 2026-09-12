//! Greedy kernel selection from a pre-computed Gram matrix.
//!
//! [`kernel_thin`] chooses unique indices by minimizing the biased MMD objective at
//! each step. [`kernel_herd`] chooses indices with replacement by maximizing its
//! current mean-embedding residual.
//!
//! # API levels
//!
//! This module operates on pre-computed Gram matrices (`&[f64]`, row-major).
//! The caller is responsible for computing the kernel matrix from raw points
//! and choosing a kernel function.
//!
//! For example, construct a linear-kernel Gram matrix from scalar points:
//!
//! ```
//! use drawset::{kernel_thin, mmd_sq_from_gram};
//!
//! let points = [-2.0, 0.0, 3.0];
//! let gram: Vec<f64> = points.iter()
//!     .flat_map(|x| points.iter().map(move |y| x * y))
//!     .collect();
//! let selected = kernel_thin(&gram, points.len(), 2);
//! let discrepancy = mmd_sq_from_gram(&gram, points.len(), &selected);
//! assert!(discrepancy >= -1e-12);
//! ```
//!
//! [`mmd_sq_from_gram`] in this module evaluates subset quality from the same
//! Gram matrix representation. Symmetry and positive semidefiniteness are the
//! caller's responsibility; they are not checked. Scale the kernel so sums and
//! products remain representable in `f64`.
//!
//! # References
//!
//! - Chen, Welling & Smola (2010): "Super-Samples from Kernel Herding".
//!
//! Despite its historical name, `kernel_thin` is greedy MMD selection, not the
//! split-and-swap algorithm in Dwivedi & Mackey (2021), "Kernel Thinning".

/// Greedy kernel thinning via MMD minimization.
///
/// Selects a subset S of `k` points from `n` candidates. At each step, adds the point
/// from `X \ S` that minimizes MMD²(S ∪ {x}, X) under the supplied Gram matrix.
///
/// # Arguments
///
/// * `gram` - finite n × n kernel Gram matrix in row-major order. Must be symmetric
///   positive semi-definite, with entries small enough to avoid arithmetic overflow.
/// * `n` - Number of candidate points (rows/cols in `gram`).
/// * `k` - Number of points to select (must be ≤ n). Each point is selected at most once.
///
/// # Returns
///
/// Indices of the `k` selected points, in selection order. Indices are unique (no
/// duplicates). Use [`mmd_sq_from_gram`] to evaluate the quality of the returned subset.
/// Equal objectives choose the lowest available index. `k = 0` returns an empty
/// vector, including when `n = 0`.
///
/// # Complexity
///
/// O(n² + nk) time and O(n) auxiliary space, excluding the dense input matrix.
///
/// # Panics
///
/// Panics for an invalid matrix shape, `k > n`, non-finite entries, or a
/// non-finite selection objective. Symmetry and positive semidefiniteness are
/// not checked.
pub fn kernel_thin(gram: &[f64], n: usize, k: usize) -> Vec<usize> {
    assert!(k <= n, "k ({k}) must be <= n ({n})");
    validate_gram(gram, n);

    if k == 0 {
        return Vec::new();
    }

    // Precompute column means: mean_col[j] = (1/n) sum_i K[i,j]
    // This is the kernel mean embedding evaluated at each point.
    let mut col_mean = vec![0.0; n];
    for j in 0..n {
        let mut s = 0.0;
        for i in 0..n {
            s += gram[i * n + j];
        }
        col_mean[j] = s / n as f64;
    }

    let mut selected = Vec::with_capacity(k);
    let mut in_set = vec![false; n];

    // Running sums for the MMD² incremental update.
    // We track: sum_within = sum_{i,j in S} K[i,j]
    //           sum_cross[c] = sum_{i in S} K[i,c] for each candidate c
    // MMD^2(S, X) = sum_within/|S|^2 - 2 * sum_{i in S} col_mean[i] / |S| + const
    //
    // MMD^2(S, X) = (1/|S|^2) sum_{i,j in S} K[i,j]
    //             - (2/(|S|*n)) sum_{i in S} sum_{j=0..n} K[i,j]
    //             + (1/n^2) sum_{i,j} K[i,j]
    //
    // The last term is constant. The second term simplifies with col_mean:
    //   -2/|S| * sum_{i in S} col_mean[i]
    //
    // For each candidate c not in S, adding c changes:
    //   sum_within' = sum_within + 2 * sum_{i in S} K[i,c] + K[c,c]
    //   |S'| = |S| + 1
    //   cross_sum_new = sum_{i in S} col_mean[i] + col_mean[c]
    //
    // We minimize: sum_within'/(|S|+1)^2 - 2*cross_sum_new/(|S|+1)

    // sum_cross[c] = sum_{i in S} K[i,c].
    let mut sum_cross = vec![0.0; n];
    let mut sum_within = 0.0;
    let mut cross_mean_sum = 0.0; // sum_{i in S} col_mean[i]

    for step in 0..k {
        let s_new = (step + 1) as f64;
        let s_new_sq = s_new * s_new;

        let mut best_idx = usize::MAX;
        let mut best_obj = f64::INFINITY;

        for c in 0..n {
            if in_set[c] {
                continue;
            }

            let new_within = sum_within + 2.0 * sum_cross[c] + gram[c * n + c];
            let new_cross_mean = cross_mean_sum + col_mean[c];

            // Objective: within_term - 2 * cross_term, omitting the shared constant.
            let obj = new_within / s_new_sq - 2.0 * new_cross_mean / s_new;
            assert!(obj.is_finite(), "kernel selection objective overflowed");

            if obj < best_obj {
                best_obj = obj;
                best_idx = c;
            }
        }

        selected.push(best_idx);
        in_set[best_idx] = true;

        // Update `sum_within` before `sum_cross`, whose next value includes K[i, i].
        sum_within += 2.0 * sum_cross[best_idx] + gram[best_idx * n + best_idx];
        cross_mean_sum += col_mean[best_idx];

        // Update sum_cross for all candidates
        for c in 0..n {
            sum_cross[c] += gram[best_idx * n + c];
        }
    }

    selected
}

/// Kernel herding: deterministic sampling via greedy mean embedding matching.
///
/// At each step, picks the point with the largest residual between the empirical
/// mean embedding and the selected points' mean embedding.
///
/// Kernel herding is the "greedy matching" complement to [`kernel_thin`]:
/// - [`kernel_thin`] minimizes MMD²(S, X) directly (without replacement).
/// - [`kernel_herd`] matches the kernel mean embedding greedily (with replacement).
///
/// The with-replacement selection permits `k > n`; returned indices may repeat.
///
/// # Arguments
///
/// * `gram` - finite n × n kernel Gram matrix in row-major order. Must be symmetric
///   positive semi-definite, with entries small enough to avoid arithmetic overflow.
/// * `n` - Number of candidate points (rows/cols in `gram`).
/// * `k` - Number of points to select. May exceed `n` (with-replacement).
///
/// # Returns
///
/// Indices of the `k` selected points, in selection order. May contain duplicates
/// when k > n or when the greedy algorithm revisits a point.
/// Equal residuals choose the lowest index. `k = 0` returns an empty vector,
/// but `n` must still be positive.
///
/// # Complexity
///
/// O(n² + nk) time, O(n) auxiliary space, and O(k) output space, excluding the
/// dense input matrix.
///
/// # Panics
///
/// Panics for `n = 0`, an invalid matrix shape, non-finite entries, or a
/// non-finite selection residual. Symmetry and positive semidefiniteness are
/// not checked.
///
/// # References
///
/// Chen, Welling & Smola (2010): "Super-Samples from Kernel Herding."
pub fn kernel_herd(gram: &[f64], n: usize, k: usize) -> Vec<usize> {
    assert!(n > 0, "n must be > 0");
    validate_gram(gram, n);

    if k == 0 {
        return Vec::new();
    }

    // Mean embedding evaluated at each point: mu[j] = (1/n) sum_i K[i,j]
    let mut mu = vec![0.0; n];
    for j in 0..n {
        let mut s = 0.0;
        for i in 0..n {
            s += gram[i * n + j];
        }
        mu[j] = s / n as f64;
    }

    // Residual weight for each point. Herding greedily picks argmax w[j],
    // then updates w[j] -= K[selected, j] / (step+1) ... but the standard
    // formulation tracks cumulative kernel sums.
    //
    // Standard kernel herding:
    //   At step t, pick x_{t} = argmax_j { mu[j] - (1/t) sum_{s<t} K[x_s, j] }
    // which is equivalent to: pick the point that maximizes the residual.

    let mut selected = Vec::with_capacity(k);
    // Running sum: sum_kernel[j] = sum_{s in selected} K[x_s, j]
    let mut sum_kernel = vec![0.0; n];

    for step in 0..k {
        let t = (step + 1) as f64;
        let mut best_idx = 0;
        let mut best_val = f64::NEG_INFINITY;

        for j in 0..n {
            let val = mu[j] - sum_kernel[j] / t;
            assert!(val.is_finite(), "kernel selection residual overflowed");
            if val > best_val {
                best_val = val;
                best_idx = j;
            }
        }

        selected.push(best_idx);

        for j in 0..n {
            sum_kernel[j] += gram[best_idx * n + j];
        }
    }

    selected
}

/// Compute MMD^2 (biased) between a subset and the full set from a Gram matrix.
///
/// Used for evaluating thinning quality. Returns the squared MMD between
/// the subset (indices in `subset`) and the full set (all n points).
///
/// Repeated indices carry repeated empirical mass. For compatibility, an empty
/// subset returns `0.0` without inspecting the matrix; this is a sentinel, not
/// the discrepancy of an empty probability distribution. Roundoff can produce
/// a small negative result even for a positive semidefinite matrix.
///
/// Requires a finite symmetric positive semidefinite matrix whose intermediate
/// sums fit in `f64`. Symmetry and positive semidefiniteness are not checked.
/// Takes O(n² + mn + m²) time and O(1) auxiliary space, where `m = subset.len()`.
///
/// # Panics
///
/// For a nonempty subset, panics on an invalid matrix shape, non-finite entries,
/// or a subset index outside `0..n`.
pub fn mmd_sq_from_gram(gram: &[f64], n: usize, subset: &[usize]) -> f64 {
    let m = subset.len();
    if m == 0 {
        return 0.0;
    }
    validate_gram(gram, n);
    assert!(subset.iter().all(|&i| i < n), "subset index must be < n");

    let mf = m as f64;
    let nf = n as f64;

    // (1/m^2) sum_{i,j in S} K[i,j]
    let mut kss = 0.0;
    for &i in subset {
        for &j in subset {
            kss += gram[i * n + j];
        }
    }
    kss /= mf * mf;

    // (2/(m*n)) sum_{i in S, j in X} K[i,j]
    let mut ksx = 0.0;
    for &i in subset {
        for j in 0..n {
            ksx += gram[i * n + j];
        }
    }
    ksx = 2.0 * ksx / (mf * nf);

    // (1/n^2) sum_{i,j in X} K[i,j]
    let mut kxx = 0.0;
    for i in 0..n {
        for j in 0..n {
            kxx += gram[i * n + j];
        }
    }
    kxx /= nf * nf;

    kss - ksx + kxx
}

fn validate_gram(gram: &[f64], n: usize) {
    let expected = n.checked_mul(n).expect("n*n overflows usize");
    assert_eq!(gram.len(), expected, "gram must be n*n");
    assert!(
        gram.iter().all(|value| value.is_finite()),
        "gram must be finite"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn simple_gram(n: usize) -> Vec<f64> {
        // RBF-like gram matrix from 1D points [0, 1, ..., n-1]
        let sigma = (n as f64) / 2.0;
        let mut g = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let d = (i as f64 - j as f64).powi(2);
                g[i * n + j] = (-d / (2.0 * sigma * sigma)).exp();
            }
        }
        g
    }

    #[test]
    fn thin_indices_unique_and_bounded() {
        let n = 20;
        let k = 5;
        let gram = simple_gram(n);
        let sel = kernel_thin(&gram, n, k);
        assert_eq!(sel.len(), k);
        for &idx in &sel {
            assert!(idx < n);
        }
        // Check uniqueness
        let mut sorted = sel.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), k);
    }

    #[test]
    fn thin_k_equals_n() {
        let n = 8;
        let gram = simple_gram(n);
        let sel = kernel_thin(&gram, n, n);
        assert_eq!(sel.len(), n);
        let mut sorted = sel.clone();
        sorted.sort();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn thin_k_zero() {
        let gram = simple_gram(5);
        let sel = kernel_thin(&gram, 5, 0);
        assert!(sel.is_empty());
    }

    #[test]
    fn thin_beats_endpoints() {
        // Thinned subset should have lower MMD than taking the first k points
        let n = 30;
        let k = 5;
        let gram = simple_gram(n);

        let thinned = kernel_thin(&gram, n, k);
        let first_k: Vec<usize> = (0..k).collect();

        let mmd_thin = mmd_sq_from_gram(&gram, n, &thinned);
        let mmd_first = mmd_sq_from_gram(&gram, n, &first_k);

        assert!(
            mmd_thin <= mmd_first + 1e-12,
            "thinned MMD^2 ({mmd_thin}) should be <= first-k MMD^2 ({mmd_first})"
        );
    }

    #[test]
    fn thin_k1_picks_closest_to_mean() {
        // With k=1, should pick the point closest to the mean embedding.
        // For symmetric 1D points around center, that's the center point.
        let n = 11; // Points 0..10, center = 5
        let gram = simple_gram(n);
        let sel = kernel_thin(&gram, n, 1);
        assert_eq!(sel.len(), 1);
        // Center point (index 5) should maximize col_mean
        // and minimize the objective
        assert_eq!(sel[0], 5, "k=1 should select the center point (index 5)");
    }

    proptest! {
        #[test]
        fn thin_each_prefix_minimizes_mmd_on_small_psd_grams(
            points in prop::collection::vec(-4.0f64..4.0, 1..8),
            requested_k in 1usize..8,
            linear_kernel in any::<bool>(),
        ) {
            let n = points.len();
            let k = requested_k.min(n);
            let mut gram = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let distance_sq = (points[i] - points[j]).powi(2);
                    gram[i * n + j] = if linear_kernel {
                        points[i] * points[j]
                    } else {
                        (-distance_sq / 2.0).exp()
                    };
                }
            }

            let selected = kernel_thin(&gram, n, k);
            let mut prefix = Vec::with_capacity(k);
            for &chosen in &selected {
                let best_candidate_mmd = (0..n)
                    .filter(|candidate| !prefix.contains(candidate))
                    .map(|candidate| {
                        let mut candidate_prefix = prefix.clone();
                        candidate_prefix.push(candidate);
                        mmd_sq_from_gram(&gram, n, &candidate_prefix)
                    })
                    .fold(f64::INFINITY, f64::min);

                let mut chosen_prefix = prefix.clone();
                chosen_prefix.push(chosen);
                let chosen_mmd = mmd_sq_from_gram(&gram, n, &chosen_prefix);
                prop_assert!(
                    chosen_mmd <= best_candidate_mmd + 1e-10,
                    "selected index {chosen} had MMD² {chosen_mmd}, but the best candidate had {best_candidate_mmd}",
                );
                prefix.push(chosen);
            }
        }
    }

    #[test]
    fn herd_correct_length() {
        let n = 10;
        let k = 7;
        let gram = simple_gram(n);
        let sel = kernel_herd(&gram, n, k);
        assert_eq!(sel.len(), k);
        for &idx in &sel {
            assert!(idx < n);
        }
    }

    #[test]
    fn herd_allows_duplicates_when_needed() {
        // With k > n, herding must reuse points
        let n = 3;
        let k = 6;
        let gram = simple_gram(n);
        let sel = kernel_herd(&gram, n, k);
        assert_eq!(sel.len(), k);
    }

    #[test]
    fn herd_beats_single_point() {
        // Herded subset of k>1 should have lower MMD than repeating one point
        let n = 20;
        let k = 5;
        let gram = simple_gram(n);

        let herded = kernel_herd(&gram, n, k);
        // Use unique indices for MMD comparison
        let mut unique_herded: Vec<usize> = herded.clone();
        unique_herded.sort();
        unique_herded.dedup();

        let single_point = vec![herded[0]];

        if unique_herded.len() > 1 {
            let mmd_herd = mmd_sq_from_gram(&gram, n, &unique_herded);
            let mmd_single = mmd_sq_from_gram(&gram, n, &single_point);
            assert!(
                mmd_herd <= mmd_single + 1e-12,
                "herded MMD^2 ({mmd_herd}) should be <= single-point MMD^2 ({mmd_single})"
            );
        }
    }

    #[test]
    fn mmd_sq_full_set_is_zero() {
        let n = 10;
        let gram = simple_gram(n);
        let all: Vec<usize> = (0..n).collect();
        let mmd = mmd_sq_from_gram(&gram, n, &all);
        assert!(mmd.abs() < 1e-12, "MMD^2(X, X) should be 0, got {mmd}");
    }

    #[test]
    fn ties_follow_index_order_and_herding_keeps_repeated_mass() {
        let gram = [1.0; 9];
        assert_eq!(kernel_thin(&gram, 3, 3), [0, 1, 2]);
        assert_eq!(kernel_herd(&gram, 3, 4), [0, 0, 0, 0]);
        // Linear kernel on [0, 2]: full mean 1, repeated-subset mean 2/3.
        let discrepancy = mmd_sq_from_gram(&[0.0, 0.0, 0.0, 4.0], 2, &[0, 0, 1]);
        assert!((discrepancy - 1.0 / 9.0).abs() < 1e-12);
        assert!(kernel_thin(&[], 0, 0).is_empty());
        assert_eq!(mmd_sq_from_gram(&[], 0, &[]), 0.0);
    }

    #[test]
    #[should_panic(expected = "gram must be finite")]
    fn thinning_rejects_nan_before_selection() {
        kernel_thin(&[f64::NAN], 1, 1);
    }

    #[test]
    #[should_panic(expected = "n*n overflows usize")]
    fn matrix_shape_does_not_wrap() {
        kernel_thin(&[], usize::MAX, 0);
    }

    #[test]
    #[should_panic(expected = "subset index must be < n")]
    fn mmd_rejects_invalid_subset_indices() {
        mmd_sq_from_gram(&[1.0; 4], 2, &[2]);
    }
}
