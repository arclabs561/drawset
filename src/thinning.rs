//! Kernel thinning: greedy coreset selection minimizing kernel discrepancy.
//!
//! Selects a subset S of k points from n candidates that best represents the
//! full set, as measured by Maximum Mean Discrepancy (MMD).
//!
//! Two algorithms:
//! - [`kernel_thin`]: Greedy MMD minimization. Picks the point that most reduces
//!   MMD^2(S, X) at each step.
//! - [`kernel_herd`]: Kernel herding (Chen, Welling & Smola 2010). Greedily
//!   matches the empirical kernel mean embedding, achieving O(1/k) MMD
//!   convergence vs O(1/sqrt(k)) for iid sampling.

/// Greedy kernel thinning via MMD minimization.
///
/// At each step, adds the point from X \ S that minimizes MMD^2(S union {x}, X).
///
/// # Arguments
///
/// * `gram` - n x n kernel Gram matrix in row-major order. Must be symmetric positive semi-definite.
/// * `n` - Number of candidate points.
/// * `k` - Number of points to select (must be <= n).
///
/// # Returns
///
/// Indices of the k selected points.
///
/// # Complexity
///
/// O(nk) time, O(n) auxiliary space.
pub fn kernel_thin(gram: &[f64], n: usize, k: usize) -> Vec<usize> {
    assert!(k <= n, "k ({k}) must be <= n ({n})");
    assert_eq!(gram.len(), n * n, "gram must be n*n");

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

    // Running sums for the MMD^2 incremental update.
    // We track: sum_within = sum_{i,j in S} K[i,j]
    //           sum_cross[c] = sum_{i in S} K[i,c] for each candidate c
    // MMD^2(S, X) = sum_within/|S|^2 - 2 * sum_{i in S} col_mean[i] / |S| + const
    //
    // Actually, it's simpler to track the objective directly.
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

    // sum_cross[c] = sum_{i in S} K[i,c]
    let mut sum_cross = vec![0.0; n];
    let mut sum_within = 0.0;
    let mut cross_mean_sum = 0.0; // sum_{i in S} col_mean[i]

    for step in 0..k {
        let s = step; // current |S|
        let s_new = (s + 1) as f64;
        let s_new_sq = s_new * s_new;

        let mut best_idx = usize::MAX;
        let mut best_obj = f64::INFINITY;

        for c in 0..n {
            if in_set[c] {
                continue;
            }

            let new_within = sum_within + 2.0 * sum_cross[c] + gram[c * n + c];
            let new_cross_mean = cross_mean_sum + col_mean[c];

            // Objective: within_term - 2 * cross_term (ignoring constant)
            let obj = new_within / s_new_sq - 2.0 * new_cross_mean / s_new;

            if obj < best_obj {
                best_obj = obj;
                best_idx = c;
            }
        }

        // Update state
        selected.push(best_idx);
        in_set[best_idx] = true;

        // Update sum_cross for all candidates
        for c in 0..n {
            sum_cross[c] += gram[best_idx * n + c];
        }
        sum_within += 2.0 * (sum_cross[best_idx] - gram[best_idx * n + best_idx])
            + gram[best_idx * n + best_idx];
        // After updating sum_cross, sum_cross[best_idx] already includes K[best_idx, best_idx].
        // sum_within should be: old_sum_within + 2*old_sum_cross[best_idx] + K[best_idx,best_idx]
        // But we updated sum_cross first, so sum_cross[best_idx] = old + K[best_idx,best_idx].
        // Fix: compute sum_within before updating sum_cross.
        // Let me restructure.

        cross_mean_sum += col_mean[best_idx];
    }

    // The sum_within tracking above has a bug from ordering. Let me rewrite cleanly.
    // Actually, let me just redo the whole loop correctly.
    selected.clear();
    in_set.fill(false);

    let mut sum_cross = vec![0.0; n];
    let mut sum_within = 0.0;
    let mut cross_mean_sum = 0.0;

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

            let obj = new_within / s_new_sq - 2.0 * new_cross_mean / s_new;

            if obj < best_obj {
                best_obj = obj;
                best_idx = c;
            }
        }

        selected.push(best_idx);
        in_set[best_idx] = true;

        // Update sum_within BEFORE updating sum_cross
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
/// At each step, picks the point whose kernel evaluation most reduces the
/// residual between the empirical mean embedding and the subset mean embedding.
/// Produces points with O(1/k) MMD convergence, compared to O(1/sqrt(k)) for
/// iid sampling.
///
/// Unlike [`kernel_thin`], herding allows selecting the same point multiple
/// times (with replacement), which is useful when k > n or when the optimal
/// coreset has repeated points. The returned indices may contain duplicates.
///
/// # Arguments
///
/// * `gram` - n x n kernel Gram matrix in row-major order.
/// * `n` - Number of candidate points.
/// * `k` - Number of points to select (can be > n due to replacement).
///
/// # Returns
///
/// Indices of the k selected points (may contain duplicates).
///
/// # References
///
/// Chen, Welling & Smola (2010). "Super-Samples from Kernel Herding."
pub fn kernel_herd(gram: &[f64], n: usize, k: usize) -> Vec<usize> {
    assert!(n > 0, "n must be > 0");
    assert_eq!(gram.len(), n * n, "gram must be n*n");

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
pub fn mmd_sq_from_gram(gram: &[f64], n: usize, subset: &[usize]) -> f64 {
    let m = subset.len();
    if m == 0 {
        return 0.0;
    }

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
