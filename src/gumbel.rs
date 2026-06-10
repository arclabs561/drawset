//! Gumbel-max sampling.
//!
//! Given logits $\ell_i$, the Gumbel-max trick samples:
//!
//! $$
//! \arg\max_i (\ell_i + g_i), \quad g_i \sim \mathrm{Gumbel}(0, 1)
//! $$
//!
//! This produces a categorical sample with probabilities proportional to
//! $\exp(\ell_i)$ (i.e. a softmax distribution) without explicitly
//! computing softmax.
//!
//! The top-k extension ([`gumbel_topk_sample`]) draws a size-k subset
//! without replacement by taking the k largest perturbed logits.
//! This is equivalent to sampling from the **Plackett-Luce** distribution
//! (Yellott, 1977): the ranking of items by perturbed log-scores
//! recovers exactly the Plackett-Luce ranking model.
//!
//! ## References
//!
//! - Jang, Gu, Poole (2017): *Categorical Reparameterization with Gumbel-Softmax*.
//! - Maddison, Mnih, Teh (2017): *The Concrete Distribution*.
//! - Huijben et al. (2022): *A Review of the Gumbel-max Trick and its Extensions for
//!   Discrete Stochasticity in Machine Learning* -- comprehensive taxonomy of Gumbel-max
//!   variants (top-k, straight-through, truncated).
//! - Sander et al. (ICML 2023): *Fast, Differentiable and Sparse Top-k: a Convex Analysis
//!   Perspective* -- convex-analysis alternative to Gumbel-Softmax for differentiable
//!   top-k selection; avoids the temperature-tuning problem.
//! - Huang et al. (ACL 2025): *Gumbel Reranking: Differentiable End-to-End Reranking with
//!   Gumbel-Top-k Sampling for Information Retrieval* -- concrete application of the
//!   Gumbel-top-k trick for differentiable reranking in neural IR pipelines.
//!
//! Notes:
//! - This module provides `*_with_rng` variants where determinism matters (tests/benches).
//! - Functions that call `rand::rng()` internally are convenience wrappers and are not deterministic
//!   across processes by design.

use rand::prelude::*;

/// Generate Gumbel noise: G = -log(-log(U)) where U ~ Uniform(0, 1).
///
/// Used in the Gumbel-Max trick for categorical sampling and Gumbel-Softmax
/// for differentiable sampling.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use kuji::gumbel_noise;
///
/// let mut rng = ChaCha8Rng::seed_from_u64(42);
/// let g = gumbel_noise(&mut rng);
/// assert!(g.is_finite());
/// ```
pub fn gumbel_noise<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u: f64 = rng.random_range(0.0..1.0);
    // Clamp to avoid log(0)
    let u = u.clamp(1e-10, 1.0 - 1e-10);
    -(-u.ln()).ln()
}

/// Sample an index using the Gumbel-max trick.
///
/// # Panics
///
/// Panics if `logits` is empty.
///
/// # Examples
///
/// Sample from a categorical distribution over four classes. Higher
/// logits correspond to higher selection probability, but any index
/// can be drawn.
///
/// ```
/// use kuji::gumbel_max_sample;
///
/// let logits = [0.0_f32, 1.0, 2.0, 3.0];
/// let idx = gumbel_max_sample(&logits);
/// assert!(idx < logits.len());
/// ```
///
/// Repeated draws are stochastic -- the index with the largest logit
/// (here index 3) is most likely, but not guaranteed on any single call:
///
/// ```
/// use kuji::gumbel_max_sample;
///
/// let logits = [0.0_f32, -1.0, 5.0]; // index 2 is strongly favoured
/// let mut counts = [0u32; 3];
/// for _ in 0..200 {
///     counts[gumbel_max_sample(&logits)] += 1;
/// }
/// // Index 2 should win the majority of draws.
/// assert!(counts[2] > counts[0] && counts[2] > counts[1]);
/// ```
pub fn gumbel_max_sample(logits: &[f32]) -> usize {
    assert!(
        !logits.is_empty(),
        "gumbel_max_sample: logits must be non-empty"
    );

    let mut rng = rand::rng();
    let mut best_i = 0usize;
    let mut best = f32::NEG_INFINITY;

    for (i, &logit) in logits.iter().enumerate() {
        let score = logit + gumbel_noise(&mut rng) as f32;
        if score > best {
            best = score;
            best_i = i;
        }
    }

    best_i
}

/// Sample k indices without replacement using the Gumbel-top-k trick.
///
/// Returns indices sorted by decreasing perturbed score (deterministic tie-break by index).
/// The resulting subset is drawn from the Plackett-Luce distribution over size-k subsets,
/// where each item's inclusion probability is proportional to exp(logit_i).
///
/// # Panics
///
/// Panics if `logits` is empty or if `k == 0` or `k > logits.len()`.
///
/// # Examples
///
/// ```
/// use kuji::gumbel_topk_sample;
///
/// let logits = [0.0_f32, 1.0, 2.0, 3.0, 4.0];
/// let indices = gumbel_topk_sample(&logits, 3);
/// assert_eq!(indices.len(), 3);
/// // All indices are valid and unique.
/// for &i in &indices {
///     assert!(i < logits.len());
/// }
/// ```
pub fn gumbel_topk_sample(logits: &[f32], k: usize) -> Vec<usize> {
    let mut rng = rand::rng();
    gumbel_topk_sample_with_rng(logits, k, &mut rng)
}

/// Gumbel-top-k with a caller-supplied RNG (for tests/benchmarks).
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use kuji::gumbel_topk_sample_with_rng;
///
/// let logits = [0.0_f32, 1.0, 2.0, 3.0, 4.0];
/// let mut rng = ChaCha8Rng::seed_from_u64(99);
/// let indices = gumbel_topk_sample_with_rng(&logits, 2, &mut rng);
/// assert_eq!(indices.len(), 2);
/// ```
pub fn gumbel_topk_sample_with_rng<R: Rng + ?Sized>(
    logits: &[f32],
    k: usize,
    rng: &mut R,
) -> Vec<usize> {
    assert!(
        !logits.is_empty(),
        "gumbel_topk_sample: logits must be non-empty"
    );
    assert!(k > 0, "gumbel_topk_sample: k must be > 0");
    assert!(
        k <= logits.len(),
        "gumbel_topk_sample: k must be <= logits.len()"
    );

    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(logits.len());
    for (i, &logit) in logits.iter().enumerate() {
        scored.push((i, logit + gumbel_noise(rng) as f32));
    }

    scored.sort_by(|(i_a, s_a), (i_b, s_b)| s_b.total_cmp(s_a).then_with(|| i_a.cmp(i_b)));

    scored.iter().take(k).map(|(i, _)| *i).collect()
}

/// Gumbel-Softmax: differentiable approximation to categorical sampling.
///
/// Returns a soft one-hot vector that approaches a hard one-hot as
/// temperature -> 0.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use kuji::gumbel_softmax;
///
/// let logits = [1.0_f64, 0.0, -1.0];
/// let mut rng = ChaCha8Rng::seed_from_u64(7);
/// let probs = gumbel_softmax(&logits, 0.7, 1.0, &mut rng);
///
/// // Result is a probability vector: non-negative, sums to 1.
/// assert_eq!(probs.len(), 3);
/// assert!(probs.iter().all(|p| *p >= 0.0 && p.is_finite()));
/// let sum: f64 = probs.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-9);
/// ```
pub fn gumbel_softmax<R: Rng + ?Sized>(
    logits: &[f64],
    temperature: f64,
    scale: f64,
    rng: &mut R,
) -> Vec<f64> {
    let n = logits.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1.0];
    }

    // If temperature is invalid, fall back to a hard (stochastic) one-hot.
    if !temperature.is_finite() || temperature <= 0.0 {
        let mut best_i = 0usize;
        let mut best = f64::NEG_INFINITY;
        for (i, &l) in logits.iter().enumerate() {
            let s = gumbel_noise(rng) + scale * l;
            if s > best {
                best = s;
                best_i = i;
            }
        }
        let mut out = vec![0.0_f64; n];
        out[best_i] = 1.0;
        return out;
    }

    let mut noisy = Vec::with_capacity(n);
    let mut max_val = f64::NEG_INFINITY;

    for &l in logits {
        let val = (gumbel_noise(rng) + scale * l) / temperature;
        if val > max_val {
            max_val = val;
        }
        noisy.push(val);
    }

    // Softmax
    let mut sum = 0.0;
    let mut probs = Vec::with_capacity(n);
    for val in noisy {
        let p = (val - max_val).exp();
        sum += p;
        probs.push(p);
    }

    if !sum.is_finite() || sum <= 0.0 {
        return vec![1.0 / n as f64; n];
    }

    for p in &mut probs {
        *p /= sum;
    }

    probs
}

/// Relaxed Top-K via Gumbel-Softmax.
///
/// Implements the “Relaxed Top-K” / “relaxed k-hot” construction
/// (Kool et al., 2019; Xie & Ermon, 2019):
/// add one Gumbel perturbation, then iteratively apply a masked softmax k times,
/// accumulating a k-hot relaxation (entries sum to approximately k).
///
/// This is different from taking `max` over k independent categorical samples
/// (which does not enforce without-replacement top-k structure).
///
/// ## Algorithm
///
/// 1. Perturb each logit once: `g_i = score_i + Gumbel()`.
/// 2. For each of the k rounds:
///    a. **Soft-unmask**: add `log(1 - onehot_i)` to each perturbed logit.
///    For positions already selected (`onehot_i ~ 1`), this drives the logit
///    toward `-inf`, suppressing re-selection. For unselected positions
///    (`onehot_i ~ 0`), the contribution is `log(1) = 0` (no effect).
///    This is the continuous relaxation of “remove the selected item.”
///    b. **Softmax**: compute `softmax(g / temperature)` to get the current
///    soft one-hot vector.
///    c. **Accumulate**: add the soft one-hot to the running k-hot sum.
///
/// The in-place mutation of `scores_gumbel` is intentional: each round's masking
/// step modifies the perturbed logits so that previously selected elements are
/// progressively suppressed, yielding a without-replacement structure.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use kuji::relaxed_topk_gumbel;
///
/// let scores = [0.1_f64, 0.2, 0.3, 0.4, 0.5];
/// let mut rng = ChaCha8Rng::seed_from_u64(9);
/// let khot = relaxed_topk_gumbel(&scores, 2, 0.8, 1.0, &mut rng);
///
/// // Result has same length as input and entries are non-negative.
/// assert_eq!(khot.len(), 5);
/// assert!(khot.iter().all(|x| *x >= 0.0 && x.is_finite()));
/// // Entries sum to approximately k=2.
/// let sum: f64 = khot.iter().sum();
/// assert!((sum - 2.0).abs() < 1e-6);
/// ```
pub fn relaxed_topk_gumbel<R: Rng + ?Sized>(
    scores: &[f64],
    k: usize,
    temperature: f64,
    scale: f64,
    rng: &mut R,
) -> Vec<f64> {
    let n = scores.len();
    if n == 0 || k == 0 {
        return vec![];
    }
    if k >= n {
        return vec![1.0; n];
    }

    // If temperature is invalid, fall back to a hard k-hot (stochastic) selection.
    if !temperature.is_finite() || temperature <= 0.0 {
        let mut scored: Vec<(usize, f64)> = scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, gumbel_noise(rng) + scale * s))
            .collect();
        scored.sort_by(|(i_a, s_a), (i_b, s_b)| s_b.total_cmp(s_a).then_with(|| i_a.cmp(i_b)));
        let mut out = vec![0.0; n];
        for (i, _) in scored.into_iter().take(k) {
            out[i] = 1.0;
        }
        return out;
    }

    // Base Gumbel perturbation.
    let mut scores_gumbel: Vec<f64> = scores
        .iter()
        .map(|&s| gumbel_noise(rng) + scale * s)
        .collect();

    let eps = 1e-8_f64;
    let mut onehot: Vec<f64> = vec![0.0; n];
    let mut khot: Vec<f64> = vec![0.0; n];

    for _ in 0..k {
        // Mask out previously selected mass: add log(1 - onehot) to logits.
        for (sg, &oh) in scores_gumbel.iter_mut().zip(onehot.iter()) {
            let m = (1.0 - oh).max(eps);
            *sg += m.ln();
        }

        // Softmax(scores_gumbel / temperature)
        let max_val = scores_gumbel
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0;
        for (oh, &sg) in onehot.iter_mut().zip(scores_gumbel.iter()) {
            let p = ((sg - max_val) / temperature).exp();
            *oh = p;
            sum += p;
        }

        if !sum.is_finite() || sum <= 0.0 {
            onehot.fill(1.0 / n as f64);
        } else {
            for oh in &mut onehot {
                *oh /= sum;
            }
        }

        for (k_i, &oh) in khot.iter_mut().zip(onehot.iter()) {
            *k_i += oh;
        }
    }

    khot
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn gumbel_topk_basic_invariants() {
        let logits = [0.0_f32, 1.0, 2.0, 3.0, 4.0];
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let idxs = gumbel_topk_sample_with_rng(&logits, 3, &mut rng);

        assert_eq!(idxs.len(), 3);
        for &i in &idxs {
            assert!(i < logits.len());
        }
        // Without-replacement selection => unique indices.
        let mut sorted = idxs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 3);
    }

    #[test]
    fn gumbel_softmax_is_a_probability_vector() {
        let logits = [1.0_f64, 0.0, -1.0];
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let probs = gumbel_softmax(&logits, 0.7, 1.0, &mut rng);

        assert_eq!(probs.len(), logits.len());
        assert!(probs.iter().all(|p| p.is_finite() && *p >= 0.0));
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={sum}");
    }

    #[test]
    fn relaxed_topk_sums_to_about_k() {
        let scores = [0.1_f64, 0.2, 0.3, 0.4, 0.5];
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let k = 2;
        let khot = relaxed_topk_gumbel(&scores, k, 0.8, 1.0, &mut rng);

        assert_eq!(khot.len(), scores.len());
        assert!(khot.iter().all(|x| x.is_finite() && *x >= 0.0));
        let sum: f64 = khot.iter().sum();
        // It’s a relaxation, not exact k, but should be close-ish for sane temperatures.
        assert!((sum - k as f64).abs() < 1e-6, "sum={sum}");
    }

    #[test]
    fn gumbel_topk_is_deterministic_given_seed() {
        let logits = [0.0_f32, 1.0, 2.0, 3.0, 4.0];
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);

        let a = gumbel_topk_sample_with_rng(&logits, 4, &mut rng1);
        let b = gumbel_topk_sample_with_rng(&logits, 4, &mut rng2);
        assert_eq!(a, b);
    }

    // --- edge case tests ---

    #[test]
    fn gumbel_noise_returns_finite() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for _ in 0..1_000 {
            let g = gumbel_noise(&mut rng);
            assert!(g.is_finite(), "gumbel_noise produced non-finite: {g}");
        }
    }

    #[test]
    fn gumbel_max_sample_single_logit_returns_zero() {
        let idx = gumbel_max_sample(&[42.0_f32]);
        assert_eq!(idx, 0);
    }

    #[test]
    fn gumbel_topk_k_equals_n_returns_permutation() {
        let logits = [0.0_f32, 1.0, 2.0, 3.0, 4.0];
        let mut rng = ChaCha8Rng::seed_from_u64(77);
        let idxs = gumbel_topk_sample_with_rng(&logits, logits.len(), &mut rng);
        assert_eq!(idxs.len(), logits.len());
        let mut sorted = idxs.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            vec![0, 1, 2, 3, 4],
            "k=n must return a permutation of all indices"
        );
    }

    #[test]
    fn gumbel_softmax_empty_logits_returns_empty() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let probs = gumbel_softmax(&[], 1.0, 1.0, &mut rng);
        assert!(probs.is_empty());
    }

    #[test]
    fn gumbel_softmax_single_logit_returns_one() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let probs = gumbel_softmax(&[5.0], 1.0, 1.0, &mut rng);
        assert_eq!(probs, vec![1.0]);
    }

    #[test]
    fn gumbel_softmax_zero_temperature_falls_back_to_hard() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let probs = gumbel_softmax(&[1.0, 2.0, 3.0], 0.0, 1.0, &mut rng);
        assert_eq!(probs.len(), 3);
        // Exactly one entry is 1.0, others are 0.0 (hard one-hot).
        let ones: Vec<_> = probs.iter().filter(|&&p| p == 1.0).collect();
        let zeros: Vec<_> = probs.iter().filter(|&&p| p == 0.0).collect();
        assert_eq!(ones.len(), 1);
        assert_eq!(zeros.len(), 2);
    }

    #[test]
    fn gumbel_softmax_nan_temperature_falls_back_to_hard() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let probs = gumbel_softmax(&[1.0, 2.0, 3.0], f64::NAN, 1.0, &mut rng);
        assert_eq!(probs.len(), 3);
        let ones: Vec<_> = probs.iter().filter(|&&p| p == 1.0).collect();
        let zeros: Vec<_> = probs.iter().filter(|&&p| p == 0.0).collect();
        assert_eq!(ones.len(), 1);
        assert_eq!(zeros.len(), 2);
    }

    #[test]
    fn relaxed_topk_k_zero_returns_empty() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let khot = relaxed_topk_gumbel(&[1.0, 2.0, 3.0], 0, 1.0, 1.0, &mut rng);
        assert!(khot.is_empty());
    }

    #[test]
    fn relaxed_topk_k_ge_n_returns_all_ones() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let khot = relaxed_topk_gumbel(&[1.0, 2.0, 3.0], 3, 1.0, 1.0, &mut rng);
        assert_eq!(khot, vec![1.0; 3]);

        let khot = relaxed_topk_gumbel(&[1.0, 2.0], 5, 1.0, 1.0, &mut rng);
        assert_eq!(khot, vec![1.0; 2]);
    }

    // =========================================================================
    // Property tests
    // =========================================================================

    mod proptests {
        use super::*;
        use proptest::prelude::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // ---- Gumbel softmax at very low temperature concentrates on max logit ----
        // The max logit gets a clear gap of 8.0 over the rest. The flip
        // probability per competitor per trial is sigmoid(-gap) (the Gumbel
        // difference is logistic): at gap 5.0 that was ~0.7%, enough for a
        // rare CI flake across 256 proptest cases x 50 trials x 7
        // competitors; at 8.0 it is ~0.03% and the 0.9 mass bound holds with
        // wide margin.
        proptest! {
            #[test]
            fn prop_gumbel_softmax_low_temp_concentrates(
                seed in 0u64..5_000,
                base_logits in proptest::collection::vec(-5.0f64..5.0f64, 2..=8),
            ) {
                // Create logits where the first element is guaranteed to be
                // the max by adding a large gap to the current max.
                let current_max = base_logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut logits = base_logits;
                logits[0] = current_max + 8.0; // clear gap (see comment above)
                let max_idx = 0;

                // At very low temperature, the mass should be concentrated.
                let n_trials = 50;
                let mut mass_at_max = 0.0;
                for t in 0..n_trials {
                    let mut rng = ChaCha8Rng::seed_from_u64(seed * 1000 + t);
                    let probs = gumbel_softmax(&logits, 0.01, 1.0, &mut rng);
                    mass_at_max += probs[max_idx];
                }
                let avg_mass = mass_at_max / n_trials as f64;

                // With a gap of 8.0 at T=0.01, the dominant logit should
                // capture nearly all mass on average.
                prop_assert!(
                    avg_mass > 0.9,
                    "At low T, max-logit idx={max_idx} got avg mass={avg_mass:.4}, logits={logits:?}"
                );
            }
        }

        // ---- relaxed_topk sum is approximately k ----
        proptest! {
            #[test]
            fn prop_relaxed_topk_sum_is_k(
                seed in 0u64..5_000,
                scores in proptest::collection::vec(-5.0f64..5.0f64, 3..=10),
                k in 1usize..=3,
                temp in 0.1f64..2.0f64,
            ) {
                prop_assume!(k < scores.len());
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                let khot = relaxed_topk_gumbel(&scores, k, temp, 1.0, &mut rng);

                prop_assert_eq!(khot.len(), scores.len());
                prop_assert!(khot.iter().all(|x| x.is_finite() && *x >= 0.0));

                let sum: f64 = khot.iter().sum();
                prop_assert!(
                    (sum - k as f64).abs() < 1e-5,
                    "relaxed_topk sum={sum}, expected {k}"
                );
            }
        }
    }

    #[test]
    fn relaxed_topk_zero_temperature_falls_back_to_hard_khot() {
        let mut rng = ChaCha8Rng::seed_from_u64(55);
        let khot = relaxed_topk_gumbel(&[1.0, 2.0, 3.0, 4.0, 5.0], 2, 0.0, 1.0, &mut rng);
        assert_eq!(khot.len(), 5);
        // Exactly k=2 entries are 1.0, the rest are 0.0.
        let sum: f64 = khot.iter().sum();
        assert!(
            (sum - 2.0).abs() < 1e-12,
            "hard k-hot should sum to exactly k, got {sum}"
        );
        for &x in &khot {
            assert!(
                x == 0.0 || x == 1.0,
                "hard k-hot entry should be 0 or 1, got {x}"
            );
        }
    }
}
