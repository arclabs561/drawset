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
//! ## References
//!
//! - Jang, Gu, Poole (2017): *Categorical Reparameterization with Gumbel-Softmax*.
//! - Maddison, Mnih, Teh (2017): *The Concrete Distribution*.
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
///
/// # Panics
///
/// Panics if `logits` is empty or if `k == 0` or `k > logits.len()`.
pub fn gumbel_topk_sample(logits: &[f32], k: usize) -> Vec<usize> {
    let mut rng = rand::rng();
    gumbel_topk_sample_with_rng(logits, k, &mut rng)
}

/// Gumbel-top-k with a caller-supplied RNG (for tests/benchmarks).
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
/// temperature → 0.
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
/// Implements the “Relaxed Top-K” / “relaxed k-hot” construction:
/// add one Gumbel perturbation, then iteratively apply a masked softmax \(k\) times,
/// accumulating a k-hot relaxation (entries sum to approximately \(k\)).
///
/// This is different from taking `max` over `k` independent categorical samples
/// (which does not enforce without-replacement top-k structure).
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
}
