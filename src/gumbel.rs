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
//!
//! Notes:
//! - This module provides `*_with_rng` variants where determinism matters (tests/benches).
//! - Functions that call `rand::rng()` internally are convenience wrappers and are not deterministic
//!   across processes by design. Random variates have finite precision, so the
//!   implemented distribution approximates the continuous law.

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
/// use drawset::gumbel_noise;
///
/// let mut rng = ChaCha8Rng::seed_from_u64(42);
/// let g = gumbel_noise(&mut rng);
/// assert!(g.is_finite());
/// ```
pub fn gumbel_noise<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    // Open01 excludes both endpoints without clipping valid tail draws.
    let u: f64 = rng.sample(rand::distr::Open01);
    -(-u.ln()).ln()
}

/// Returns Gumbel-perturbed scores relative to the largest scaled logit.
///
/// The reference logit has the largest scaled value, so each returned scaled
/// logit difference is non-positive. Representing differences instead of
/// `scale * logit` keeps an extreme finite winner finite for consumers that
/// compare raw scores or subsequently subtract their maximum.
fn gumbel_perturbed_scores<R: Rng + ?Sized>(logits: &[f64], scale: f64, rng: &mut R) -> Vec<f64> {
    debug_assert!(!logits.is_empty());
    let noise: Vec<f64> = (0..logits.len()).map(|_| gumbel_noise(rng)).collect();
    gumbel_perturbed_scores_from_noise(logits, scale, &noise)
}

/// Applies a supplied Gumbel perturbation to scores relative to their maximum.
fn gumbel_perturbed_scores_from_noise(logits: &[f64], scale: f64, noise: &[f64]) -> Vec<f64> {
    debug_assert!(!logits.is_empty());
    debug_assert_eq!(logits.len(), noise.len());

    // Non-finite scales were never a supported input. Keep the previous
    // downstream fallback behavior for them rather than assigning a new law.
    if !scale.is_finite() {
        return logits
            .iter()
            .zip(noise)
            .map(|(&logit, &noise)| noise + scale * logit)
            .collect();
    }

    let reference = if scale > 0.0 {
        logits.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    } else if scale < 0.0 {
        logits.iter().copied().fold(f64::INFINITY, f64::min)
    } else {
        logits[0]
    };

    logits
        .iter()
        .zip(noise)
        .map(|(&logit, &noise)| {
            let difference = logit - reference;
            let scaled_difference = if difference.is_finite() {
                scale * difference
            } else {
                let scaled_logit = scale * logit;
                let scaled_reference = scale * reference;
                if scaled_logit.is_finite() && scaled_reference.is_finite() {
                    scaled_logit - scaled_reference
                } else {
                    f64::NEG_INFINITY
                }
            };
            noise + scaled_difference
        })
        .collect()
}

/// Calculates `scale * logit / temperature` without intermediate overflow.
///
/// `temperature` must be finite and positive. The fast paths preserve ordinary
/// arithmetic where it is representable; the logarithmic fallback is used only
/// when neither multiplication order can represent the finite result.
fn scaled_logit_over_temperature(logit: f64, scale: f64, temperature: f64) -> f64 {
    debug_assert!(temperature.is_finite() && temperature > 0.0);
    debug_assert!(logit.is_finite() && scale.is_finite());

    if logit == 0.0 || scale == 0.0 {
        return 0.0;
    }

    let product = scale * logit;
    if product.is_finite() {
        return product / temperature;
    }

    let coefficient = scale / temperature;
    let normalized = coefficient * logit;
    if coefficient != 0.0 && normalized.is_finite() {
        return normalized;
    }

    let log_magnitude = scale.abs().ln() + logit.abs().ln() - temperature.ln();
    let magnitude = log_magnitude.exp();
    if scale.is_sign_positive() == logit.is_sign_positive() {
        magnitude
    } else {
        -magnitude
    }
}

/// Returns a perturbed f32 score and the noise used to break rounded ties.
///
/// `f32` logits can be too large for adding a unit-scale Gumbel variate to
/// change their representation. Computing in `f64` covers the common case;
/// retaining the noise preserves the categorical law when equal large logits
/// still round to the same `f64` score.
fn gumbel_perturbed_f32<R: Rng + ?Sized>(logit: f32, rng: &mut R) -> (f64, f64) {
    let noise = gumbel_noise(rng);
    (f64::from(logit) + noise, noise)
}

/// Sample an index using the Gumbel-max trick.
///
/// `logits` must contain finite values. This convenience wrapper obtains its
/// own random source; use [`gumbel_topk_sample_with_rng`] with `k = 1` when a
/// caller-controlled RNG is required.
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
/// use drawset::gumbel_max_sample;
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
/// use drawset::gumbel_max_sample;
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
    let mut best = (f64::NEG_INFINITY, f64::NEG_INFINITY);

    for (i, &logit) in logits.iter().enumerate() {
        let score = gumbel_perturbed_f32(logit, &mut rng);
        if score.0 > best.0 || (score.0 == best.0 && score.1 > best.1) {
            best = score;
            best_i = i;
        }
    }

    best_i
}

/// Sample k indices without replacement using the Gumbel-top-k trick.
///
/// Returns indices sorted by decreasing perturbed score (deterministic tie-break by index).
/// The resulting ordered draw follows the Plackett-Luce procedure: at each
/// rank, an item is drawn with probability proportional to `exp(logit)` among
/// the remaining items. Marginal inclusion probabilities for `k > 1` are not
/// simply proportional to `exp(logit_i)`.
/// `logits` must contain finite values.
///
/// # Panics
///
/// Panics if `logits` is empty or if `k == 0` or `k > logits.len()`.
///
/// # Examples
///
/// ```
/// use drawset::gumbel_topk_sample;
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
/// For identical inputs, drawset version, RNG type, and RNG state, this
/// produces the same ordered result. `logits` must contain finite values.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use drawset::gumbel_topk_sample_with_rng;
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

    let mut scored: Vec<(usize, f64, f64)> = Vec::with_capacity(logits.len());
    for (i, &logit) in logits.iter().enumerate() {
        let (score, noise) = gumbel_perturbed_f32(logit, rng);
        scored.push((i, score, noise));
    }

    scored.sort_by(|(i_a, score_a, noise_a), (i_b, score_b, noise_b)| {
        score_b
            .total_cmp(score_a)
            .then_with(|| noise_b.total_cmp(noise_a))
            .then_with(|| i_a.cmp(i_b))
    });

    scored.iter().take(k).map(|(i, _, _)| *i).collect()
}

/// Gumbel-Softmax: differentiable approximation to categorical sampling.
///
/// Returns a soft one-hot vector that approaches a hard one-hot as
/// temperature -> 0.
///
/// For finite logits, finite `scale`, and positive finite `temperature`, the
/// result is finite, non-negative, and sums to one. Extreme finite logits are
/// normalized before temperature scaling. Non-finite logits or `scale` have no
/// defined probabilistic interpretation.
///
/// An empty input returns an empty vector and a singleton returns `[1.0]`.
/// Zero, negative, or non-finite temperatures use a stochastic hard one-hot
/// fallback instead of a continuous relaxation.
///
/// This evaluates the relaxation using plain floating-point values. It does not
/// record an autodiff graph or return gradients. With fixed noise and positive
/// temperature, the underlying softmax formula is differentiable in the logits.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use drawset::gumbel_softmax;
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
        let noisy = gumbel_perturbed_scores(logits, scale, rng);
        let mut best_i = 0usize;
        let mut best = f64::NEG_INFINITY;
        for (i, &s) in noisy.iter().enumerate() {
            if s > best {
                best = s;
                best_i = i;
            }
        }
        let mut out = vec![0.0_f64; n];
        out[best_i] = 1.0;
        return out;
    }

    let noise: Vec<f64> = (0..n).map(|_| gumbel_noise(rng)).collect();
    let raw_scores = gumbel_perturbed_scores_from_noise(logits, scale, &noise);
    let (scores, divide_after_max): (Vec<f64>, bool) =
        if logits.iter().all(|logit| logit.is_finite())
            && raw_scores.iter().all(|score| score.is_finite())
        {
            (raw_scores, true)
        } else if logits.iter().all(|logit| logit.is_finite()) && scale.is_finite() {
            let normalized: Vec<f64> = logits
                .iter()
                .zip(&noise)
                .map(|(&logit, &noise)| {
                    scaled_logit_over_temperature(logit, scale, temperature) + noise / temperature
                })
                .collect();

            if normalized.iter().all(|score| score.is_finite()) {
                (normalized, false)
            } else {
                (raw_scores, true)
            }
        } else {
            (raw_scores, true)
        };
    let max_val = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Softmax
    let mut sum = 0.0;
    let mut probs = Vec::with_capacity(n);
    for val in scores {
        let difference = val - max_val;
        let p = if divide_after_max {
            (difference / temperature).exp()
        } else {
            difference.exp()
        };
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
/// Returns plain floating-point values, without automatic differentiation.
/// For finite scores, finite `scale`, and positive finite `temperature`, every
/// element is finite and the elements sum to `k` within floating-point error.
/// Non-finite scores or `scale` have no defined probabilistic interpretation.
/// Empty inputs and `k = 0` return an empty vector; `k >= scores.len()` returns
/// all ones. Zero, negative, or non-finite temperatures use a stochastic hard
/// k-hot fallback when `0 < k < scores.len()`.
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
/// use drawset::relaxed_topk_gumbel;
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
        let mut scored: Vec<(usize, f64)> = gumbel_perturbed_scores(scores, scale, rng)
            .into_iter()
            .enumerate()
            .collect();
        scored.sort_by(|(i_a, s_a), (i_b, s_b)| s_b.total_cmp(s_a).then_with(|| i_a.cmp(i_b)));
        let mut out = vec![0.0; n];
        for (i, _) in scored.into_iter().take(k) {
            out[i] = 1.0;
        }
        return out;
    }

    // Work in temperature-normalized coordinates when each finite input can
    // be represented there. That avoids losing a high-temperature finite
    // difference merely because the unnormalized product overflows. The raw
    // relative path preserves the low-temperature behavior when normalization
    // itself overflows.
    let noise: Vec<f64> = (0..n).map(|_| gumbel_noise(rng)).collect();
    let raw_scores = gumbel_perturbed_scores_from_noise(scores, scale, &noise);
    let (mut scores_gumbel, normalized_coordinates): (Vec<f64>, bool) =
        if scores.iter().all(|score| score.is_finite())
            && raw_scores.iter().all(|score| score.is_finite())
        {
            (raw_scores, false)
        } else if scores.iter().all(|score| score.is_finite()) && scale.is_finite() {
            let normalized: Vec<f64> = scores
                .iter()
                .zip(&noise)
                .map(|(&score, &noise)| {
                    scaled_logit_over_temperature(score, scale, temperature) + noise / temperature
                })
                .collect();
            if normalized.iter().all(|score| score.is_finite()) {
                (normalized, true)
            } else {
                (raw_scores, false)
            }
        } else {
            (raw_scores, false)
        };

    let eps = 1e-8_f64;
    let mut onehot: Vec<f64> = vec![0.0; n];
    let mut khot: Vec<f64> = vec![0.0; n];

    for _ in 0..k {
        // Mask out previously selected mass: add log(1 - onehot) to logits.
        for (sg, &oh) in scores_gumbel.iter_mut().zip(onehot.iter()) {
            let m = (1.0 - oh).max(eps);
            *sg += if normalized_coordinates {
                m.ln() / temperature
            } else {
                m.ln()
            };
        }

        // Softmax(scores_gumbel / temperature)
        let max_val = scores_gumbel
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0;
        for (oh, &sg) in onehot.iter_mut().zip(scores_gumbel.iter()) {
            let difference = sg - max_val;
            let p = if normalized_coordinates {
                difference.exp()
            } else {
                (difference / temperature).exp()
            };
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
    fn gumbel_max_frequencies_match_categorical_probabilities() {
        let logits = [0.0_f32, 2.0_f32.ln(), 4.0_f32.ln()];
        let expected = [1.0 / 7.0, 2.0 / 7.0, 4.0 / 7.0];
        let trials = 70_000;
        let mut counts = [0usize; 3];
        let mut rng = ChaCha8Rng::seed_from_u64(0xCA7E_60A1);

        for _ in 0..trials {
            let selected = gumbel_topk_sample_with_rng(&logits, 1, &mut rng)[0];
            counts[selected] += 1;
        }

        for (i, (&count, &probability)) in counts.iter().zip(expected.iter()).enumerate() {
            let observed = count as f64 / trials as f64;
            assert!(
                (observed - probability).abs() < 0.01,
                "category {i}: observed={observed:.5}, expected={probability:.5}, counts={counts:?}"
            );
        }
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
    fn gumbel_softmax_extreme_finite_inputs_do_not_fall_back_to_uniform() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xE57E_0001);
        let probabilities = gumbel_softmax(&[1e308, -1e308], 1e-308, 1.0, &mut rng);

        assert_eq!(probabilities, vec![1.0, 0.0]);

        // `scale * logit` overflows before division here, yet the normalized
        // scores are finite (+1 and -1). Check the fixed-noise analytic value
        // rather than accepting a uniform fallback.
        let seed = 0xE57E_0002;
        let mut expected_rng = ChaCha8Rng::seed_from_u64(seed);
        let first_noise = gumbel_noise(&mut expected_rng);
        let second_noise = gumbel_noise(&mut expected_rng);
        let expected = 1.0
            / (1.0 + (-((1.0 + first_noise / f64::MAX) - (-1.0 + second_noise / f64::MAX))).exp());
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let probabilities = gumbel_softmax(&[f64::MAX, -f64::MAX], f64::MAX, 1.0, &mut rng);
        assert!((probabilities[0] - expected).abs() < 1e-15);
        assert!((probabilities[1] - (1.0 - expected)).abs() < 1e-15);

        // The other multiplication order is required when `scale` overflows
        // first but `scale / temperature` is representable.
        let seed = 0xE57E_0003;
        let mut expected_rng = ChaCha8Rng::seed_from_u64(seed);
        let first_noise = gumbel_noise(&mut expected_rng);
        let second_noise = gumbel_noise(&mut expected_rng);
        let expected = 1.0
            / (1.0 + (-((2.0 + first_noise / f64::MAX) - (1.0 + second_noise / f64::MAX))).exp());
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let probabilities = gumbel_softmax(&[2.0, 1.0], f64::MAX, f64::MAX, &mut rng);
        assert!((probabilities[0] - expected).abs() < 1e-15);
        assert!((probabilities[1] - (1.0 - expected)).abs() < 1e-15);

        // For k=1, the relaxed construction has the same first softmax round.
        let seed = 0xE57E_0004;
        let mut expected_rng = ChaCha8Rng::seed_from_u64(seed);
        let first_noise = gumbel_noise(&mut expected_rng);
        let second_noise = gumbel_noise(&mut expected_rng);
        let expected = 1.0
            / (1.0 + (-((1.0 + first_noise / f64::MAX) - (-1.0 + second_noise / f64::MAX))).exp());
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mask = relaxed_topk_gumbel(&[f64::MAX, -f64::MAX], 1, f64::MAX, 1.0, &mut rng);
        assert!((mask[0] - expected).abs() < 1e-15);
        assert!((mask[1] - (1.0 - expected)).abs() < 1e-15);
    }

    #[test]
    fn common_large_logit_offsets_preserve_gumbel_noise() {
        let seed = 0xE57E_0007;
        let mut baseline_rng = ChaCha8Rng::seed_from_u64(seed);
        let baseline = gumbel_softmax(&[0.0, 0.0], 1.0, 1.0, &mut baseline_rng);
        let mut offset_rng = ChaCha8Rng::seed_from_u64(seed);
        let offset = gumbel_softmax(&[1e30, 1e30], 1.0, 1.0, &mut offset_rng);
        assert_eq!(offset, baseline);

        let mut baseline_rng = ChaCha8Rng::seed_from_u64(seed);
        let baseline = relaxed_topk_gumbel(&[0.0, 0.0], 1, 1.0, 1.0, &mut baseline_rng);
        let mut offset_rng = ChaCha8Rng::seed_from_u64(seed);
        let offset = relaxed_topk_gumbel(&[1e30, 1e30], 1, 1.0, 1.0, &mut offset_rng);
        assert_eq!(offset, baseline);
    }

    #[test]
    fn relaxed_topk_extreme_finite_scores_stay_finite() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xE57E_0005);
        let mask = relaxed_topk_gumbel(&[f64::MAX, 0.0, -f64::MAX], 1, 0.5, 2.0, &mut rng);

        assert_eq!(mask, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn gumbel_topk_uses_noise_when_large_equal_f32_logits_round_together() {
        let logits = [1e30_f32, 1e30_f32];
        let mut observed = [false; 2];
        for seed in 0..32 {
            let mut expected_rng = ChaCha8Rng::seed_from_u64(seed);
            let first_noise = gumbel_noise(&mut expected_rng);
            let second_noise = gumbel_noise(&mut expected_rng);
            let expected = if second_noise > first_noise { 1 } else { 0 };
            observed[expected] = true;

            let mut actual_rng = ChaCha8Rng::seed_from_u64(seed);
            let actual = gumbel_topk_sample_with_rng(&logits, 1, &mut actual_rng);
            assert_eq!(actual, vec![expected]);
        }
        assert!(observed.into_iter().all(|selected| selected));
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
    fn relaxed_topk_low_temperature_matches_hard_topk() {
        let scores = [0.3, -0.8, 1.4, 0.1, 2.0];
        let k = 3;
        let seed = 0x70F_60A1;
        let mut relaxed_rng = ChaCha8Rng::seed_from_u64(seed);
        let mut hard_rng = ChaCha8Rng::seed_from_u64(seed);

        let relaxed = relaxed_topk_gumbel(&scores, k, 1e-6, 1.0, &mut relaxed_rng);
        let mut perturbed: Vec<_> = scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score + gumbel_noise(&mut hard_rng)))
            .collect();
        perturbed.sort_by(|(i_a, a), (i_b, b)| b.total_cmp(a).then_with(|| i_a.cmp(i_b)));
        let mut hard = vec![0.0; scores.len()];
        for (i, _) in perturbed.into_iter().take(k) {
            hard[i] = 1.0;
        }

        let sum: f64 = relaxed.iter().sum();
        assert!((sum - k as f64).abs() < 1e-12, "sum={sum}");
        for (i, (&actual, &expected)) in relaxed.iter().zip(hard.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-8,
                "index {i}: relaxed={actual}, hard={expected}"
            );
        }
    }

    #[test]
    fn gumbel_softmax_jacobian_matches_finite_differences() {
        let logits = [-0.7, 0.2, 1.1];
        let temperature = 0.8;
        let scale = 1.3;
        let seed = 0x5A17_0001;
        let mut base_rng = ChaCha8Rng::seed_from_u64(seed);
        let probabilities = gumbel_softmax(&logits, temperature, scale, &mut base_rng);
        let step = 1e-6;

        for column in 0..logits.len() {
            let mut plus = logits;
            let mut minus = logits;
            plus[column] += step;
            minus[column] -= step;
            let mut plus_rng = ChaCha8Rng::seed_from_u64(seed);
            let mut minus_rng = ChaCha8Rng::seed_from_u64(seed);
            let plus_probs = gumbel_softmax(&plus, temperature, scale, &mut plus_rng);
            let minus_probs = gumbel_softmax(&minus, temperature, scale, &mut minus_rng);

            for row in 0..logits.len() {
                let finite_difference = (plus_probs[row] - minus_probs[row]) / (2.0 * step);
                let kronecker = if row == column { 1.0 } else { 0.0 };
                let analytic =
                    scale / temperature * probabilities[row] * (kronecker - probabilities[column]);
                assert!(
                    (finite_difference - analytic).abs() < 1e-8,
                    "jacobian[{row},{column}]: finite={finite_difference}, analytic={analytic}"
                );
            }
        }
    }

    #[test]
    fn relaxed_topk_finite_difference_jacobian_preserves_total_mass() {
        let scores = [-1.2, -0.1, 0.8, 1.7];
        let seed = 0x5A17_0002;
        let step = 1e-6;

        for column in 0..scores.len() {
            let mut plus = scores;
            let mut minus = scores;
            plus[column] += step;
            minus[column] -= step;
            let mut plus_rng = ChaCha8Rng::seed_from_u64(seed);
            let mut minus_rng = ChaCha8Rng::seed_from_u64(seed);
            let plus_mask = relaxed_topk_gumbel(&plus, 2, 0.9, 1.0, &mut plus_rng);
            let minus_mask = relaxed_topk_gumbel(&minus, 2, 0.9, 1.0, &mut minus_rng);
            let derivative_sum: f64 = plus_mask
                .iter()
                .zip(minus_mask.iter())
                .map(|(plus, minus)| (plus - minus) / (2.0 * step))
                .sum();

            assert!(derivative_sum.is_finite());
            assert!(
                derivative_sum.abs() < 1e-8,
                "Jacobian column {column} changes total mass by {derivative_sum}"
            );
        }
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
