//! Reservoir sampling.
//!
//! Maintains a uniform sample of size `k` from a stream of unknown length.
//!
//! Uses **Algorithm L** (Li, 1994) for optimal performance.
//! Instead of generating a random number for every item (Algorithm R),
//! we compute the number of items to *skip* before the next replacement.
//! This reduces complexity from O(N) RNG calls to O(k * (1 + log(N/k))).
//!
//! ## References
//!
//! - Vitter (1985): reservoir sampling “Algorithm R”.
//! - Li (1994): reservoir sampling “Algorithm L” (skip-based).
//! - Efraimidis & Spirakis (2006): weighted reservoir sampling (A-Res).
//! - Meligrana & Fazzone (2024): “Weighted Reservoir Sampling With Replacement from
//!   Data Streams” -- extends A-Res to with-replacement sampling (potential future variant)
//! - Pettie & Wang (2024): “Universal Perfect Samplers for Incremental Streams” -- exact
//!   G-samplers using O(1) space for general moment functions
//!
//! Notes:
//! - This module provides `*_with_rng` entrypoints for deterministic testing/benchmarking.

use rand::prelude::*;

/// A reservoir sampler that maintains a uniform sample of size `k` from a stream.
///
/// Uses **Algorithm L** (Li, 1994) for O(k(1 + log(N/k))) complexity.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use kuji::ReservoirSampler;
///
/// let mut sampler = ReservoirSampler::new(3);
/// let mut rng = ChaCha8Rng::seed_from_u64(42);
/// for i in 0..100 {
///     sampler.add_with_rng(i, &mut rng);
/// }
/// assert_eq!(sampler.samples().len(), 3);
/// assert_eq!(sampler.seen(), 100);
/// ```
#[derive(Debug, Clone)]
pub struct ReservoirSampler<T> {
    k: usize,
    seen: usize,
    samples: Vec<T>,
    skip_counter: usize,
    w: f64,
}

impl<T> ReservoirSampler<T> {
    /// Create a new sampler that keeps at most `k` samples.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            seen: 0,
            samples: Vec::with_capacity(k),
            skip_counter: 0,
            w: 0.0, // Initialized when reservoir fills
        }
    }

    /// Add an item from the stream.
    ///
    /// If `k == 0`, this discards all items.
    ///
    /// # Examples
    ///
    /// ```
    /// use kuji::ReservoirSampler;
    ///
    /// let mut sampler = ReservoirSampler::new(2);
    /// sampler.add("alpha");
    /// sampler.add("beta");
    /// sampler.add("gamma");
    /// assert!(sampler.samples().len() <= 2);
    /// ```
    #[inline]
    pub fn add(&mut self, item: T) {
        let mut rng = rand::rng();
        self.add_with_rng(item, &mut rng);
    }

    /// Add an item from the stream, using a caller-supplied RNG.
    ///
    /// This exists primarily for deterministic testing/benchmarking.
    ///
    /// # Examples
    ///
    /// ```
    /// use rand::SeedableRng;
    /// use rand_chacha::ChaCha8Rng;
    /// use kuji::ReservoirSampler;
    ///
    /// let mut sampler = ReservoirSampler::new(5);
    /// let mut rng = ChaCha8Rng::seed_from_u64(0);
    /// for i in 0..50 {
    ///     sampler.add_with_rng(i, &mut rng);
    /// }
    /// assert_eq!(sampler.samples().len(), 5);
    /// ```
    #[inline]
    pub fn add_with_rng<R: Rng + ?Sized>(&mut self, item: T, rng: &mut R) {
        self.seen += 1;

        if self.k == 0 {
            return;
        }

        // Phase 1: Filling the reservoir
        if self.samples.len() < self.k {
            self.samples.push(item);

            if self.samples.len() == self.k {
                // Initial weight for Algorithm L: W = exp(log(u) / k)
                // Clamp away from 0 so W ∈ (0,1) and log(1 - W) stays finite/negative.
                let u = rng.random::<f64>().max(f64::MIN_POSITIVE);
                self.w = (u.ln() / self.k as f64).exp();
                self.update_skip(rng);
            }
            return;
        }

        // Phase 2: Algorithm L (skip items)
        if self.skip_counter > 0 {
            self.skip_counter -= 1;
            return;
        }

        // Skip counter hit 0: Replace an item
        // Index to replace is uniform(0, k)
        let replace_idx = rng.random_range(0..self.k);
        self.samples[replace_idx] = item;

        // Update W and calculate new skip
        let u = rng.random::<f64>().max(f64::MIN_POSITIVE);
        self.w *= (u.ln() / self.k as f64).exp();
        self.update_skip(rng);
    }

    /// Update the skip counter using Li's formula.
    ///
    /// Formula from Li (1994):
    /// S = floor(log(U) / log(1 - W))
    /// where U ~ Uniform(0,1) and W is the weight parameter.
    fn update_skip<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let u = rng.random::<f64>();
        // Avoid log(0) if w=1 (unlikely) or u=0
        let denom = (1.0 - self.w).max(1e-10).ln();
        let num = u.max(1e-10).ln();
        let skip = (num / denom).floor();
        self.skip_counter = skip as usize;
    }

    /// Get the current sample (size ≤ k).
    pub fn samples(&self) -> &[T] {
        &self.samples
    }

    /// Number of items observed so far.
    pub fn seen(&self) -> usize {
        self.seen
    }
}

/// A reservoir sampler using **Algorithm R** (Vitter, 1985).
///
/// This is the classic O(N) baseline. It is useful as a correctness reference
/// and for comparisons against Algorithm L.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use kuji::ReservoirSamplerR;
///
/// let mut sampler = ReservoirSamplerR::new(3);
/// let mut rng = ChaCha8Rng::seed_from_u64(42);
/// for i in 0..100 {
///     sampler.add_with_rng(i, &mut rng);
/// }
/// assert_eq!(sampler.samples().len(), 3);
/// assert_eq!(sampler.seen(), 100);
/// ```
#[derive(Debug, Clone)]
pub struct ReservoirSamplerR<T> {
    k: usize,
    seen: usize,
    samples: Vec<T>,
}

impl<T> ReservoirSamplerR<T> {
    /// Create a new sampler that keeps at most `k` samples.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            seen: 0,
            samples: Vec::with_capacity(k),
        }
    }

    /// Add an item from the stream.
    #[inline]
    pub fn add(&mut self, item: T) {
        let mut rng = rand::rng();
        self.add_with_rng(item, &mut rng);
    }

    /// Add an item from the stream, using a caller-supplied RNG.
    #[inline]
    pub fn add_with_rng<R: Rng + ?Sized>(&mut self, item: T, rng: &mut R) {
        self.seen += 1;

        if self.k == 0 {
            return;
        }

        if self.samples.len() < self.k {
            self.samples.push(item);
            return;
        }

        // Algorithm R: replace with probability k / seen.
        let j = rng.random_range(0..self.seen);
        if j < self.k {
            self.samples[j] = item;
        }
    }

    /// Get the current sample (size ≤ k).
    pub fn samples(&self) -> &[T] {
        &self.samples
    }

    /// Number of items observed so far.
    pub fn seen(&self) -> usize {
        self.seen
    }
}

/// Errors for weighted reservoir sampling.
#[derive(Debug, Clone, PartialEq)]
pub enum WeightedReservoirError {
    /// Weight is not finite (NaN/inf).
    NonFiniteWeight(f64),
    /// Weight is non-positive.
    NonPositiveWeight(f64),
}

impl std::fmt::Display for WeightedReservoirError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteWeight(w) => write!(f, "weight must be finite (got {w})"),
            Self::NonPositiveWeight(w) => write!(f, "weight must be > 0 (got {w})"),
        }
    }
}

impl std::error::Error for WeightedReservoirError {}

/// A weighted reservoir sampler (Efraimidis-Spirakis, A-Res).
///
/// Each item with weight `w_i` gets a key `u^(1/w_i)` where `u ~ Uniform(0,1)`.
/// Keep the top-k keys.
///
/// # Examples
///
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use kuji::WeightedReservoirSampler;
///
/// let mut sampler = WeightedReservoirSampler::new(2);
/// let mut rng = ChaCha8Rng::seed_from_u64(42);
/// sampler.add_with_rng("a", 10.0, &mut rng).unwrap();
/// sampler.add_with_rng("b", 1.0, &mut rng).unwrap();
/// sampler.add_with_rng("c", 1.0, &mut rng).unwrap();
/// assert_eq!(sampler.samples().len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct WeightedReservoirSampler<T> {
    k: usize,
    seen: usize,
    items: Vec<T>,
    keys: Vec<f64>,
    /// Cached index of the minimum key. Avoids O(k) scan on every insertion
    /// after the reservoir is full; only rescanned when the min element is
    /// actually replaced.
    min_idx: usize,
}

impl<T> WeightedReservoirSampler<T> {
    /// Create a new sampler that keeps at most `k` items.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            seen: 0,
            items: Vec::with_capacity(k),
            keys: Vec::with_capacity(k),
            min_idx: 0,
        }
    }

    /// Add a weighted item from the stream.
    #[inline]
    pub fn add(&mut self, item: T, weight: f64) -> Result<(), WeightedReservoirError> {
        let mut rng = rand::rng();
        self.add_with_rng(item, weight, &mut rng)
    }

    /// Add a weighted item using a caller-supplied RNG.
    ///
    /// # Examples
    ///
    /// ```
    /// use rand::SeedableRng;
    /// use rand_chacha::ChaCha8Rng;
    /// use kuji::WeightedReservoirSampler;
    ///
    /// let mut sampler = WeightedReservoirSampler::new(1);
    /// let mut rng = ChaCha8Rng::seed_from_u64(0);
    /// sampler.add_with_rng(42, 5.0, &mut rng).unwrap();
    /// assert_eq!(sampler.samples(), &[42]);
    /// ```
    #[inline]
    pub fn add_with_rng<R: Rng + ?Sized>(
        &mut self,
        item: T,
        weight: f64,
        rng: &mut R,
    ) -> Result<(), WeightedReservoirError> {
        self.seen += 1;

        if self.k == 0 {
            return Ok(());
        }

        if !weight.is_finite() {
            return Err(WeightedReservoirError::NonFiniteWeight(weight));
        }
        if weight <= 0.0 {
            return Err(WeightedReservoirError::NonPositiveWeight(weight));
        }

        let u = rng.random::<f64>().max(f64::MIN_POSITIVE);
        let key = (u.ln() / weight).exp();

        if self.items.len() < self.k {
            // Track min during the fill phase.
            if self.keys.is_empty() || key < self.keys[self.min_idx] {
                self.min_idx = self.items.len();
            }
            self.items.push(item);
            self.keys.push(key);
            return Ok(());
        }

        if key > self.keys[self.min_idx] {
            self.items[self.min_idx] = item;
            self.keys[self.min_idx] = key;
            // Rescan for new min only when an element was replaced.
            self.min_idx = 0;
            for (i, &k_i) in self.keys.iter().enumerate().skip(1) {
                if k_i < self.keys[self.min_idx] {
                    self.min_idx = i;
                }
            }
        }

        Ok(())
    }

    /// Get the current sample (size ≤ k).
    pub fn samples(&self) -> &[T] {
        &self.items
    }

    /// Keys for diagnostics/benchmarking.
    pub fn keys(&self) -> &[f64] {
        &self.keys
    }

    /// Number of items observed so far.
    pub fn seen(&self) -> usize {
        self.seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[derive(Debug, Clone, Default)]
    struct ZeroRng;

    impl rand::RngCore for ZeroRng {
        fn next_u32(&mut self) -> u32 {
            0
        }

        fn next_u64(&mut self) -> u64 {
            0
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            dest.fill(0);
        }
    }

    #[test]
    fn reservoir_keeps_k_items() {
        let mut s = ReservoirSampler::new(5);
        for i in 0..100 {
            s.add(i);
        }
        assert_eq!(s.samples().len(), 5);
        assert_eq!(s.seen(), 100);
    }

    #[test]
    fn reservoir_distribution_uniform() {
        // Deterministic chi-squared smoke test for “looks roughly uniform”.
        //
        // This is not a proof, but it catches egregious bugs (e.g. biased replacement index,
        // broken skip math, off-by-one in stream counting) without being flaky.
        let n = 100;
        let k = 10;
        let trials = 10_000;
        let mut counts = vec![0; n];

        for t in 0..trials {
            let mut s = ReservoirSampler::new(k);
            let mut rng = ChaCha8Rng::seed_from_u64(t as u64);
            for i in 0..n {
                s.add_with_rng(i, &mut rng);
            }
            for &item in s.samples() {
                counts[item] += 1;
            }
        }

        let expected = trials as f64 * (k as f64 / n as f64); // E[count_i]
        let chi2: f64 = counts
            .iter()
            .map(|&c| {
                let diff = c as f64 - expected;
                (diff * diff) / expected
            })
            .sum();

        // df = n-1 = 99; E[chi2] ~ df, Var ~ 2*df.
        // Use a conservative cutoff to avoid false positives.
        assert!(
            chi2 < 250.0,
            "chi2 too large (chi2={chi2:.2}, expected~{}). counts={counts:?}",
            n - 1
        );
    }

    #[test]
    fn reservoir_r_keeps_k_items() {
        let mut s = ReservoirSamplerR::new(5);
        for i in 0..100 {
            s.add(i);
        }
        assert_eq!(s.samples().len(), 5);
        assert_eq!(s.seen(), 100);
    }

    #[test]
    fn reservoir_r_distribution_uniform() {
        let n = 100;
        let k = 10;
        let trials = 5_000;
        let mut counts = vec![0; n];

        for t in 0..trials {
            let mut s = ReservoirSamplerR::new(k);
            let mut rng = ChaCha8Rng::seed_from_u64(t as u64);
            for i in 0..n {
                s.add_with_rng(i, &mut rng);
            }
            for &item in s.samples() {
                counts[item] += 1;
            }
        }

        let expected = trials as f64 * (k as f64 / n as f64);
        let chi2: f64 = counts
            .iter()
            .map(|&c| {
                let diff = c as f64 - expected;
                (diff * diff) / expected
            })
            .sum();

        assert!(
            chi2 < 250.0,
            "chi2 too large (chi2={chi2:.2}, expected~{}). counts={counts:?}",
            n - 1
        );
    }

    #[test]
    fn weighted_reservoir_keeps_k_items() {
        let mut s = WeightedReservoirSampler::new(5);
        for i in 0..100 {
            s.add(i, 1.0).expect("weight ok");
        }
        assert_eq!(s.samples().len(), 5);
        assert_eq!(s.seen(), 100);
        assert_eq!(s.keys().len(), 5);
    }

    #[test]
    fn weighted_reservoir_rejects_bad_weights() {
        let mut s = WeightedReservoirSampler::new(2);
        let err = s.add(1, 0.0).expect_err("zero weight rejected");
        assert_eq!(err, WeightedReservoirError::NonPositiveWeight(0.0));
        let err = s.add(2, f64::NAN).expect_err("nan weight rejected");
        assert!(matches!(err, WeightedReservoirError::NonFiniteWeight(w) if !w.is_finite()));
    }

    #[test]
    fn weighted_reservoir_biases_toward_large_weights() {
        let n_trials = 2_000;
        let mut counts = [0usize; 3];

        for t in 0..n_trials {
            let mut s = WeightedReservoirSampler::new(1);
            let mut rng = ChaCha8Rng::seed_from_u64(t as u64);
            s.add_with_rng(0, 100.0, &mut rng).expect("weight ok");
            s.add_with_rng(1, 1.0, &mut rng).expect("weight ok");
            s.add_with_rng(2, 1.0, &mut rng).expect("weight ok");
            let sample = s.samples()[0];
            counts[sample] += 1;
        }

        assert!(counts[0] > counts[1]);
        assert!(counts[0] > counts[2]);
    }

    #[test]
    fn reservoir_algorithm_l_handles_zero_rng_draws() {
        // Edge-case: some RNGs (or mocked RNGs) can yield an all-zero stream.
        // The implementation should not hit ln(0) or log(1 - W)=0 paths.
        let mut s = ReservoirSampler::new(5);
        let mut rng = ZeroRng;
        for i in 0..100 {
            s.add_with_rng(i, &mut rng);
        }
        assert_eq!(s.samples().len(), 5);
        assert_eq!(s.seen(), 100);
    }

    // --- edge case tests ---

    #[test]
    fn reservoir_k_zero_discards_everything() {
        let mut s = ReservoirSampler::new(0);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for i in 0..100 {
            s.add_with_rng(i, &mut rng);
        }
        assert!(s.samples().is_empty());
        assert_eq!(s.seen(), 100);
    }

    #[test]
    fn reservoir_k_larger_than_stream_returns_all() {
        let mut s = ReservoirSampler::new(50);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for i in 0..10 {
            s.add_with_rng(i, &mut rng);
        }
        assert_eq!(s.samples().len(), 10);
        let mut sorted: Vec<_> = s.samples().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn reservoir_r_k_zero_discards_everything() {
        let mut s = ReservoirSamplerR::new(0);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for i in 0..100 {
            s.add_with_rng(i, &mut rng);
        }
        assert!(s.samples().is_empty());
        assert_eq!(s.seen(), 100);
    }

    #[test]
    fn reservoir_r_k_larger_than_stream_returns_all() {
        let mut s = ReservoirSamplerR::new(50);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for i in 0..10 {
            s.add_with_rng(i, &mut rng);
        }
        assert_eq!(s.samples().len(), 10);
        let mut sorted: Vec<_> = s.samples().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn weighted_reservoir_k_zero_discards_everything() {
        let mut s = WeightedReservoirSampler::new(0);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for i in 0..100 {
            s.add_with_rng(i, 1.0, &mut rng).unwrap();
        }
        assert!(s.samples().is_empty());
        assert_eq!(s.seen(), 100);
    }

    #[test]
    fn weighted_reservoir_negative_weight_error() {
        let mut s = WeightedReservoirSampler::new(5);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let err = s.add_with_rng(1, -1.0, &mut rng).unwrap_err();
        assert_eq!(err, WeightedReservoirError::NonPositiveWeight(-1.0));
    }

    #[test]
    fn weighted_reservoir_infinity_weight_error() {
        let mut s = WeightedReservoirSampler::new(5);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let err = s.add_with_rng(1, f64::INFINITY, &mut rng).unwrap_err();
        assert!(matches!(err, WeightedReservoirError::NonFiniteWeight(w) if w.is_infinite()));
    }

    // =========================================================================
    // Property tests
    // =========================================================================

    mod proptests {
        use super::*;
        use proptest::prelude::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        // ---- A-Res cached min_idx is correct after every add ----
        proptest! {
            #[test]
            fn prop_weighted_reservoir_min_idx_correct(
                seed in 0u64..10_000,
                k in 1usize..=10,
                n in 1usize..=50,
                weights in proptest::collection::vec(0.01f64..100.0f64, 1..=50),
            ) {
                let n = n.min(weights.len());
                let mut sampler = WeightedReservoirSampler::new(k);
                let mut rng = ChaCha8Rng::seed_from_u64(seed);

                for (i, &w) in weights.iter().enumerate().take(n) {
                    sampler.add_with_rng(i, w, &mut rng).unwrap();

                    // After each add, if the reservoir has items, min_idx should be correct.
                    if !sampler.keys.is_empty() {
                        let actual_min_idx = sampler.keys.iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| a.total_cmp(b))
                            .unwrap()
                            .0;
                        prop_assert!(
                            sampler.min_idx == actual_min_idx,
                            "After adding item {}: cached min_idx={} but actual min is at {}. keys={:?}",
                            i, sampler.min_idx, actual_min_idx, sampler.keys
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn algorithm_l_and_r_agree_on_size() {
        let k = 10;
        let n = 200;
        let mut s_l = ReservoirSampler::new(k);
        let mut s_r = ReservoirSamplerR::new(k);
        let mut rng_l = ChaCha8Rng::seed_from_u64(99);
        let mut rng_r = ChaCha8Rng::seed_from_u64(99);
        for i in 0..n {
            s_l.add_with_rng(i, &mut rng_l);
            s_r.add_with_rng(i, &mut rng_r);
        }
        assert_eq!(s_l.samples().len(), k);
        assert_eq!(s_r.samples().len(), k);
        assert_eq!(s_l.seen(), n);
        assert_eq!(s_r.seen(), n);
    }
}
